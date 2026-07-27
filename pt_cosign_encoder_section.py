# pyright: reportUndefinedVariable=false
"""Detailed pseudocode for the body-part-aware CoSign encoder."""


def construct_part_graph(joint_layout):
    """
    Build a normalized distance-partitioned graph.

    One adjacency partition represents self-connections and the other
    represents one-hop anatomical connections.
    """
    edges = self_connections(joint_layout) + anatomical_edges(joint_layout)
    hop_distance = shortest_hop_distance(edges, maximum_hop=1)
    adjacency = normalize_directed_graph(edges)

    return stack(
        [
            keep_edges_at_hop(adjacency, hop_distance, hop=0),
            keep_edges_at_hop(adjacency, hop_distance, hop=1),
        ]
    )  # [2, number_of_joints, number_of_joints]


def build_stgcn_branch(
    adjacency,
    temporal_kernel,
    depth_level,
    adaptive_graph=True,
):
    """Create the spatial-temporal residual blocks for one body part."""
    if depth_level == "0":
        channel_and_depth = [(64, 1), (128, 1), (256, 1)]
    elif depth_level == "1":
        channel_and_depth = [(64, 2), (128, 2), (256, 1)]

    blocks = []
    input_channels = 64
    for output_channels, repeat_count in channel_and_depth:
        for _ in range(repeat_count):
            blocks.append(
                STGCNBlock(
                    input_channels=input_channels,
                    output_channels=output_channels,
                    temporal_kernel=temporal_kernel,
                    adjacency=learnable(adjacency)
                    if adaptive_graph
                    else fixed(adjacency),
                    residual=True,
                )
            )
            input_channels = output_channels
    return sequential(blocks), input_channels


def stgcn_block_forward(x, block):
    """
    Spatial graph convolution followed by temporal convolution and a residual.

    x has shape [batch, channels, time, joints].
    """
    residual = identity_or_projection(x, block.output_channels)

    # Produce one feature group per adjacency partition.
    spatial = convolution_2d(
        x,
        output_channels=block.output_channels * block.number_of_partitions,
        kernel=(1, 1),
    )
    spatial = reshape(
        spatial,
        "[B, K*C, T, V] -> [B, K, C, T, V]",
    )
    spatial = einsum(
        "bkctv,kvw->bctw",
        spatial,
        block.adjacency,
    )
    spatial = relu(batch_normalization(spatial))

    temporal = convolution_2d(
        spatial,
        output_channels=block.output_channels,
        kernel=(block.temporal_kernel, 1),
        padding="same_on_time",
    )
    temporal = dropout(batch_normalization(temporal))
    return relu(temporal + residual)


def build_cosign_encoder(
    ordered_part_layouts,
    cumulative_joint_splits,
    input_channels,
    hidden_size,
    temporal_kernel,
    depth_level,
):
    """Construct one ST-GCN branch for each semantic keypoint group."""
    input_projection = Linear(input_channels, 64) + ReLU()
    branches = {}

    for part_name, joint_layout in ordered_part_layouts:
        adjacency = construct_part_graph(joint_layout)
        branches[part_name], branch_output_size = build_stgcn_branch(
            adjacency=adjacency,
            temporal_kernel=temporal_kernel,
            depth_level=depth_level,
            adaptive_graph=True,
        )

    # There are four branches (body, hand21, mouth_8, face), but five pooled
    # parts because left and right hands share one branch and remain separate.
    number_of_pooled_parts = len(cumulative_joint_splits)
    fusion = Linear(
        branch_output_size * number_of_pooled_parts,
        hidden_size,
    ) + ReLU()

    return input_projection, branches, fusion


def cosign_forward(
    keypoint_sequence,
    cumulative_joint_splits,
    ordered_part_names,
    input_projection,
    stgcn_branches,
    fusion,
):
    """
    Input:
        keypoint_sequence: [batch, time, joints, raw_channels]
        cumulative_joint_splits: [9, 30, 51, 59, 77]
        ordered_part_names: ["body", "hand21", "mouth_8", "face"]
    Output:
        frame-level part features and their fused representation
    """
    if keypoint_sequence.shape[-1] == 7:
        # The raw stream contains static and dynamic channels.
        # CoSign keeps x, y, and confidence for graph encoding.
        keypoint_sequence = concatenate(
            [
                keypoint_sequence[..., 0:2],
                keypoint_sequence[..., 6:7],
            ],
            axis="channel",
        )
    keypoint_sequence = keypoint_sequence[..., :input_projection.input_size]

    projected = relu(input_projection(keypoint_sequence))
    projected = transpose(
        projected,
        "[B, T, V, C] -> [B, C, T, V]",
    )

    encoded_parts = []
    split_index = 0
    for part_name in ordered_part_names:
        if part_name == "hand21":
            left_hand = slice_joints(
                projected,
                cumulative_joint_splits,
                split_index,
            )
            right_hand = slice_joints(
                projected,
                cumulative_joint_splits,
                split_index + 1,
            )

            # Process both hands in one batch with a shared hand graph.
            paired_hands = concatenate(
                [left_hand, right_hand],
                axis="batch",
            )
            paired_output = stgcn_branches["hand21"](paired_hands)
            left_output, right_output = split_in_half(
                paired_output,
                axis="batch",
            )
            encoded_parts.extend([left_output, right_output])
            split_index += 2
        else:
            part_input = slice_joints(
                projected,
                cumulative_joint_splits,
                split_index,
            )
            encoded_parts.append(stgcn_branches[part_name](part_input))
            split_index += 1

    # Reassemble parts in their original order, then average each part's joints.
    encoded = concatenate(encoded_parts, axis="joint")
    pooled_parts = []
    for index, end in enumerate(cumulative_joint_splits):
        start = 0 if index == 0 else cumulative_joint_splits[index - 1]
        part_features = encoded[..., start:end]
        pooled_parts.append(average(part_features, axis="joint"))

    part_features = concatenate(pooled_parts, axis="channel")
    part_features = transpose(
        part_features,
        "[B, C_parts, T] -> [B, T, C_parts]",
    )
    fused_features = relu(fusion(part_features))

    return {
        "cat": part_features,
        "fusion": fused_features,
    }


# ---------------------------------------------------------------------------
# Keypoint data supplied to CoSign
# ---------------------------------------------------------------------------


def build_keypoint_sample(raw_pose, selected_joint_indices, training):
    """
    Convert a raw pose sequence into the 77-joint, 7-channel CoSign input.

    The selected joints contain body (9), left hand (21), right hand (21),
    mouth (8), and face (18) landmarks.
    """
    pose = raw_pose[:, selected_joint_indices, :]  # [T, 77, (x, y, conf)]

    raw_coordinates = pose[..., 0:2]
    confidence = pose[..., 2:3]

    forward_motion = raw_coordinates[1:] - raw_coordinates[:-1]
    backward_motion = raw_coordinates[:-1] - raw_coordinates[1:]
    forward_motion = align_motion_to_frames(forward_motion, direction="forward")
    backward_motion = align_motion_to_frames(
        backward_motion,
        direction="backward",
    )

    raw_features = concatenate(
        [
            raw_coordinates,
            forward_motion,
            backward_motion,
            confidence,
        ],
        axis="channel",
    )
    normalized_features = raw_features / ((256 - 1) / 2) - 1

    # Center only x/y by body-part reference joints. Motion channels retain
    # their globally normalized values; confidence is restored unnormalized.
    centered_coordinates = center_each_body_part(
        normalized_features[..., 0:2],
        cumulative_splits=[9, 30, 51, 59, 77],
    )

    keypoints = concatenate(
        [
            centered_coordinates,
            normalized_features[..., 2:6],
            confidence,
        ],
        axis="channel",
    )  # [T, 77, 7]

    if training:
        keypoints = random_temporal_rescale(keypoints, maximum_ratio=0.2)
    return keypoints


def collate_keypoint_samples(samples, tokenizer_pad_id, downsample=True):
    """Edge-pad keypoints and build the post-TemporalConv attention mask."""
    raw_lengths = [len(sample.keypoints) for sample in samples]

    # Round each valid length for the two pooling stages, then add four repeated
    # boundary frames on both sides.
    padded_valid_lengths = [
        round_up(length, multiple=4) + 8 for length in raw_lengths
    ]
    batch_length = max(padded_valid_lengths)

    keypoint_batch = []
    for sample, valid_length in zip(samples, padded_valid_lengths):
        sequence = repeat_first_frame(sample.keypoints, count=4)
        sequence = concatenate([sequence, sample.keypoints], axis="time")
        sequence = repeat_last_frame_until(
            sequence,
            length=batch_length,
        )
        keypoint_batch.append(sequence)

    if downsample:
        encoded_lengths = [
            integer_divide(length, 4) - 2
            for length in padded_valid_lengths
        ]
    else:
        encoded_lengths = padded_valid_lengths

    # Valid positions become one and batch padding becomes zero.
    encoder_mask = lengths_to_mask(
        encoded_lengths,
        padding_sentinel=tokenizer_pad_id,
    )

    return {
        "keypoints": stack(keypoint_batch),  # [B, T_batch, 77, 7]
        "lengths": tensor(padded_valid_lengths),
        "encoder_mask": encoder_mask,
        "names": [sample.name for sample in samples],
    }


# ---------------------------------------------------------------------------
# Integration into the sign-to-text model
# ---------------------------------------------------------------------------


def build_sign_to_text_model(configuration):
    """Attach CoSign to temporal, translation, contrastive, and CTC modules."""
    text_generator = load_pretrained_seq2seq_transformer(
        configuration.translation_model
    )
    cosign = build_cosign_encoder(
        ordered_part_layouts=configuration.part_layouts,
        cumulative_joint_splits=configuration.joint_splits,
        input_channels=configuration.keypoint_channels,
        hidden_size=configuration.cosign_hidden_size,
        temporal_kernel=configuration.temporal_kernel,
        depth_level=configuration.stgcn_level,
    )

    temporal_encoder = TemporalConvolution(
        input_size=configuration.cosign_hidden_size,
        hidden_size=512,
        convolution_type=configuration.temporal_convolution_type,
    )
    sign_embedding = Sequential(
        Linear(512, text_generator.model_dimension),
        LayerNormalization(text_generator.model_dimension),
        ReLU(),
    )

    modules = {
        "cosign": cosign,
        "temporal_encoder": temporal_encoder,
        "sign_embedding": sign_embedding,
        "text_generator": text_generator,
    }

    if configuration.use_contrastive_loss:
        text_encoder = load_pretrained_text_encoder(
            configuration.contrastive_text_model
        )
        freeze_parameters(text_encoder)
        modules.update(
            {
                "contrastive_text_encoder": text_encoder,
                "visual_projection": Linear(
                    text_generator.model_dimension, 1024
                ),
                "text_projection": Linear(
                    text_encoder.model_dimension, 1024
                ),
                "logit_scale": learnable_scalar(initial_value=log(1 / 0.07)),
            }
        )

    if configuration.use_ctc_loss:
        modules.update(
            {
                "temporal_gloss_classifier": Linear(
                    512, configuration.gloss_vocabulary_size
                ),
                "transformer_gloss_classifier": Linear(
                    text_generator.model_dimension,
                    configuration.gloss_vocabulary_size,
                ),
            }
        )

    return SignToTextModel(**modules)


def encode_keypoints_for_translation(model, source_batch):
    """
    Convert padded keypoints into transformer-compatible visual tokens.

    source_batch.keypoints: [B, T_padded, 77, C_raw]
    source_batch.lengths: valid pre-temporal lengths
    source_batch.encoder_mask: [B, L_visual] after temporal downsampling
    """
    cosign_output = cosign_forward(
        keypoint_sequence=source_batch.keypoints,
        cumulative_joint_splits=model.joint_splits,
        ordered_part_names=model.part_names,
        input_projection=model.cosign.input_projection,
        stgcn_branches=model.cosign.branches,
        fusion=model.cosign.fusion,
    )
    framewise_features = cosign_output["fusion"]  # [B, T_padded, H]

    temporal_features = model.temporal_encoder(
        framewise_features
    )  # [B, L_visual, 512]
    visual_tokens = model.sign_embedding(
        temporal_features
    )  # [B, L_visual, D_model]

    return {
        "framewise_features": framewise_features,
        "temporal_features": temporal_features,
        "visual_tokens": visual_tokens,
        "visual_mask": source_batch.encoder_mask,
    }


def cico_alignment_similarity(
    visual_features,
    visual_mask,
    text_features,
    text_mask,
    logit_scale,
):
    """Compute symmetric fine-grained visual-text alignment scores."""
    frame_token_scores = einsum(
        "bid,njd->bnij",
        visual_features,
        text_features,
    )

    # Visual-to-text: softly align every visual frame to all valid text tokens.
    v2t_scores = apply_key_mask(frame_token_scores, text_mask)
    v2t_attention = softmax(v2t_scores / 0.07, axis="text_token")
    v2t_per_frame = sum(
        v2t_scores * v2t_attention,
        axis="text_token",
    )
    v2t_per_frame = zero_invalid_queries(v2t_per_frame, visual_mask)
    visual_to_text = (
        exp(logit_scale)
        * sum(v2t_per_frame, axis="visual_frame")
        / count_valid_tokens(visual_mask)
    )

    # Text-to-visual: reverse the alignment direction.
    t2v_scores = apply_query_mask(frame_token_scores, visual_mask)
    t2v_attention = softmax(t2v_scores / 0.07, axis="visual_frame")
    t2v_per_token = sum(
        t2v_scores * t2v_attention,
        axis="visual_frame",
    )
    t2v_per_token = zero_invalid_queries(t2v_per_token, text_mask)
    text_to_visual = (
        exp(logit_scale)
        * sum(t2v_per_token, axis="text_token")
        / count_valid_tokens(text_mask)
    )

    return visual_to_text, text_to_visual


def compute_training_losses(model, source_batch, target_batch):
    """Run the full multi-objective CoSign sign-to-text forward pass."""
    encoded = encode_keypoints_for_translation(model, source_batch)

    transformer_output = model.text_generator(
        inputs_embeds=encoded["visual_tokens"],
        attention_mask=encoded["visual_mask"],
        labels=target_batch.translation_token_ids,
        decoder_attention_mask=target_batch.translation_mask,
        return_encoder_states=True,
    )
    contextual_visual_tokens = transformer_output.encoder_last_hidden_state

    # Translation objective over decoder vocabulary; padding does not contribute.
    language_model_loss = cross_entropy(
        flatten(transformer_output.decoder_logits),
        flatten(target_batch.translation_token_ids),
        ignore_index=model.tokenizer.pad_token_id,
        label_smoothing=0.2,
    )

    if model.uses_contrastive_loss:
        visual_features = l2_normalize(
            model.visual_projection(contextual_visual_tokens),
            axis=-1,
        )
        with no_gradient():
            raw_text_features = model.contrastive_text_encoder(
                input_ids=target_batch.translation_token_ids,
                attention_mask=target_batch.translation_mask,
            )
        text_features = l2_normalize(
            model.text_projection(raw_text_features),
            axis=-1,
        )

        visual_to_text, text_to_visual = cico_alignment_similarity(
            visual_features=visual_features,
            visual_mask=encoded["visual_mask"],
            text_features=text_features,
            text_mask=target_batch.translation_mask,
            logit_scale=model.logit_scale,
        )
        positive_pairs = identity_matrix(batch_size(source_batch))
        contrastive_loss = mean(
            [
                cross_entropy(
                    visual_to_text,
                    positive_pairs,
                    label_smoothing=0.2,
                ),
                cross_entropy(
                    transpose(text_to_visual),
                    positive_pairs,
                    label_smoothing=0.2,
                ),
            ]
        )
    else:
        contrastive_loss = 0

    if model.uses_ctc_loss:
        input_lengths = source_batch.lengths - 2

        temporal_gloss_logits = log_softmax(
            model.temporal_gloss_classifier(encoded["temporal_features"]),
            axis="gloss",
        )
        transformer_gloss_logits = log_softmax(
            model.transformer_gloss_classifier(contextual_visual_tokens),
            axis="gloss",
        )

        temporal_ctc = ctc_loss(
            logits=transpose_to_time_major(temporal_gloss_logits),
            targets=target_batch.gloss_ids,
            input_lengths=input_lengths,
            target_lengths=target_batch.gloss_lengths,
            blank_id=0,
            zero_infinity=True,
        )
        transformer_ctc = ctc_loss(
            logits=transpose_to_time_major(transformer_gloss_logits),
            targets=target_batch.gloss_ids,
            input_lengths=input_lengths,
            target_lengths=target_batch.gloss_lengths,
            blank_id=0,
            zero_infinity=True,
        )
        ctc_objective = temporal_ctc + transformer_ctc
    else:
        ctc_objective = 0

    return {
        "language_model_loss": language_model_loss,
        "contrastive_loss": contrastive_loss,
        "ctc_loss": ctc_objective,
    }


def build_optimizer_and_schedule(model, training_configuration):
    """Apply weight-decay grouping, warmup, and cosine learning-rate decay."""
    parameter_groups = separate_decay_and_no_decay_parameters(
        model,
        weight_decay=training_configuration.weight_decay,
    )

    if training_configuration.optimizer == "SGD":
        optimizer = SGD(
            parameter_groups,
            learning_rate=training_configuration.learning_rate,
            momentum=0.9,
        )
    elif training_configuration.optimizer == "Adam":
        optimizer = Adam(
            parameter_groups,
            learning_rate=training_configuration.learning_rate,
            betas=training_configuration.betas,
            epsilon=training_configuration.epsilon,
        )
    else:
        optimizer = AdamW(
            parameter_groups,
            learning_rate=training_configuration.learning_rate,
            betas=training_configuration.betas,
            epsilon=training_configuration.epsilon,
            weight_decay=training_configuration.weight_decay,
        )

    warmup = LinearWarmup(
        optimizer,
        start_factor=training_configuration.warmup_factor,
        number_of_epochs=training_configuration.warmup_epochs,
    )
    cosine_decay = CosineAnnealing(
        optimizer,
        total_epochs=training_configuration.total_epochs,
        minimum_learning_rate=training_configuration.minimum_learning_rate,
    )
    return optimizer, ChainedSchedule([warmup, cosine_decay])


def optimize_sign_to_text_model(
    model,
    distributed_training_loader,
    optimizer,
    scheduler,
    loss_weights,
    gradient_accumulation_steps,
):
    """Train CoSign jointly with all unfrozen downstream translation modules."""
    model = convert_to_synchronized_batch_normalization(model)
    model = wrap_distributed_data_parallel(
        model,
        detect_unused_parameters=False,
    )
    model.train()

    for epoch in training_epochs():
        distributed_training_loader.sampler.set_epoch(epoch)
        optimizer.zero_grad()

        for step, (source_batch, target_batch) in enumerate(
            distributed_training_loader,
            start=1,
        ):
            losses = compute_training_losses(
                model,
                source_batch,
                target_batch,
            )
            total_loss = (
                loss_weights.language_model
                * losses["language_model_loss"]
                + loss_weights.contrastive
                * losses["contrastive_loss"]
                + loss_weights.ctc
                * losses["ctc_loss"]
            )
            total_loss = total_loss / gradient_accumulation_steps
            total_loss.backward()

            if step % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        scheduler.step()  # linear warmup chained with cosine decay
        validation_metrics = evaluate_sign_to_text_model(model)
        save_checkpoint_if_improved(model, optimizer, validation_metrics)


def generate_translation(model, source_batch, beam_width, maximum_new_tokens):
    """Reuse the same visual path for autoregressive translation."""
    encoded = encode_keypoints_for_translation(model, source_batch)
    generated_tokens = model.text_generator.generate(
        inputs_embeds=encoded["visual_tokens"],
        attention_mask=encoded["visual_mask"],
        number_of_beams=beam_width,
        maximum_new_tokens=maximum_new_tokens,
        decoder_start_token_id=model.decoder_start_token_id,
    )

    if model.uses_ctc_loss:
        encoder_states = model.text_generator.encode(
            encoded["visual_tokens"],
            attention_mask=encoded["visual_mask"],
        )
        gloss_logits = model.transformer_gloss_classifier(
            encoder_states[:, 1:, :]
        )
        gloss_tokens = ctc_greedy_decode(
            gloss_logits,
            lengths=source_batch.lengths - 2,
        )
    else:
        gloss_tokens = empty_gloss_predictions(batch_size(source_batch))

    return gloss_tokens, generated_tokens


def export_framewise_cosign_features(model, source_batch):
    """
    Cache full-rate CoSign fusion features before temporal downsampling.

    The implementation starts at offset four and retains the stored per-sample
    length; this removes batch-level padding while preserving the intended
    framewise sequence used by the downstream multimodal model.
    """
    framewise = model.cosign(source_batch.keypoints)["fusion"]
    unpadded_sequences = []

    for sample_index, valid_length in enumerate(source_batch.lengths):
        unpadded_sequences.append(
            framewise[sample_index, 4 : valid_length + 4]
        )
    return unpadded_sequences

