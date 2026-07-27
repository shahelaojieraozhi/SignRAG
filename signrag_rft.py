# pyright: reportUndefinedVariable=false
"""Detailed pseudocode for retrieval-dependent SignRAG-RFT."""


def prepare_sample(sample, tokenizer, maximum_frames):
    """
    Tokenize paired instructions prepared by the offline retrieval pipeline.

    The dataset already stores one user instruction containing retrieved
    demonstrations and a matched instruction with those demonstrations absent.
    """
    system_message = localized_translation_instruction(sample.language)

    prompt_with_rag = apply_chat_template(
        system=system_message,
        user=sample.instruction_with_retrieved_demonstrations,
        add_generation_marker=True,
    )
    prompt_without_rag = apply_chat_template(
        system=system_message,
        user=sample.same_instruction_without_demonstrations,
        add_generation_marker=True,
    )

    # Load sign features produced by the pretrained sign encoder.
    sign = load_sign_features(sample.sign_id)
    if sample.uses_auxiliary_cues:
        face = load_optional_face_features(sample.sign_id)
        left_hand = load_optional_left_hand_features(sample.sign_id)
        right_hand = load_optional_right_hand_features(sample.sign_id)

        # Auxiliary streams are symmetrically padded to the sign-stream length.
        face = symmetric_zero_pad(face, target_length=len(sign))
        left_hand = symmetric_zero_pad(left_hand, target_length=len(sign))
        right_hand = symmetric_zero_pad(right_hand, target_length=len(sign))
    else:
        face, left_hand, right_hand = None, None, None

    if len(sign) > maximum_frames:
        # Apply one shared temporal subsample to all aligned visual streams.
        frame_indices = sorted_random_sample(len(sign), maximum_frames)
        sign = sign[frame_indices]
        if sample.uses_auxiliary_cues:
            face = face[frame_indices]
            left_hand = left_hand[frame_indices]
            right_hand = right_hand[frame_indices]

    return {
        "prompt_with_rag": tokenize_with_visual_placeholder(
            prompt_with_rag, tokenizer
        ),
        "prompt_without_rag": tokenize_with_visual_placeholder(
            prompt_without_rag, tokenizer
        ),
        "sign": sign,
        "face": face,
        "left_hand": left_hand,
        "right_hand": right_hand,
        "reference_text": sample.reference_translation,
    }


def collate_multimodal_batch(samples, temporal_downsample):
    """Pad visual streams consistently and derive the post-encoder mask."""
    original_lengths = [len(sample["sign"]) for sample in samples]
    maximum_length = max(original_lengths)
    boundary_padding = 4
    padded_length = maximum_length + 2 * boundary_padding

    for sample in samples:
        sample["sign"] = edge_pad(
            sample["sign"],
            left=boundary_padding,
            total_length=padded_length,
        )
        for stream in ["face", "left_hand", "right_hand"]:
            if sample[stream] is not None:
                sample[stream] = edge_pad(
                    sample[stream],
                    left=boundary_padding,
                    total_length=padded_length,
                )

    # The temporal encoder reduces sequence length; the mask follows its output.
    encoded_lengths = [
        integer_divide(length + 2 * boundary_padding, temporal_downsample) - 2
        for length in original_lengths
    ]
    visual_attention_mask = lengths_to_mask(encoded_lengths)

    return {
        "prompt_with_rag": stack(
            sample["prompt_with_rag"] for sample in samples
        ),
        "prompt_without_rag": stack(
            sample["prompt_without_rag"] for sample in samples
        ),
        "sign": stack(sample["sign"] for sample in samples),
        "face": optional_stack(sample["face"] for sample in samples),
        "left_hand": optional_stack(
            sample["left_hand"] for sample in samples
        ),
        "right_hand": optional_stack(
            sample["right_hand"] for sample in samples
        ),
        "visual_attention_mask": visual_attention_mask,
        "reference_text": [
            sample["reference_text"] for sample in samples
        ],
    }


def initialize_policy(
    supervised_multimodal_model,
    sign_encoder_checkpoint,
    multimodal_checkpoint,
    lora_configuration,
):
    """Load supervised initialization and expose only LoRA parameters."""
    policy = load_multimodal_causal_language_model(
        supervised_multimodal_model,
        attention_implementation="scaled_dot_product",
    )
    policy.attach_lora(lora_configuration)
    policy.initialize_sign_modules(sign_encoder_checkpoint)
    policy.load_multimodal_weights(multimodal_checkpoint, strict=False)

    freeze_all_parameters(policy)
    unfreeze_lora_parameters(policy)

    # For a PEFT policy, disabling its trainable adapter supplies the reference.
    reference_policy = adapter_disabled_view(policy)
    return policy, reference_policy


def build_grouped_rollout_batch(dataset, number_of_generations):
    """
    Repeat each sample index G consecutive times before collation.

    Each repeated item produces one rollout; grouping is performed by index
    order rather than by requesting G outputs from one generation call.
    """
    repeated_indices = repeat_each_dataset_index(
        dataset,
        repeat_count=number_of_generations,
    )
    return batch_sampler(
        repeated_indices,
        require_global_batch_divisible_by=number_of_generations,
    )


def generate_group(policy, batch, maximum_new_tokens):
    """Sample one translation for every repeated item using its RAG prompt."""
    multimodal_inputs = {
        "sign": batch["sign"],
        "visual_attention_mask": batch["visual_attention_mask"],
        "face": batch["face"],
        "left_hand": batch["left_hand"],
        "right_hand": batch["right_hand"],
    }
    completions = policy.generate(
        prompt=batch["prompt_with_rag"],
        multimodal_inputs=multimodal_inputs,
        do_sample=True,
        temperature=1.0,
        maximum_new_tokens=maximum_new_tokens,
    )

    completion_mask = mask_through_first_end_token(completions)
    return completions, completion_mask, multimodal_inputs


def translation_quality_reward(completions, references, language):
    """Use complementary n-gram and sequence-overlap translation rewards."""
    completions, references = language_aware_tokenize(
        completions, references, language
    )
    bleu_reward = mean(
        [
            bleu_1(completions, references),
            bleu_2(completions, references),
            bleu_3(completions, references),
            bleu_4(completions, references),
        ]
    )
    rouge_reward = rouge_l(completions, references)
    return bleu_reward + rouge_reward


def average_teacher_forced_log_probability(
    policy,
    prompt,
    completions,
    completion_mask,
    multimodal_inputs,
):
    """
    Average valid completion-token log probabilities; ignore prompt and pad.

    The visual placeholder is outside the language vocabulary, so it must
    never be used as a target in the token-probability gather.
    """
    full_sequence = concatenate([prompt, completions], axis="token")
    attention_mask = concatenate(
        [ones_like(prompt), completion_mask],
        axis="token",
    )

    image_positions = positions_equal_to(
        full_sequence,
        VISUAL_PLACEHOLDER_ID,
    )
    assert exactly_one_image_placeholder_per_sample(image_positions)
    assert same_image_placeholder_position_within_batch(image_positions)

    labels = copy(full_sequence)
    labels[:, : prompt.length] = IGNORE_INDEX
    labels[is_padding(full_sequence)] = IGNORE_INDEX

    logits = policy(
        input_ids=full_sequence,
        attention_mask=attention_mask,
        multimodal_inputs=multimodal_inputs,
    ).logits

    predicted_logits = logits[:, :-1, :]
    target_ids = full_sequence[:, 1:]
    shifted_labels = labels[:, 1:]
    shifted_attention = attention_mask[:, 1:]

    valid = shifted_attention * (shifted_labels != IGNORE_INDEX)
    valid = valid * (target_ids != VISUAL_PLACEHOLDER_ID)
    valid = valid * token_ids_are_inside_vocabulary(target_ids, logits)

    safe_target_ids = where(valid, target_ids, 0)
    token_log_probability = gather_target_probability(
        log_softmax(to_float32(predicted_logits), axis="vocabulary"),
        safe_target_ids,
    )
    token_log_probability = where(valid, token_log_probability, 0)

    return sum(token_log_probability, axis="token") / clamp_min(
        sum(valid, axis="token"),
        1,
    )


def retrieval_dependence_reward(
    policy,
    batch,
    completions,
    completion_mask,
    multimodal_inputs,
    dependence_strength,
    threshold=0.0,
):
    """
    Evaluate the same generated translation with and without retrieved context.

    Positive log-likelihood gain means that retrieved demonstrations support
    the sampled translation rather than being ignored.
    """
    with no_gradient():
        logp_with_rag = average_teacher_forced_log_probability(
            policy,
            batch["prompt_with_rag"],
            completions,
            completion_mask,
            multimodal_inputs,
        )
        logp_without_rag = average_teacher_forced_log_probability(
            policy,
            batch["prompt_without_rag"],
            completions,
            completion_mask,
            multimodal_inputs,
        )

    dependence = logp_with_rag - logp_without_rag
    positive_dependence = relu(dependence - threshold)
    return dependence_strength * positive_dependence


def combine_rewards(quality, dependence, mixing_weight=0.3):
    """Match reward scales before adding retrieval dependence."""
    quality_magnitude = mean(abs(quality))
    dependence_magnitude = clamp_min(mean(abs(dependence)), SMALL_NUMBER)
    scale = min(quality_magnitude / dependence_magnitude, 100.0)
    return stop_gradient(quality + mixing_weight * scale * dependence)


def group_relative_advantage(rewards, number_of_generations):
    """Normalize rewards only among completions from the same input."""
    groups = reshape(rewards, "[(batch G)] -> [batch, G]")
    group_mean = mean(groups, axis="G")
    group_std = standard_deviation(groups, axis="G")
    advantages = (groups - group_mean) / (group_std + 1e-4)
    return flatten(advantages)


def completion_log_probabilities_for_grpo(
    policy,
    prompt,
    completions,
    completion_mask,
    multimodal_inputs,
):
    """
    Extract completion log probabilities after multimodal token expansion.

    The language input contains one visual placeholder, whereas the model
    logits contain a variable-length block of projected visual embeddings.
    Remove that expanded block before next-token alignment.
    """
    full_sequence = concatenate([prompt, completions], axis="token")
    full_attention = concatenate(
        [ones_like(prompt), completion_mask],
        axis="token",
    )
    image_index = index_of_visual_placeholder(full_sequence)

    logits = policy(
        input_ids=full_sequence,
        attention_mask=full_attention,
        multimodal_inputs=multimodal_inputs,
    ).logits

    text_suffix_length = full_sequence.length - image_index - 1
    text_aligned_logits = concatenate(
        [
            logits[:, :image_index, :],
            logits[:, -text_suffix_length:, :],
        ],
        axis="token",
    )
    text_only_ids = remove_token_at(full_sequence, image_index)

    next_token_logits = text_aligned_logits[:, :-1, :]
    next_token_ids = text_only_ids[:, 1:]
    return gather_target_probability(
        log_softmax(to_float32(next_token_logits), axis="vocabulary"),
        next_token_ids,
    )


def clipped_grpo_loss(
    policy_logp,
    old_policy_logp,
    reference_logp,
    advantages,
    completion_mask,
    clip_low,
    clip_high,
    kl_weight,
):
    """Compute the completion-masked clipped objective plus KL regularization."""
    log_ratio = clamp(policy_logp - old_policy_logp, -5.0, 5.0)
    ratio = exp(log_ratio)
    clipped_ratio = clamp(ratio, 1 - clip_low, 1 + clip_high)

    policy_term = -minimum(
        ratio * advantages[:, None],
        clipped_ratio * advantages[:, None],
    )

    reference_ratio = clamp(reference_logp - policy_logp, -5.0, 5.0)
    token_kl = exp(reference_ratio) - reference_ratio - 1
    token_loss = policy_term + kl_weight * token_kl

    sequence_loss = (
        sum(token_loss * completion_mask, axis="token")
        / sum(completion_mask, axis="token")
    )
    return mean(sequence_loss)


def signrag_rft_step(batch, policy, reference_policy, configuration):
    """Perform one complete retrieval-aware group-relative update."""
    completions, completion_mask, multimodal_inputs = generate_group(
        policy,
        batch,
        configuration.maximum_new_tokens,
    )

    quality = translation_quality_reward(
        completions,
        batch["reference_text"],
        configuration.language,
    )
    dependence = retrieval_dependence_reward(
        policy,
        batch,
        completions,
        completion_mask,
        multimodal_inputs,
        dependence_strength=configuration.dependence_strength,
        threshold=configuration.dependence_threshold,
    )
    rewards = combine_rewards(quality, dependence, mixing_weight=0.3)
    advantages = group_relative_advantage(
        all_gather(rewards),
        configuration.number_of_generations,
    )

    policy_logp = completion_log_probabilities_for_grpo(
        policy,
        batch["prompt_with_rag"],
        completions,
        completion_mask,
        multimodal_inputs,
    )
    old_policy_logp = stop_gradient(policy_logp)
    reference_logp = completion_log_probabilities_for_grpo(
        reference_policy,
        batch["prompt_with_rag"],
        completions,
        completion_mask,
        multimodal_inputs,
    )

    loss = clipped_grpo_loss(
        policy_logp=to_float32(policy_logp),
        old_policy_logp=to_float32(old_policy_logp),
        reference_logp=to_float32(reference_logp),
        advantages=to_float32(local_worker_slice(advantages)),
        completion_mask=completion_mask,
        clip_low=configuration.clip_low,
        clip_high=configuration.clip_high,
        kl_weight=configuration.kl_weight,
    )
    update_only_lora_parameters(loss)
    return loss


def train_and_evaluate(dataset, policy, reference_policy, configuration):
    """Distributed optimization followed by deterministic translation."""
    grouped_sampler = build_grouped_rollout_batch(
        dataset,
        configuration.number_of_generations,
    )
    for batch in distributed_batches(
        dataset,
        configuration.batch_size,
        sampler=grouped_sampler,
    ):
        batch = collate_multimodal_batch(
            batch,
            temporal_downsample=configuration.temporal_downsample,
        )
        loss = signrag_rft_step(
            batch,
            policy,
            reference_policy,
            configuration,
        )
        accumulate_gradients(loss, configuration.gradient_accumulation_steps)
        periodically_save_lora_checkpoint(policy)

    evaluate_saved_adapters(policy, configuration)


def collate_inference_batch(samples, tokenizer, temporal_downsample):
    """Use left text padding so generation starts at one shared position."""
    multimodal_batch = collate_multimodal_batch(
        samples,
        temporal_downsample,
    )
    prompt_ids = multimodal_batch["prompt_with_rag"]
    maximum_prompt_length = max(len(ids) for ids in prompt_ids)

    left_padded_ids = full(
        [len(samples), maximum_prompt_length],
        tokenizer.pad_token_id,
    )
    attention_mask = zeros_like(left_padded_ids)
    for row, ids in enumerate(prompt_ids):
        left_padded_ids[row, -len(ids) :] = ids
        attention_mask[row, -len(ids) :] = 1

    multimodal_batch["prompt_with_rag"] = left_padded_ids
    multimodal_batch["prompt_attention_mask"] = attention_mask
    return multimodal_batch


def evaluate_saved_adapters(policy, configuration):
    """Reload adapter checkpoints and perform deterministic distributed decoding."""
    for checkpoint in selected_adapter_checkpoints():
        adapter_state = load_adapter_tensors(checkpoint)
        adapter_state = normalize_adapter_key_names(adapter_state)
        policy.load_state_dict(adapter_state, strict=False)
        policy.eval()

        local_predictions, local_references = [], []
        for samples in distributed_test_batches():
            batch = collate_inference_batch(
                samples,
                policy.tokenizer,
                configuration.temporal_downsample,
            )
            with no_gradient():
                generated_ids = policy.generate(
                    input_ids=batch["prompt_with_rag"],
                    attention_mask=batch["prompt_attention_mask"],
                    sign=batch["sign"],
                    visual_attention_mask=batch["visual_attention_mask"],
                    face=batch["face"],
                    left_hand=batch["left_hand"],
                    right_hand=batch["right_hand"],
                    do_sample=False,
                    number_of_beams=configuration.inference_beam_width,
                    maximum_new_tokens=configuration.inference_maximum_tokens,
                    use_generation_cache=True,
                )
            local_predictions.extend(generated_ids)
            local_references.extend(batch["reference_text"])

        predictions = distributed_gather_and_decode(local_predictions)
        references = distributed_gather_and_decode(local_references)
        if current_worker() == 0:
            report_translation_metrics(
                predictions,
                references,
                metrics=["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-L"],
            )

