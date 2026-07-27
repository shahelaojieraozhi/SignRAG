# pyright: reportUndefinedVariable=false
"""Detailed pseudocode for distributed knee-based exemplar retrieval."""


def build_retrieval_loaders(
    training_set,
    validation_set,
    test_set,
    retrieval_batch_size,
):
    """Shard every split across workers while preserving evaluation order."""
    training_sampler = DistributedSampler(training_set, shuffle=True)
    validation_sampler = DistributedSampler(validation_set, shuffle=False)
    test_sampler = DistributedSampler(test_set, shuffle=False)

    return {
        "train": DataLoader(
            training_set,
            batch_size=retrieval_batch_size,
            sampler=training_sampler,
            collate_fn=training_set.collate,
            pinned_memory=True,
        ),
        "validation": DataLoader(
            validation_set,
            batch_size=retrieval_batch_size,
            sampler=validation_sampler,
            collate_fn=validation_set.collate,
            pinned_memory=True,
        ),
        "test": DataLoader(
            test_set,
            batch_size=retrieval_batch_size,
            sampler=test_sampler,
            collate_fn=test_set.collate,
            pinned_memory=True,
        ),
    }


def initialize_retrieval_model(model_configuration, checkpoint):
    """Restore the jointly trained sign-to-text encoder for retrieval."""
    model = build_sign_to_text_model(model_configuration)
    model = convert_to_synchronized_batch_normalization(model)
    model = wrap_distributed_data_parallel(model)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model


def extract_retrieval_features(data_loader, model):
    """
    Encode each sign sequence as normalized frame-level features.

    Returns CPU batches because sequence lengths differ across mini-batches.
    """
    feature_batches, mask_batches = [], []
    sample_names, reference_texts = [], []

    for source, target in data_loader:
        cosign_output = model.cosign_encoder(
            source.keypoints,
            source.lengths,
        )
        temporal = model.temporal_encoder(cosign_output["fusion"])
        visual_tokens = model.sign_embedding(temporal)

        # Run the labeled seq2seq forward and retain its encoder-side states.
        seq2seq_output = model.text_generator(
            inputs_embeds=visual_tokens,
            attention_mask=source.attention_mask,
            labels=target.token_ids,
            decoder_attention_mask=target.attention_mask,
        )
        visual = l2_normalize(
            model.visual_projection(
                seq2seq_output.encoder_last_hidden_state
            ),
            axis=-1,
        )

        feature_batches.append(move_to_cpu(visual))          # [B, L, D]
        mask_batches.append(move_to_cpu(source.attention_mask))  # [B, L]
        sample_names.extend(source.names)
        reference_texts.extend(target.raw_text)

    return {
        "features": feature_batches,
        "masks": mask_batches,
        "names": sample_names,
        "texts": reference_texts,
    }


def pad_local_sequences(dataset, global_length):
    """Right-pad variable-length batches, then concatenate along samples."""
    padded_features, padded_masks = [], []

    for features, masks in zip(dataset["features"], dataset["masks"]):
        padding = global_length - features.shape[1]
        padded_features.append(right_pad(features, padding, value=0))
        padded_masks.append(right_pad(masks, padding, value=0))

    return {
        "features": concatenate(padded_features, axis=0),  # [N_local, L, D]
        "masks": concatenate(padded_masks, axis=0),        # [N_local, L]
        "names": dataset["names"],
        "texts": dataset["texts"],
    }


def framewise_attention_similarity(query, query_mask, gallery, gallery_mask):
    """
    Compute masked query-to-gallery similarity.

    q: [Bq, Lq, D], g: [Bg, Lg, D]
    pairwise[b, n, i, j] is the dot product between query frame i and
    gallery frame j. Attention selects the best aligned gallery frames.
    """
    pairwise = einsum("bid,njd->bnij", query, gallery)

    gallery_weights = where(gallery_mask == 0, 1e-5, gallery_mask)
    weighted_pairwise = pairwise * broadcast_over_query_frames(
        gallery_weights
    )
    frame_attention = softmax(
        weighted_pairwise / 0.07,
        axis="gallery_frame",
    )
    aligned_scores = nan_sum(
        weighted_pairwise * frame_attention,
        axis="gallery_frame",
    )

    aligned_scores = where(
        broadcast_over_gallery(query_mask == 1),
        aligned_scores,
        0,
    )
    sequence_scores = (
        exp(learnable_temperature)
        * nan_sum(aligned_scores, axis="query_frame")
        / count_valid_frames(query_mask)
    )
    return sequence_scores  # [Bq, Bg]


def compute_similarity_chunk(
    model,
    query_chunk,
    gallery_chunk_on_cpu,
    execution_mode="auto",
):
    """Use the accelerator when possible; retry on CPU after memory overflow."""
    if execution_mode == "cpu":
        return model.framewise_similarity_cpu(query_chunk, gallery_chunk_on_cpu)

    try:
        gallery_chunk = move_to_accelerator(gallery_chunk_on_cpu)
        scores = model.framewise_similarity(query_chunk, gallery_chunk)
        return move_to_cpu(scores)
    except AcceleratorOutOfMemory:
        if execution_mode != "auto":
            raise
        release_accelerator_cache()
        return model.framewise_similarity_cpu(
            move_to_cpu(query_chunk),
            gallery_chunk_on_cpu,
        )


def select_by_knee(sorted_scores, maximum_k, minimum_k=3, minimum_drop=0.0):
    """Select a variable number of neighbors using the largest score drop."""
    if len(sorted_scores) == 0:
        return 0

    number_of_differences = min(maximum_k, len(sorted_scores) - 1)
    if number_of_differences == 0:
        return min(1, maximum_k)

    drops = [
        sorted_scores[j] - sorted_scores[j + 1]
        for j in range(number_of_differences)
    ]
    largest_drop_index = argmax(drops)       # zero-based
    knee_position = largest_drop_index + 1   # one-based position

    if drops[largest_drop_index] < minimum_drop:
        number_to_keep = minimum_k
    else:
        # Keep the candidates before the knee, with a small minimum context.
        number_to_keep = max(minimum_k, knee_position - 1)

    return min(number_to_keep, maximum_k, len(sorted_scores))


def distributed_knee_retrieval(
    training_loader,
    query_loader,
    model,
    maximum_k,
    query_chunk_size,
    gallery_chunk_size,
    query_is_training_set=False,
    execution_mode="auto",
):
    """
    Search the complete training gallery while retaining only one shard per worker.

    Large tensors remain sharded. Host-backed object collectives communicate
    query chunks, metadata, and compact candidate lists.
    """
    local_gallery_raw = extract_retrieval_features(training_loader, model)
    if query_is_training_set:
        local_query_raw = local_gallery_raw
    else:
        local_query_raw = extract_retrieval_features(query_loader, model)

    local_maximum = max_sequence_length(local_gallery_raw, local_query_raw)
    global_maximum = all_reduce_max(local_maximum)
    local_gallery = pad_local_sequences(local_gallery_raw, global_maximum)
    local_queries = pad_local_sequences(local_query_raw, global_maximum)

    query_counts = all_gather_object(len(local_queries["names"]))
    local_outputs = {}

    # Every worker becomes the query owner in turn.
    for owner in range(number_of_workers()):
        for start in range(0, query_counts[owner], query_chunk_size):
            if current_worker() == owner:
                query_chunk = slice_dataset(
                    local_queries, start, start + query_chunk_size
                )
            else:
                query_chunk = None

            query_chunk = broadcast_object(query_chunk, source=owner)
            query_on_device = move_to_accelerator(
                query_chunk["features"],
                query_chunk["masks"],
            )

            # Compare the shared query chunk with this worker's gallery shard.
            local_score_blocks = []
            for gallery_chunk in chunks(local_gallery, gallery_chunk_size):
                local_score_blocks.append(
                    compute_similarity_chunk(
                        model,
                        query_on_device,
                        gallery_chunk,
                        execution_mode,
                    )
                )
            local_scores = concatenate(local_score_blocks, axis="gallery")

            # Keep only top candidates locally before cross-worker communication.
            candidates_per_query = []
            for query, scores in pair(query_chunk, local_scores):
                local_candidates = []
                for index in argsort(scores, descending=True):
                    candidate = local_gallery.item(index)

                    if query_is_training_set and candidate.name == query.name:
                        continue
                    if query_is_training_set and candidate.text == query.text:
                        continue

                    local_candidates.append(
                        Candidate(scores[index], candidate.name, candidate.text)
                    )
                    if len(local_candidates) == maximum_k:
                        break
                candidates_per_query.append(local_candidates)

            gathered_candidates = all_gather_object(candidates_per_query)

            # Only the owner merges candidates and applies the adaptive cutoff.
            if current_worker() == owner:
                for row, query in enumerate(query_chunk):
                    merged_candidates = flatten(
                        worker_candidates[row]
                        for worker_candidates in gathered_candidates
                    )
                    merged_candidates = sort_by_score(
                        merged_candidates,
                        descending=True,
                    )

                    # Repeat the mode-dependent filters after global merging.
                    candidates = []
                    for candidate in merged_candidates:
                        if (
                            query_is_training_set
                            and candidate.name == query.name
                        ):
                            continue
                        if (
                            query_is_training_set
                            and candidate.text == query.text
                        ):
                            continue
                        candidates.append(candidate)
                        if len(candidates) == maximum_k:
                            break

                    keep = select_by_knee(
                        [candidate.score for candidate in candidates],
                        maximum_k=maximum_k,
                    )
                    selected = candidates[:keep]
                    local_outputs[query.name] = {
                        "reference_text": query.text,
                        "retrieval_texts": [
                            candidate.text for candidate in selected
                        ],
                        "scores_for_merging": [
                            candidate.score for candidate in selected
                        ],
                    }

    # Resolve sampler-induced duplicate query keys, then remove internal scores.
    global_outputs = all_gather_object(local_outputs)
    global_outputs = merge_duplicate_queries(
        global_outputs,
        prefer="more_neighbors_then_higher_best_score",
    )
    return remove_internal_scores(global_outputs)


def run_retrieval_for_split(
    split,
    loaders,
    model,
    maximum_k,
    query_chunk_size,
    gallery_chunk_size,
    execution_mode,
):
    """Select the mode-specific query set while always using train as gallery."""
    if split == "train":
        query_loader = loaders["train"]
        query_is_training_set = True
    elif split == "validation":
        query_loader = loaders["validation"]
        query_is_training_set = False
    elif split == "test":
        query_loader = loaders["test"]
        query_is_training_set = False
    else:
        raise UnsupportedSplitError(split)

    results = distributed_knee_retrieval(
        training_loader=loaders["train"],
        query_loader=query_loader,
        model=model,
        maximum_k=maximum_k,
        query_chunk_size=query_chunk_size,
        gallery_chunk_size=gallery_chunk_size,
        query_is_training_set=query_is_training_set,
        execution_mode=execution_mode,
    )

    synchronize_workers()
    if current_worker() == 0:
        save_anonymized_json(results)
        report_neighbor_count_distribution(
            [len(item["retrieval_texts"]) for item in results.values()]
        )

