# pyright: reportUndefinedVariable=false
"""Detailed pseudocode for face and hand region preprocessing."""


PARTS = ("face", "left_hand", "right_hand")


def initialize_region_image_encoder(device):
    """Load a pretrained DINO-style image transformer and its processor."""
    processor = load_pretrained_image_processor("DINOv2-base")
    encoder = load_pretrained_image_transformer("DINOv2-base")
    encoder = move_to_device(encoder, device)
    encoder.set_evaluation_mode()
    return processor, encoder


def square_box_from_landmarks(
    normalized_landmarks,
    image_width,
    image_height,
    expansion_scale=1.2,
):
    """
    Convert normalized landmarks into an enlarged square crop.

    The square preserves region aspect ratio during later image processing.
    The returned right/bottom coordinates are exclusive slicing boundaries.
    """
    if normalized_landmarks is None or len(normalized_landmarks) == 0:
        return None

    pixel_points = [
        (
            landmark.x * image_width,
            landmark.y * image_height,
        )
        for landmark in normalized_landmarks
    ]
    x_min, x_max = minimum_x(pixel_points), maximum_x(pixel_points)
    y_min, y_max = minimum_y(pixel_points), maximum_y(pixel_points)

    width = x_max - x_min
    height = y_max - y_min
    side = max(width, height)
    if side <= NUMERICAL_EPSILON:
        return None

    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    half_side = side * expansion_scale / 2

    # Floor the inclusive start and ceil the exclusive end.
    x1 = clip(floor(center_x - half_side), 0, image_width - 1)
    y1 = clip(floor(center_y - half_side), 0, image_height - 1)
    x2 = clip(ceil(center_x + half_side), 1, image_width)
    y2 = clip(ceil(center_y + half_side), 1, image_height)

    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return None
    return x1, y1, x2, y2


def detect_regions(frame_bgr, face_tracker, hand_tracker, expansion_scale):
    """Detect one face and at most one crop for each handedness label."""
    image_height, image_width = frame_bgr.shape[:2]
    frame_rgb = convert_color(frame_bgr, source="BGR", target="RGB")
    crops = {part: None for part in PARTS}
    detected = {part: False for part in PARTS}

    face_result = face_tracker.process(frame_rgb)
    if face_result.has_face_landmarks:
        face_landmarks = face_result.first_face_landmarks
        face_box = square_box_from_landmarks(
            face_landmarks,
            image_width,
            image_height,
            expansion_scale,
        )
        if face_box is not None:
            crops["face"] = copy_crop(frame_bgr, face_box)
            detected["face"] = True

    hand_result = hand_tracker.process(frame_rgb)
    if hand_result.has_landmarks_and_handedness:
        for landmarks, handedness in zip(
            hand_result.hand_landmarks,
            hand_result.handedness,
        ):
            hand_box = square_box_from_landmarks(
                landmarks,
                image_width,
                image_height,
                expansion_scale,
            )
            if hand_box is None:
                continue

            if handedness == "Left":
                part = "left_hand"
            elif handedness == "Right":
                part = "right_hand"
            else:
                continue

            crops[part] = copy_crop(frame_bgr, hand_box)
            detected[part] = True

    return crops, detected


def fill_missing_region(
    part,
    frame_index,
    detected_crop,
    detected_this_frame,
    last_valid_crop,
    pending_initial_frames,
    emit_crop,
):
    """
    Produce one crop per frame after the first valid detection.

    Initial misses are queued and backfilled with the first valid crop.
    Later misses reuse the most recently detected crop itself, rather than
    applying an old bounding box to a new frame.
    """
    if detected_this_frame:
        for pending_index in pending_initial_frames[part]:
            emit_crop(part, pending_index, detected_crop)
        pending_initial_frames[part].clear()

        emit_crop(part, frame_index, detected_crop)
        last_valid_crop[part] = detected_crop
        return

    if last_valid_crop[part] is not None:
        emit_crop(part, frame_index, copy(last_valid_crop[part]))
    else:
        pending_initial_frames[part].append(frame_index)


def crop_face_and_hands_from_video(
    video,
    output_writer,
    expansion_scale=1.2,
    maximum_hands=2,
    detection_threshold=0.5,
    tracking_threshold=0.5,
):
    """Save temporally completed face/left-hand/right-hand crop sequences."""
    face_tracker = FaceMeshTracker(
        streaming=True,
        maximum_faces=1,
        refined_landmarks=True,
        minimum_detection_confidence=detection_threshold,
        minimum_tracking_confidence=tracking_threshold,
    )
    hand_tracker = HandTracker(
        streaming=True,
        maximum_hands=maximum_hands,
        model_complexity=1,
        minimum_detection_confidence=detection_threshold,
        minimum_tracking_confidence=tracking_threshold,
    )

    last_valid_crop = {part: None for part in PARTS}
    pending_initial_frames = {part: [] for part in PARTS}
    detection_flags = {part: [] for part in PARTS}

    for frame_index, frame_bgr in enumerate(video, start=1):
        crops, detected = detect_regions(
            frame_bgr,
            face_tracker,
            hand_tracker,
            expansion_scale,
        )

        for part in PARTS:
            # Metadata records true detector output, not imputation status.
            detection_flags[part].append(int(detected[part]))
            fill_missing_region(
                part=part,
                frame_index=frame_index,
                detected_crop=crops[part],
                detected_this_frame=detected[part],
                last_valid_crop=last_valid_crop,
                pending_initial_frames=pending_initial_frames,
                emit_crop=output_writer.save_image,
            )

    metadata = {}
    number_of_frames = len(detection_flags["face"])
    for part in PARTS:
        number_detected = sum(detection_flags[part])
        metadata[part] = {
            "per_frame_detection": detection_flags[part],
            "number_detected": number_detected,
            "valid_ratio": safe_divide(number_detected, number_of_frames),
        }
    output_writer.save_metadata(metadata)


def encode_crop_batch(
    crop_images,
    frame_indices,
    image_processor,
    image_encoder,
    device,
    output_by_frame,
):
    """Preprocess crop images and retain the image transformer's CLS token."""
    if len(crop_images) == 0:
        return

    model_inputs = image_processor(crop_images)
    model_inputs = move_to_device(model_inputs, device)

    with no_gradient():
        hidden_states = image_encoder(**model_inputs).last_hidden_state
        cls_features = hidden_states[:, 0, :]

    for row, frame_index in enumerate(frame_indices):
        output_by_frame[frame_index] = move_to_cpu(cls_features[row])

    crop_images.clear()
    frame_indices.clear()


def extract_region_features(
    video,
    image_processor,
    image_encoder,
    device,
    batch_size=32,
    expansion_scale=1.2,
    maximum_hands=2,
    detection_threshold=0.5,
    tracking_threshold=0.5,
):
    """
    Detect, temporally complete, and encode the three regions in one pass.

    Parts never detected in the complete video are omitted from the output.
    Every retained part must have exactly one feature for every video frame.
    """
    face_tracker = FaceMeshTracker(
        streaming=True,
        maximum_faces=1,
        refined_landmarks=True,
        minimum_detection_confidence=detection_threshold,
        minimum_tracking_confidence=tracking_threshold,
    )
    hand_tracker = HandTracker(
        streaming=True,
        maximum_hands=maximum_hands,
        model_complexity=1,
        minimum_detection_confidence=detection_threshold,
        minimum_tracking_confidence=tracking_threshold,
    )

    detected_at_least_once = {part: False for part in PARTS}
    last_valid_crop = {part: None for part in PARTS}
    pending_initial_frames = {part: [] for part in PARTS}
    feature_by_frame = {part: {} for part in PARTS}
    crop_batches = {part: [] for part in PARTS}
    index_batches = {part: [] for part in PARTS}

    def enqueue(part, frame_index, crop_bgr):
        crop_rgb = convert_color(crop_bgr, source="BGR", target="RGB")
        crop_batches[part].append(to_image_object(crop_rgb))
        index_batches[part].append(frame_index)

        if len(crop_batches[part]) >= batch_size:
            encode_crop_batch(
                crop_batches[part],
                index_batches[part],
                image_processor,
                image_encoder,
                device,
                feature_by_frame[part],
            )

    number_of_frames = 0
    for frame_index, frame_bgr in enumerate(video, start=1):
        number_of_frames = frame_index
        crops, detected = detect_regions(
            frame_bgr,
            face_tracker,
            hand_tracker,
            expansion_scale,
        )

        for part in PARTS:
            if detected[part]:
                detected_at_least_once[part] = True

            fill_missing_region(
                part=part,
                frame_index=frame_index,
                detected_crop=crops[part],
                detected_this_frame=detected[part],
                last_valid_crop=last_valid_crop,
                pending_initial_frames=pending_initial_frames,
                emit_crop=enqueue,
            )

    # Flush partial image-encoder batches only for available parts.
    for part in PARTS:
        if detected_at_least_once[part]:
            encode_crop_batch(
                crop_batches[part],
                index_batches[part],
                image_processor,
                image_encoder,
                device,
                feature_by_frame[part],
            )

    available_parts = [
        part for part in PARTS if detected_at_least_once[part]
    ]
    if len(available_parts) == 0:
        raise NoRegionDetectedError

    features = {}
    feature_dimension = image_encoder.hidden_size
    for part in available_parts:
        if len(feature_by_frame[part]) != number_of_frames:
            raise IncompleteFeatureSequenceError(part)

        sequence = zeros([number_of_frames, feature_dimension])
        for one_based_frame_index, vector in feature_by_frame[part].items():
            sequence[one_based_frame_index - 1] = vector
        features[part] = sequence

    # Shape: [number_of_available_parts, number_of_frames, feature_dimension].
    return {
        "number_of_frames": number_of_frames,
        "available_parts": available_parts,
        "feature_dimension_per_part": feature_dimension,
        "features": stack(
            [features[part] for part in available_parts],
            axis="part",
        ),
    }


def preprocess_video_collection(video_collection, image_model):
    """Sequentially process videos and serialize one compressed payload each."""
    image_model.set_evaluation_mode()
    for video in progress(video_collection):
        payload = extract_region_features(
            video=video,
            image_processor=image_model.processor,
            image_encoder=image_model.encoder,
            device=image_model.device,
        )
        save_compressed_feature_payload(video.identifier, payload)

