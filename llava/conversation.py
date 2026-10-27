"""Prompts and chat-template helpers for SignRAG (retrieval-dependent SLT)."""

from typing import Optional

_ZH_DATASETS = {
    "csl-daily",
    "csl_daily",
    "csl-news",
    "csl_news",
    "buu-sl",
    "buu_sl",
    "buu",
}


def infer_dataset_from_paths(
    data_path_prefix: Optional[str] = None,
    image_feature_path: Optional[str] = None,
    keypoint_path: Optional[str] = None,
) -> str:
    combined = " ".join(p for p in (data_path_prefix, image_feature_path, keypoint_path) if p)
    combined = combined.lower().replace("_", "-")
    if not combined:
        return "default"
    if "p14t" in combined or "phonexi" in combined:
        return "p14t"
    if "csl-daily" in combined or "csl-news" in combined or "cslnews" in combined:
        return "csl-daily"
    if "buu" in combined:
        return "csl-daily"
    if "openasl" in combined or "open-asl" in combined:
        return "openasl"
    if "h2s" in combined:
        return "h2s"
    if "csl" in combined:
        return "csl-daily"
    return "default"


def _normalize_dataset(dataset_name: Optional[str] = None, data_path_prefix: Optional[str] = None) -> str:
    return (dataset_name or infer_dataset_from_paths(data_path_prefix=data_path_prefix)).lower()


def get_qwen_rag_dep_system(
    dataset_name: Optional[str] = None,
    data_path_prefix: Optional[str] = None,
) -> str:
    dataset = _normalize_dataset(dataset_name, data_path_prefix)
    if dataset in _ZH_DATASETS:
        return (
            "助手根据手语视频生成翻译文本。若提供检索到的相似文本，仅作为参考证据，可能存在噪声；"
            "助手需以视频信息为主，综合判断，不能直接照抄或编造未给信息。"
        )
    if dataset in {"h2s", "openasl", "default"}:
        return (
            "The assistant generates the translation text based on the sign language video. "
            "If similar retrieved texts are provided, they serve only as reference evidence and may contain noise; "
            "the assistant should prioritize the video information, make a comprehensive judgment, "
            "and should not directly copy or fabricate information not provided."
        )
    if dataset == "p14t":
        return (
            "Der Assistent erzeugt den Uebersetzungstext auf Basis des Gebaerdensprachvideos. "
            "Wenn aehnliche abgerufene Texte bereitgestellt werden, dienen sie nur als Referenzbelege und koennen Rauschen enthalten; "
            "der Assistent soll die Videoinformation priorisieren, eine umfassende Bewertung vornehmen und keine nicht gegebenen Informationen direkt uebernehmen oder erfinden."
        )
    return get_qwen_rag_dep_system(dataset_name="h2s")


def get_qwen_system_for_version(
    version: str = "qwen_rag_dep",
    dataset_name: Optional[str] = None,
    data_path_prefix: Optional[str] = None,
    image_feature_path: Optional[str] = None,
    keypoint_path: Optional[str] = None,
) -> str:
    if version != "qwen_rag_dep":
        raise ValueError(f"Unsupported conversation version for SignRAG: {version!r}")
    resolved = dataset_name or infer_dataset_from_paths(
        data_path_prefix=data_path_prefix,
        image_feature_path=image_feature_path,
        keypoint_path=keypoint_path,
    )
    return get_qwen_rag_dep_system(dataset_name=resolved, data_path_prefix=data_path_prefix)


def apply_qwen_chat_template(
    tokenizer,
    messages,
    add_generation_prompt: bool = True,
    enable_thinking: bool = False,
) -> str:
    kwargs = dict(tokenize=False, add_generation_prompt=add_generation_prompt)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)
