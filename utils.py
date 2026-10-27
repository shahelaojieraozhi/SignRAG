"""Shared utilities for SignRAG inference and evaluation."""

import gzip
import math
import pickle
import random
import re
from typing import List, Union

import numpy as np
import torch

from metrics import Rouge, sacrebleu
from rouge import Rouge as SLT_Rouge

_THINK_BLOCK_RE = re.compile(r"</think>.*?</think>", flags=re.DOTALL)


def strip_qwen_thinking(text: Union[str, None]) -> Union[str, None]:
    if text is None:
        return None
    text = _THINK_BLOCK_RE.sub("", text)
    text = text.replace(" thinking", "").replace("</think>", "")
    return text.strip()


def strip_qwen_thinking_batch(texts: List[str]) -> List[str]:
    return [strip_qwen_thinking(t) for t in texts]


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, skeleton):
        for t in self.transforms:
            skeleton = t(skeleton)
        return skeleton


class ToTensor:
    def __call__(self, skeleton):
        if isinstance(skeleton, np.ndarray):
            skeleton = torch.from_numpy(skeleton).float()
        return skeleton


class TemporalRescale:
    def __init__(self, temp_scaling=0.2) -> None:
        self.min_len = 32
        self.max_len = 230
        self.L = 1.0 - temp_scaling
        self.U = 1.0 + temp_scaling

    def __call__(self, clip):
        vid_len = len(clip)
        new_len = int(vid_len * (self.L + (self.U - self.L) * np.random.random()))
        if new_len < self.min_len:
            new_len = self.min_len
        if new_len > self.max_len:
            new_len = self.max_len
        if (new_len - 4) % 4 != 0:
            new_len += 4 - (new_len - 4) % 4
        if new_len <= vid_len:
            index = sorted(random.sample(range(vid_len), new_len))
        else:
            index = sorted(random.choices(range(vid_len), k=new_len))
        return clip[index]


def collate_slt_visual(src_batch, pad_token_id, use_flex_visual_encoder=False):
    src_length_batch = torch.tensor([len(s) for s in src_batch], dtype=torch.long)
    left_pad = 4
    batch_max_length = int(src_length_batch.max())

    if use_flex_visual_encoder:
        video_length = torch.LongTensor(
            [int(math.ceil(float(l) / 4.0) * 4 + 8) for l in src_length_batch]
        )
        conv_right_pad = int(math.ceil(batch_max_length / 4.0)) * 4 - batch_max_length + 4
    else:
        video_length = torch.LongTensor([int(l) + 8 for l in src_length_batch])
        conv_right_pad = 4

    batch_max_length = batch_max_length + left_pad + conv_right_pad
    padded = []
    for sample in src_batch:
        extra = batch_max_length - len(sample) - left_pad
        rest = sample.shape[1:]
        padded.append(
            torch.cat(
                [
                    sample[0][None].expand(left_pad, *rest),
                    sample,
                    sample[-1][None].expand(extra, *rest),
                ],
                dim=0,
            )
        )
    stacked = torch.stack(padded, dim=0)

    mask_gen = []
    for i in (video_length / 4 - 2).int():
        mask_gen.append(torch.ones([int(i)]) + 7)
    mask_gen = torch.nn.utils.rnn.pad_sequence(
        mask_gen, padding_value=pad_token_id, batch_first=True
    )
    image_padding_mask = (mask_gen != pad_token_id).long()

    if use_flex_visual_encoder:
        images = {
            "input_ids": stacked,
            "src_length_batch": video_length,
            "img_mask": image_padding_mask,
        }
    else:
        images = stacked
    return images, image_padding_mask, batch_max_length


def move_images_to_device(images, device):
    if isinstance(images, dict):
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in images.items()}
    if images is None:
        return None
    return images.to(device)


def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        return pickle.load(f)


def rouge(references, hypotheses, level="word"):
    if level == "char":
        hyp = [list(x) for x in hypotheses]
        ref = [list(x) for x in references]
    else:
        hyp = [x.split() for x in hypotheses]
        ref = [x.split() for x in references]
    a = Rouge.rouge([" ".join(x) for x in hyp], [" ".join(x) for x in ref])
    return a["rouge_l/f_score"] * 100


def bleu(references, hypotheses, level="word"):
    if level == "char":
        references = [" ".join(list(r)) for r in references]
        hypotheses = [" ".join(list(r)) for r in hypotheses]
    bleu_scores = sacrebleu.raw_corpus_bleu(sys_stream=hypotheses, ref_streams=[references]).scores
    return {"bleu" + str(n + 1): bleu_scores[n] for n in range(len(bleu_scores))}


def sableu(references, hypotheses, tokenizer):
    bleu_scores = sacrebleu.corpus_bleu(
        sys_stream=hypotheses,
        ref_streams=[references],
        tokenize=tokenizer,
    ).scores
    return {"bleu" + str(n + 1): bleu_scores[n] for n in range(len(bleu_scores))}


def translation_performance(txt_ref, txt_hyp):
    txt_hyp = [h if (h is not None and str(h).strip() != "") else " " for h in txt_hyp]
    txt_ref = [r if (r is not None and str(r).strip() != "") else " " for r in txt_ref]
    rouge_scorer = SLT_Rouge()
    scores = rouge_scorer.get_scores(txt_hyp, txt_ref, avg=True)
    scores["rouge-l"]["f"] = scores["rouge-l"]["f"] * 100
    sableu_dict = sableu(references=txt_ref, hypotheses=txt_hyp, tokenizer="13a")
    return sableu_dict, float(scores["rouge-l"]["f"])
