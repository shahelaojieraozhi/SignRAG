"""Data loading utilities for SignRAG inference."""

import os
import pickle
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import utils
from llava import conversation as conversation_lib
from llava.constants import IMAGE_TOKEN_INDEX
from llava.mm_utils import tokenizer_image_token


class KeypointLoader:
    """Load and normalize raw skeleton keypoints (flex visual encoder)."""

    def __init__(self, keypoint_path, train_aug=False, data_version=None):
        if not keypoint_path:
            raise ValueError("--keypoint_path is required for SignRAG inference")
        self.path_kpt = keypoint_path
        self.data_version = data_version
        self.used_part = ["body", "hand21", "mouth_8", "face"]
        self.split = [9, 30, 51, 59, 77]
        self.norm_point = [3, 9, 30, 53, 76]
        self.norm_div = (256 - 1) / 2.0
        self.pose_idx = []
        for part in self.used_part:
            if part == "body":
                self.pose_idx += [0, 1, 2, 5, 6, 7, 8, 9, 10]
            elif part == "hand21":
                self.pose_idx += [i - 1 for i in range(92, 113)]
                self.pose_idx += [i - 1 for i in range(113, 134)]
            elif part == "mouth_8":
                self.pose_idx += [i - 1 for i in range(84, 92)]
            elif part == "face":
                self.pose_idx += [i - 1 for i in range(24, 41)]
                self.pose_idx += [53]
        if train_aug:
            self.data_aug = utils.Compose([utils.TemporalRescale(0.2), utils.ToTensor()])
        else:
            self.data_aug = utils.ToTensor()
        if self.data_version:
            print(f"[KeypointLoader] data_version={self.data_version}, keypoint_path={self.path_kpt}")

    def _pose_name(self, line):
        name = line.get("name", line["image"])
        for suffix in (".h5", ".pt", ".pth", ".npy", ".pkl"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        return name

    def _read_pose(self, line):
        name = self._pose_name(line)
        if self.data_version == "CSL_Daily":
            stem = name.split("/")[-1]
            pose_path = os.path.join(self.path_kpt, stem + ".pkl")
            with open(pose_path, "rb") as handle:
                pose = pickle.load(handle)
            keypoints = np.concatenate(pose["keypoints"], axis=0)
            scores = np.concatenate(pose["scores"], axis=0)[..., None]
            src_kpt_sample = np.concatenate([keypoints, scores], axis=-1).astype(np.float32)
            if "start" in pose and "end" in pose:
                start, end = int(pose["start"]), int(pose["end"])
                src_kpt_sample = src_kpt_sample[start:end]
            return src_kpt_sample
        pose_path = os.path.join(self.path_kpt, name + ".npy")
        return np.load(pose_path)

    def simple_normalize(self, origin_input_data):
        conf = origin_input_data[:, :, 6]
        origin_input_data = origin_input_data / self.norm_div - 1
        input_data = origin_input_data[:, :, 0:2]
        if self.norm_point is not None:
            index = 0
            for part in self.used_part:
                if index == 0:
                    start, end = 0, self.split[0]
                else:
                    start, end = self.split[index - 1], self.split[index]
                if part == "body":
                    input_data[:, start:end] = (
                        input_data[:, start:end]
                        - input_data[0, self.norm_point[index] : self.norm_point[index] + 2].mean(0)[None, None]
                    )
                elif part == "hand21":
                    input_data[:, start:end] = (
                        input_data[:, start:end] - input_data[:, self.norm_point[index]][:, None, :]
                    )
                    index += 1
                    start, end = self.split[index - 1], self.split[index]
                    input_data[:, start:end] = (
                        input_data[:, start:end] - input_data[:, self.norm_point[index]][:, None, :]
                    )
                else:
                    input_data[:, start:end] = (
                        input_data[:, start:end] - input_data[:, self.norm_point[index]][:, None, :]
                    )
                index += 1
        return torch.cat([input_data, origin_input_data[:, :, 2:6], conf.unsqueeze(-1)], dim=-1)

    def load(self, line):
        src_kpt_sample = self._read_pose(line)
        conf = src_kpt_sample[:, self.pose_idx, 2]
        src_kpt_sample = src_kpt_sample[:, self.pose_idx, :2]
        total_motion = np.zeros(src_kpt_sample.shape[0:2] + (4,))
        total_motion[1:, :, 0:2] = src_kpt_sample[1:, :, 0:2] - src_kpt_sample[0:-1, :, 0:2]
        total_motion[0:-1, :, 2:4] = src_kpt_sample[:-1, :, 0:2] - src_kpt_sample[1:, :, 0:2]
        final = np.concatenate([src_kpt_sample, total_motion, conf[:, :, None]], axis=-1)
        src_kpt_sample = self.data_aug(final)
        return self.simple_normalize(src_kpt_sample)

    def length(self, line):
        return int(self._read_pose(line).shape[0])


class CustomDatasetInfer(Dataset):
    def __init__(self, questions, tokenizer, data_args):
        self.questions = questions
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.max_length = 300
        self.kpt_loader = KeypointLoader(
            data_args.keypoint_path,
            train_aug=False,
            data_version=getattr(data_args, "data_version", None),
        )
        self.dataset_name = conversation_lib.infer_dataset_from_paths(
            data_path_prefix=getattr(data_args, "data_path", None),
            keypoint_path=getattr(data_args, "keypoint_path", None),
        )
        self.system_content = conversation_lib.get_qwen_system_for_version(
            "qwen_rag_dep",
            dataset_name=self.dataset_name,
            data_path_prefix=getattr(data_args, "data_path", None),
            keypoint_path=getattr(data_args, "keypoint_path", None),
        )

    def __getitem__(self, index):
        line = self.questions[index]
        instr = line["conversations"][0]["value"]
        messages = [{"role": "system", "content": self.system_content}, {"role": "user", "content": instr}]
        prompt = conversation_lib.apply_qwen_chat_template(
            self.tokenizer, messages, add_generation_prompt=True, enable_thinking=False
        )

        src_sample = self.kpt_loader.load(line)
        if len(src_sample) > self.max_length:
            tmp = sorted(random.sample(range(len(src_sample)), k=self.max_length))
            src_sample = torch.stack([src_sample[i, :] for i in tmp])

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        text_label = line["conversations"][1]["value"]
        text_token = self.tokenizer(text_label, return_tensors="pt")["input_ids"][0]
        return input_ids, src_sample, text_token

    def __len__(self):
        return len(self.questions)

    def collate_fn(self, batch):
        input_ids, image_tensors, label_text_tokens = zip(*batch)
        src_batch = list(image_tensors)

        image_tensors, image_padding_mask, _ = utils.collate_slt_visual(
            src_batch, self.tokenizer.pad_token_id, use_flex_visual_encoder=True
        )

        input_ids_list = [x.squeeze(0) if x.dim() == 2 else x for x in input_ids]
        bs = len(input_ids_list)
        max_len = max(x.numel() for x in input_ids_list)
        pad_id = self.tokenizer.pad_token_id
        input_ids = torch.full((bs, max_len), fill_value=pad_id, dtype=input_ids_list[0].dtype)
        attention_mask = torch.zeros((bs, max_len), dtype=torch.long)
        for i, ids in enumerate(input_ids_list):
            length = ids.numel()
            input_ids[i, -length:] = ids
            attention_mask[i, -length:] = 1

        label_text_tokens = pad_sequence(list(label_text_tokens), batch_first=True, padding_value=pad_id)
        return input_ids, attention_mask, image_tensors, label_text_tokens, image_padding_mask


def create_infer_loader(questions, tokenizer, data_args, batch_size=None):
    from torch.utils.data import DataLoader
    bs = batch_size or data_args.infer_batch_size
    dataset = CustomDatasetInfer(questions, tokenizer, data_args)
    return DataLoader(
        dataset,
        batch_size=bs,
        num_workers=data_args.infer_num_workers,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
    )
