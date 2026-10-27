#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSeq2SeqLM

from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)

from .multimodal_projector.builder import build_vision_projector

"""
Short videos are padded to the longest length (repeating first/last frames) during batching.
img_mask blocks the padding in the visual encoder. These positions are kept when feeding
into the LLM to preserve training-time positional encodings, but fake frames are masked
by img_mask. Generation uses left padding, otherwise large batches would attend to fake frames.
"""
class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.mm_projector = build_vision_projector(config)

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    @staticmethod
    def use_flex_visual_encoder(model_args_or_config) -> bool:
        """True: Visual_Encoder_flex (raw keypoints). False: Visual_Encoder (precomputed features)."""
        return bool(getattr(model_args_or_config, "use_flex_visual_encoder", False))

    def get_slt_vision(self, model_args):
        if self.use_flex_visual_encoder(model_args):
            visual_encoder = Visual_Encoder_flex(model_args).to("cuda")
        else:
            visual_encoder = Visual_Encoder(model_args).to("cuda")
        return visual_encoder

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_dino(self, model_args):
        dino_model = Dino_EX(model_args).to(device="cuda")
        for param in dino_model.parameters():
            param.requires_grad = False
        return dino_model

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = getattr(model_args, "vision_tower", None)
        mm_vision_select_layer = model_args.mm_vision_select_layer  # -1
        mm_vision_select_feature = model_args.mm_vision_select_feature  # "patch"
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter  # None
        mm_patch_merge_type = model_args.mm_patch_merge_type  # "flat"

        self.config.mm_vision_tower = vision_tower

        # SLT training only uses slt_vision (sign features) + mm_projector, not the CLIP vision tower.
        # No CLIP tower is ever built; slt_vision handles all visual encoding.
        vision_tower = self.get_vision_tower()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(
            model_args, "mm_projector_type", "linear"
        )
        if vision_tower is not None:
            self.config.mm_hidden_size = vision_tower.hidden_size   # 1024
        else:
            self.config.mm_hidden_size = getattr(
                self.config, "mm_hidden_size", 1024
            )
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type
        self.config.use_flex_visual_encoder = self.use_flex_visual_encoder(model_args)

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {
                    k.split(keyword + ".")[1]: v
                    for k, v in weights.items()
                    if keyword in k
                }

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))

        self.slt_vision = self.get_slt_vision(model_args)
        self.dim_proj_face = nn.Linear(768, 1024)
        self.conv_1d_face = TemporalConv(input_size=1024, hidden_size=512, conv_type=2)  ###


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_dino(self):
        return self.get_model().get_dino()

    def get_MOEModel(self):
        return self.get_model().get_MOEModel()

    def get_dim_projector(self):
        return self.get_model().get_dim_projector()

    def encode_images(self, images, img_mask, imgs_batch, face, left_hand, right_hand):
        if LlavaMetaModel.use_flex_visual_encoder(self.get_model().config):
            # flex: images is dict(input_ids, src_length_batch, img_mask)
            image_features = self.get_model().slt_vision(images)
        else:
            image_features = self.get_model().slt_vision(
                images.to(device="cuda"), img_mask.to(device="cuda")
            )
        if face is not None:
            face = self.get_model().dim_proj_face(face)
            src_face = self.get_model().conv_1d_face(face)
            image_features = image_features + 0.3 * src_face
        
        projector = self.get_model().mm_projector

        image_features = image_features.to(
            device=next(projector.parameters()).device,
            dtype=next(projector.parameters()).dtype,
        )

        image_features = projector(image_features)
        return image_features

    def _select_valid_visual_tokens(self, image_features, img_mask, index):
        """Keep padded visual tokens; mask them in the LLM attention mask."""
        return image_features[index]

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
        img_mask=None,
        imgs_batch=None,
        face=None,
        left_hand=None,
        right_hand=None,
    ):
        # images [4, max_len, h_dim]
        if images is None or input_ids.shape[1] == 1:  # TODO:  it's amazing
            return (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                None,
                labels,
            )
        image_features = self.encode_images(images, img_mask, imgs_batch, face, left_hand, right_hand)
        _labels = labels    # shape of image_features: [b, 16, 4096]
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)   # [b, 52]
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)   # [b, 52] | 

        # Remove only text padding. Keep visual padding so text positions remain
        # identical to training; visual padding is masked below.
        text_attention_masks = [mask.bool() for mask in attention_mask]
        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, text_attention_masks)
        ]
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, text_attention_masks)
        ]
        text_attention_masks = [
            cur_attention_mask[cur_attention_mask]
            for cur_attention_mask in text_attention_masks
        ]

        new_input_embeds = []
        new_labels = []
        new_attention_masks = []
        cur_image_idx = 0
        image_special_token_idx = []
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()     # 1
            if num_images == 0:
                cur_image_features = self._select_valid_visual_tokens(image_features, img_mask, cur_image_idx)
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                new_attention_masks.append(text_attention_masks[batch_idx])
                cur_image_idx += 1
                continue
            
            # # [-1, 37, 52]
            image_token_indices = ([-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]])
            image_special_token_idx.append(image_token_indices[1])  # only for one image token   37
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):  # this loop need attention
                # image left section + right(input ids)
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                 # image left section + right (label ids)
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])

            split_sizes = [x.shape[0] for x in cur_labels_noim]  # [37, 14] 
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))  # text embedding for all parts except <image>  [51, 4096]
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)  # split back
            cur_new_input_embeds = []   # 37 + 16 + 17
            cur_new_labels = []     # 
            cur_new_attention_mask = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                cur_new_attention_mask.append(
                    text_attention_masks[batch_idx][
                        image_token_indices[i] + 1 : image_token_indices[i + 1]
                    ]
                )
                if i < num_images:
                    cur_image_features = self._select_valid_visual_tokens(
                        image_features, img_mask, cur_image_idx
                    )
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    if getattr(self.config, "slt_batch_inference", False) and img_mask is not None:
                        cur_new_attention_mask.append(
                            img_mask[cur_image_idx - 1]
                            .to(device=cur_image_features.device)
                            .bool()
                        )
                    else:
                        cur_new_attention_mask.append(
                            torch.ones(
                                cur_image_features.shape[0],
                                dtype=torch.bool,
                                device=cur_image_features.device,
                            )
                        )

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)  # [70, 4096]
            cur_new_labels = torch.cat(cur_new_labels)
            cur_new_attention_mask = torch.cat(cur_new_attention_mask)
            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            new_attention_masks.append(cur_new_attention_mask)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]   # tokenizer_model_max_length = 512

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)  # max_len=67
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len),IGNORE_INDEX,dtype=new_labels[0].dtype,device=new_labels[0].device)
        new_attention_mask_padded = torch.zeros(
            (batch_size, max_len),
            dtype=torch.bool,
            device=attention_mask.device,
        )
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        # Only batched inference changes the established training layout.
        use_left_padding = getattr(self.config, "slt_batch_inference", False) or (
            getattr(self.config, "tokenizer_padding_side", "right") == "left"
        )
        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]    # 67
            if use_left_padding:
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype,device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    new_attention_mask_padded[i, -cur_len:] = new_attention_masks[i]
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat( (cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]),dtype=cur_new_embed.dtype, device=cur_new_embed.device,),),dim=0,))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    new_attention_mask_padded[i, :cur_len] = new_attention_masks[i]
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if getattr(self.config, "slt_batch_inference", False):
            # Keep absolute positions identical to the original padded layout.
            # The visual padding is masked for attention, but must still occupy
            # its original positions for the text following the image token.
            position_ids = torch.arange(
                max_len, dtype=position_ids.dtype, device=position_ids.device
            ).unsqueeze(0).expand(batch_size, -1)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if getattr(self.config, "slt_batch_inference", False):
            attention_mask = new_attention_mask_padded
        else:
            attention_mask = new_attention_mask_padded.to(dtype=torch.float32)

        # Let Transformers derive position ids from the left-padded attention
        # mask and update them correctly during beam search/generation.
        if _position_ids is None and not getattr(self.config, "slt_batch_inference", False):
            position_ids = None

        return (
            None,
            position_ids,
            attention_mask,
            past_key_values,
            new_input_embeds,
            new_labels,
        )

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(
                    model_args.pretrain_mm_mlp_adapter, map_location="cpu"
                )
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[
                        -num_new_tokens:
                    ]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                    )
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


class TemporalConv(nn.Module):
    def __init__(self, input_size, hidden_size, conv_type=2):
        super(TemporalConv, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.conv_type = conv_type

        if self.conv_type == 0:
            self.kernel_size = ["K3"]
        elif self.conv_type == 1:
            self.kernel_size = ["K5", "P2"]
        elif self.conv_type == 2:
            # self.kernel_size = ['K5', "P2", 'K5', "P2"]
            self.kernel_size = ["K3", "P2", "K3", "P2"]

        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            if ks[0] == "P":
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == "K":
                modules.append(
                    nn.Conv1d(
                        input_sz,
                        self.hidden_size,
                        kernel_size=int(ks[1]),
                        stride=1,
                        padding=0,
                    )
                )
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))
        self.temporal_conv = nn.Sequential(*modules)

    def forward(self, x):
        x = self.temporal_conv(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


def _remap_visual_encoder_state_dict(state_dict):
    """Remap S2T best_model.pth or SFT non_lora_trainables.bin into Visual_Encoder keys."""
    new_state_dict = OrderedDict()

    # S2T checkpoint: {"model": {... "transformer.model.encoder.*", "conv_1d.*", "sign_emb.*" ...}}
    if isinstance(state_dict, dict) and "model" in state_dict and isinstance(state_dict["model"], dict):
        for k, v in state_dict["model"].items():
            if "transformer.model.encoder" in k:
                new_state_dict[k.replace("transformer.model.encoder.", "encoder.")] = v
            if "conv_1d" in k:
                new_state_dict[k.replace("conv_1d", "conv_1d")] = v
            if "sign_emb" in k:
                new_state_dict[k.replace("sign_emb", "sign_emb")] = v
        if new_state_dict:
            return new_state_dict

    # Flat SFT non_lora_trainables.bin: base_model.model.model.slt_vision.{encoder,conv_1d,sign_emb}.*
    slt_vision_prefixes = (
        "base_model.model.model.slt_vision.",
        "base_model.model.slt_vision.",
        "model.slt_vision.",
        "slt_vision.",
    )
    for k, v in state_dict.items():
        nk = k
        for prefix in slt_vision_prefixes:
            if nk.startswith(prefix):
                nk = nk[len(prefix) :]
                break
        if nk.startswith(("encoder.", "conv_1d.", "sign_emb.")):
            new_state_dict[nk] = v
    return new_state_dict


class Visual_Encoder(nn.Module):
    def __init__(self, model_args):  # model_args
        super(Visual_Encoder, self).__init__()

        self.conv_1d = TemporalConv(input_size=1024, hidden_size=512, conv_type=2)  ###
        encoder = AutoModelForSeq2SeqLM.from_pretrained(model_args.v_init_model)

        modules = []
        modules.append(nn.Linear(512, encoder.config.d_model))
        modules.append(nn.LayerNorm(encoder.config.d_model))
        modules.append(nn.ReLU(inplace=True))
        self.sign_emb = nn.Sequential(*modules)
        self.encoder = encoder.get_encoder()

        state_dict = torch.load(model_args.pretrained_ve, map_location="cpu")
        new_state_dict = _remap_visual_encoder_state_dict(state_dict)
        if not new_state_dict:
            raise ValueError(
                f"No Visual_Encoder weights found in {model_args.pretrained_ve}. "
                "Expected S2T best_model.pth (state_dict['model']) or "
                "SFT non_lora_trainables.bin (slt_vision.encoder/conv_1d/sign_emb)."
            )
        ret = self.load_state_dict(new_state_dict, strict=True)
        # print("Missing keys: \n", ret.missing_keys)
        # print("Unexpected keys: \n", ret.unexpected_keys)


    def forward(self, src_input, img_mask):
        conv1d_out = self.conv_1d(src_input)
        src = self.sign_emb(conv1d_out)
        src = self.encoder(inputs_embeds=src, attention_mask=img_mask).last_hidden_state
        return src


class Visual_Encoder_flex(nn.Module):
    """Online CoSign1s encoding from raw skeleton keypoints (T, N, 7)."""

    def __init__(self, model_args):
        super(Visual_Encoder_flex, self).__init__()

        self.ve_args = {
            "in_channels": 3,
            "split": [9, 30, 51, 59, 77],
            "temporal_kernel": 5,
            "hidden_size": 1024,
            "modes": ["body", "hand21", "mouth_8", "face"],
            "level": "0",
            "adaptive": True,
        }
        from .cosign1s import CoSign1s

        self.visual_module = CoSign1s(**self.ve_args)

        self.conv_1d = TemporalConv(input_size=1024, hidden_size=512, conv_type=2)
        encoder = AutoModelForSeq2SeqLM.from_pretrained(model_args.v_init_model)

        modules = []
        modules.append(nn.Linear(512, encoder.config.d_model))
        modules.append(nn.LayerNorm(encoder.config.d_model))
        modules.append(nn.ReLU(inplace=True))
        self.sign_emb = nn.Sequential(*modules)
        self.encoder = encoder.get_encoder()

        state_dict = torch.load(model_args.pretrained_ve, map_location="cpu")
        new_state_dict = OrderedDict()
        if isinstance(state_dict, dict) and isinstance(state_dict.get("model"), dict):
            for k, v in state_dict["model"].items():
                if "transformer.model.encoder" in k:
                    new_state_dict[k.replace("transformer.model.encoder.", "encoder.")] = v
                if "conv_1d" in k:
                    new_state_dict[k.replace("conv_1d", "conv_1d")] = v
                if "sign_emb" in k:
                    new_state_dict[k.replace("sign_emb", "sign_emb")] = v
                if "visual_module" in k:
                    new_state_dict[k.replace("visual_module", "visual_module")] = v
            self.load_state_dict(new_state_dict, strict=True)
        else:
            marker = "slt_vision."
            for k, v in state_dict.items():
                pos = k.find(marker)
                if pos != -1:
                    new_state_dict[k[pos + len(marker) :]] = v
            if len(new_state_dict) == 0:
                new_state_dict = state_dict
            ret = self.load_state_dict(new_state_dict, strict=False)
            print(
                f"[Visual_Encoder_flex] loaded {len(new_state_dict)} tensors from "
                f"'{model_args.pretrained_ve}' (non-strict); "
                f"missing={len(ret.missing_keys)}, unexpected={len(ret.unexpected_keys)}"
            )
            if len(ret.unexpected_keys) > 0:
                print(
                    f"[Visual_Encoder_flex] WARNING unexpected keys example: "
                    f"{ret.unexpected_keys[:5]}"
                )

    def forward(self, src_input):
        visual_ret = self.visual_module(
            src_input["input_ids"], src_input["src_length_batch"]
        )
        src = visual_ret["fusion"]
        conv1d_out = self.conv_1d(src)
        src = self.sign_emb(conv1d_out)
        src = self.encoder(
            inputs_embeds=src, attention_mask=src_input["img_mask"]
        ).last_hidden_state
        return src