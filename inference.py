"""Inference entry point for SignRAG (sign language translation)."""

import argparse
import os

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

import utils
from data import create_infer_loader
from llava.model.language_model.llava_qwen import LlavaQwen3ForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description="SignRAG inference")
    parser.add_argument("--model_dir", required=True, help="Path to merged SignRAG model directory")
    parser.add_argument("--non_lora_trainables", default=None, help="Path to non_lora_trainables.bin (default: <model_dir>/non_lora_trainables.bin)")
    parser.add_argument("--v_init_model", required=True, help="Path to the mBART-based visual encoder init model (mytran_*_512)")
    parser.add_argument("--data_path", required=True, help="Path to a gzipped pickle dataset (e.g. rag_dep.test)")
    parser.add_argument("--keypoint_path", required=True, help="Path to pose_format directory (raw keypoints)")
    parser.add_argument("--data_version", default="CSL_Daily", help="Pose data version")
    parser.add_argument("--language", default="zh", help="Target language code (zh/en/de)")
    parser.add_argument("--output_dir", default="./outputs", help="Directory for inference artifacts")
    parser.add_argument("--infer_batch_size", type=int, default=2)
    parser.add_argument("--infer_num_workers", type=int, default=4)
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return parser.parse_args()


def main():
    args = parse_args()

    if args.torch_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif args.torch_dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed:
        import torch.distributed as dist

        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    non_lora_path = args.non_lora_trainables or os.path.join(args.model_dir, "non_lora_trainables.bin")

    model_args = type("ModelArgs", (), {})()
    model_args.model_name_or_path = args.model_dir
    model_args.version = "qwen_rag_dep"
    model_args.vision_tower = None
    model_args.mm_vision_select_layer = -1
    model_args.mm_vision_select_feature = "patch"
    model_args.pretrain_mm_mlp_adapter = None
    model_args.mm_patch_merge_type = "flat"
    model_args.mm_projector_type = "mlp2x_gelu"
    model_args.mm_use_im_start_end = False
    model_args.mm_use_im_patch_token = False
    model_args.mm_fea_size = 512
    model_args.lora_modules = "q_proj,v_proj"
    model_args.v_init_model = args.v_init_model
    model_args.sft_model = os.path.dirname(non_lora_path)
    model_args.load_ve_from_non_lora = True
    model_args.use_flex_visual_encoder = True
    model_args.freeze_slt_vision = True
    model_args.attn_implementation = args.attn_implementation
    model_args.tune_mm_mlp_adapter = False
    model_args.pretrained_ve = non_lora_path

    data_args = type("DataArgs", (), {})()
    data_args.data_path = args.data_path
    data_args.keypoint_path = args.keypoint_path
    data_args.data_version = args.data_version
    data_args.infer_batch_size = args.infer_batch_size
    data_args.infer_num_workers = args.infer_num_workers
    data_args.language = args.language
    data_args.image_aspect_ratio = "pad"

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        model_max_length=2048,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Model
    model_init_kwargs = {
        "attn_implementation": args.attn_implementation,
        "torch_dtype": torch_dtype,
        "use_cache": False,
        "device_map": ({"": device} if (distributed or device != "cuda") else "cuda"),
        "trust_remote_code": True,
    }

    model = LlavaQwen3ForCausalLM.from_pretrained(args.model_dir, **model_init_kwargs)

    # Vision modules
    model.get_model().initialize_vision_modules(model_args=model_args, fsdp=None)
    model.config.use_flex_visual_encoder = model_args.use_flex_visual_encoder
    vision_tower = model.get_vision_tower()
    if vision_tower is not None:
        vision_tower.to(dtype=torch_dtype, device="cpu")
    model.get_model().mm_projector.to(dtype=torch_dtype, device=device)
    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    # Load non-lora trainables (visual encoder + projector). Remap Peft-style
    # keys (base_model.model.*) to the merged (non-Peft) model layout.
    if os.path.isfile(non_lora_path):
        state_dict = torch.load(non_lora_path, map_location="cpu")
        remapped = {}
        for k, v in state_dict.items():
            nk = k
            if nk.startswith("base_model.model."):
                nk = nk[len("base_model.model."):]
            elif nk.startswith("base_model."):
                nk = nk[len("base_model."):]
            remapped[nk] = v
        ret = model.load_state_dict(remapped, strict=False)
        print("Missing keys:", len(ret.missing_keys))
        print("Unexpected keys:", len(ret.unexpected_keys))
        if ret.unexpected_keys:
            print("Unexpected (first 5):", ret.unexpected_keys[:5])
    else:
        raise FileNotFoundError(f"non_lora_trainables.bin not found: {non_lora_path}")

    model.eval()
    if str(device).startswith("cuda"):
        model.to(device)

    # Data
    questions = utils.load_dataset_file(args.data_path)
    if isinstance(questions, dict):
        questions = list(questions.values())

    if distributed:
        sharded_indices = list(range(rank, len(questions), world_size))
        sharded_questions = [questions[i] for i in sharded_indices]
    else:
        sharded_indices = list(range(len(questions)))
        sharded_questions = questions
    print(f"[rank {rank}] world_size={world_size} distributed={distributed} n_questions={len(questions)} n_sharded={len(sharded_questions)}", flush=True)

    loader = create_infer_loader(sharded_questions, tokenizer, data_args, batch_size=args.infer_batch_size)

    model.config.slt_batch_inference = True
    model.config.tokenizer_padding_side = "left"

    local_results = []
    offset = 0
    for batch in tqdm(loader, disable=(rank != 0 if distributed else False)):
        input_ids, attention_mask, image_tensor, text_token, image_mask = batch
        with torch.inference_mode():
            output = model.generate(
                input_ids.to(device),
                attention_mask=attention_mask.to(device),
                img_mask=image_mask.to(device),
                images=utils.move_images_to_device(image_tensor, device),
                do_sample=False,
                num_beams=args.beam_size,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        text_token = text_token.to(device)
        bs = input_ids.shape[0]
        for i in range(bs):
            local_results.append((sharded_indices[offset + i], output[i, :].detach().cpu(), text_token[i, :].detach().cpu()))
        offset += bs

    if distributed:
        import torch.distributed as dist

        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_results)
        dist.barrier()
        if rank != 0:
            return
        all_results = []
        for r in gathered:
            if r:
                all_results.extend(r)
        all_results.sort(key=lambda x: x[0])
    else:
        all_results = local_results

    tgt_pres = [x[1] for x in all_results]
    tgt_refs = [x[2] for x in all_results]

    from torch.nn.utils.rnn import pad_sequence

    tgt_pres = pad_sequence(tgt_pres, batch_first=True, padding_value=tokenizer.pad_token_id)
    tgt_refs = pad_sequence(tgt_refs, batch_first=True, padding_value=tokenizer.pad_token_id)

    tgt_pres = tokenizer.batch_decode(tgt_pres, skip_special_tokens=True)
    tgt_refs = tokenizer.batch_decode(tgt_refs, skip_special_tokens=True)
    tgt_pres = utils.strip_qwen_thinking_batch(tgt_pres)

    # Metrics
    if args.language == "zh":
        tgt_pres_metric = [' '.join(list(r.replace(" ", '').replace("\n", ''))) for r in tgt_pres]
        tgt_refs_metric = [' '.join(list(r.replace(" ", ''))) for r in tgt_refs]
    else:
        tgt_pres_metric = tgt_pres
        tgt_refs_metric = tgt_refs

    n_empty = sum(1 for p in tgt_pres_metric if p is None or not str(p).strip())
    if n_empty:
        print(f"[evaluate] {n_empty}/{len(tgt_pres_metric)} empty hypotheses; using ' ' placeholder")
    tgt_pres_m = [p if (p is not None and str(p).strip()) else " " for p in tgt_pres_metric]
    tgt_refs_m = [r if (r is not None and str(r).strip()) else " " for r in tgt_refs_metric]

    bleu, rouge = utils.translation_performance(tgt_refs_m, tgt_pres_m)
    metrics = {
        "Bleu_1": bleu["bleu1"],
        "Bleu_2": bleu["bleu2"],
        "Bleu_3": bleu["bleu3"],
        "Bleu_4": bleu["bleu4"],
        "ROUGE_L": rouge,
    }

    print("Metrics: ", metrics)

    # Save artifacts
    with open(os.path.join(args.output_dir, "tmp_pres.txt"), "w") as f:
        for p in tgt_pres:
            f.write((p.replace("\n", "") if p is not None else "None") + "\n")
    with open(os.path.join(args.output_dir, "tmp_refs.txt"), "w") as f:
        for r in tgt_refs:
            f.write((r if r is not None else "None") + "\n")
    with open(os.path.join(args.output_dir, "result_bleu.txt"), "w") as f:
        f.write("Metrics: " + str(metrics) + "\n")

    print("Saved artifacts to", args.output_dir)


if __name__ == "__main__":
    main()
