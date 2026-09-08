# ============================================================
# eval_qlora.py
# Load a trained LoRA adapter and run dataset-specific VQA eval.
# ============================================================

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)

from src.data_utils import (
    answer_from_record,
    open_rgb_image,
    question_from_record,
)
from src.gated_lora import (
    inject_gated_lora,
    load_gated_adapter,
    move_adapters_to_dtype,
)
from src.metrics import METRIC_NAMES, score_prediction
from src.profiler import InferenceTimer, write_json
from src.progress import report


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default="/content/Qwen3-VL-8B-Instruct",
    )

    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--adapter_type",
        type=str,
        default="peft",
        choices=("peft", "gated"),
    )

    parser.add_argument(
        "--eval_manifest",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Root that contains chartqa/, gqa/, etc. "
             "Defaults to <repo>/data when running from this repo.",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
    )

    return parser.parse_args()


def default_data_root():
    return str(ROOT / "data")


def load_model_and_processor(model_name, lora_path, adapter_type="peft"):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    processor = AutoProcessor.from_pretrained(model_name)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if adapter_type == "gated":
        cfg_path = Path(lora_path) / "gated_config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        inject_gated_lora(
            model,
            rank=cfg.get("rank", 8),
            alpha=cfg.get("alpha", 16),
            dropout=cfg.get("dropout", 0.05),
            gate_init=cfg.get("gate_init", 2.0),
        )
        move_adapters_to_dtype(model, torch.bfloat16)
        load_gated_adapter(model, lora_path)
    else:
        model = PeftModel.from_pretrained(model, lora_path)

    model.eval()
    print(f"Loaded {adapter_type} adapter from:")
    print(lora_path)
    return model, processor


def load_records(eval_manifest, max_samples=None):
    records = []

    with open(eval_manifest, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))

    if max_samples is not None:
        records = records[:max_samples]

    print(f"Loaded {len(records):,} evaluation examples.")
    return records


def generate_answer(model, processor, image_path, question, device, max_new_tokens):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": str(image_path),
                },
                {
                    "type": "text",
                    "text": str(question),
                },
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in inputs.items()
    }

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )

    prompt_len = inputs["input_ids"].shape[1]
    output_ids = generated_ids[:, prompt_len:]

    return processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def main():
    args = parse_args()
    data_root = args.data_root or default_data_root()

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    report(phase="setup", message="loading base model and adapter")
    model, processor = load_model_and_processor(
        args.model_name,
        args.lora_path,
        adapter_type=args.adapter_type,
    )
    report(phase="data", message="loading eval manifest")
    records = load_records(args.eval_manifest, args.max_samples)
    report(
        phase="eval",
        message=f"eval examples: {len(records):,}",
        max_steps=len(records),
        step=0,
        percent=0,
    )

    device = next(model.parameters()).device
    dataset_scores = defaultdict(float)
    dataset_counts = defaultdict(int)
    results = []
    timer = InferenceTimer(method=args.adapter_type)

    with torch.inference_mode():
        for r in tqdm(records):
            dataset = (r.get("dataset") or "").lower()
            try:
                _rgb, image_path = open_rgb_image(r.get("image"), data_root)
            except FileNotFoundError:
                continue

            question = question_from_record(r)
            gt = answer_from_record(r)

            if image_path is None or not question or gt is None:
                continue

            prediction = timer.time_call(
                lambda: generate_answer(
                    model,
                    processor,
                    image_path,
                    question,
                    device,
                    args.max_new_tokens,
                )
            )

            score = score_prediction(dataset, prediction, gt)

            dataset_scores[dataset] += score
            dataset_counts[dataset] += 1

            results.append({
                "id": r.get("id"),
                "original_ids": r.get("original_ids"),
                "dataset": dataset,
                "question": question,
                "ground_truth": gt,
                "prediction": prediction,
                "score": float(score),
            })

            if len(results) == 1 or len(results) % 20 == 0:
                total = max(len(records), 1)
                report(
                    phase="eval",
                    message=f"scored {len(results):,}/{total:,}",
                    step=len(results),
                    max_steps=total,
                    percent=round(100.0 * len(results) / total, 2),
                    last_score=float(score),
                    last_dataset=dataset,
                )

    with open(args.output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n")
    print("=" * 72)
    print("EVALUATION RESULTS")
    print("=" * 72)

    for dataset in sorted(dataset_counts.keys()):
        n = dataset_counts[dataset]
        mean_score = dataset_scores[dataset] / n
        metric_name = METRIC_NAMES.get(dataset, "Score")

        print(
            f"{dataset:18s} "
            f"{metric_name:18s}: "
            f"{mean_score:.4f} "
            f"({n:,} samples)"
        )

    print("=" * 72)
    report(
        phase="eval",
        message="evaluation finished",
        step=len(results),
        max_steps=len(records),
        percent=100,
        output_file=args.output_file,
    )
    scores = {
        dataset: dataset_scores[dataset] / dataset_counts[dataset]
        for dataset in dataset_counts
    }
    efficiency_path = str(Path(args.output_file).with_name(
        Path(args.output_file).stem + "_efficiency.json"
    ))
    write_json(
        efficiency_path,
        timer.summary({
            "adapter_type": args.adapter_type,
            "scores": scores,
            "counts": dict(dataset_counts),
        }),
    )

    print("\nPredictions saved to:")
    print(args.output_file)
    print("Efficiency report:")
    print(efficiency_path)


if __name__ == "__main__":
    main()
