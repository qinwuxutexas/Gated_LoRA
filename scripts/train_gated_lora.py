# ============================================================
# train_gated_lora.py
#
# Same data / hparams / collator as train_qlora.py.
# Adapter: y = Wx + (alpha/r) B(g ⊙ Ax)
# ============================================================

import argparse
import inspect
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from peft import prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint

from src.data_utils import ManifestDataset, Qwen3VLCollator
from src.gated_lora import (
    TARGET_LEAF_NAMES,
    adapter_parameter_summary,
    configure_trainable_parameters,
    inject_gated_lora,
    load_gated_adapter,
    mark_quantized_model_has_adapters,
    move_adapters_to_dtype,
    print_gate_summary,
    print_trainable_parameters,
    save_gated_adapter,
    gated_adapter_state_dict,
    get_gate_statistics,
)
from src.profiler import EfficiencyCallback
from src.progress import report


class GatedTrainer(Trainer):
    """Save/load only gated adapter tensors. The 4-bit base is not PEFT."""

    def _save(self, output_dir=None, state_dict=None):
        output_dir = Path(output_dir or self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(gated_adapter_state_dict(self.model), output_dir / "gated_adapter.pt")
        torch.save(self.args, output_dir / "training_args.bin")
        (output_dir / "gate_statistics.json").write_text(
            json.dumps(get_gate_statistics(self.model), indent=2),
            encoding="utf-8",
        )
        print("Saved gated adapter checkpoint:", output_dir / "gated_adapter.pt")

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        adapter = Path(resume_from_checkpoint) / "gated_adapter.pt"
        if not adapter.exists():
            print("No gated_adapter.pt in", resume_from_checkpoint, "— starting from scratch.")
            return
        model = model or self.model
        load_gated_adapter(model, resume_from_checkpoint)


def prune_incomplete_gated_checkpoints(output_dir):
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return
    for path in sorted(output_dir.glob("checkpoint-*")):
        if not path.is_dir():
            continue
        adapter = path / "gated_adapter.pt"
        state = path / "trainer_state.json"
        if adapter.exists() and state.exists():
            continue
        print("Removing incomplete gated checkpoint:", path)
        shutil.rmtree(path, ignore_errors=True)


def last_complete_gated_checkpoint(output_dir):
    prune_incomplete_gated_checkpoints(output_dir)
    last = get_last_checkpoint(output_dir)
    if last is None:
        return None
    adapter = Path(last) / "gated_adapter.pt"
    state = Path(last) / "trainer_state.json"
    if adapter.exists() and state.exists():
        return last
    print("Ignoring incomplete checkpoint:", last)
    return None


class DriveProgressCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        step = state.global_step or 0
        max_steps = state.max_steps or 0
        percent = round(100.0 * step / max_steps, 2) if max_steps else 0
        report(
            phase="train",
            message="gated training started" if step == 0 else f"resumed from step {step}",
            step=step,
            max_steps=max_steps,
            epoch=state.epoch or 0,
            percent=percent,
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        max_steps = state.max_steps or 0
        step = state.global_step or 0
        percent = round(100.0 * step / max_steps, 2) if max_steps else None
        report(
            phase="train",
            message="gated training",
            step=step,
            max_steps=max_steps,
            epoch=logs.get("epoch", state.epoch),
            loss=logs.get("loss"),
            learning_rate=logs.get("learning_rate"),
            eval_loss=logs.get("eval_loss"),
            percent=percent,
        )

    def on_train_end(self, args, state, control, **kwargs):
        report(phase="train", message="gated training finished", percent=100)


def build_training_args(**kwargs):
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "warmup_ratio" in params:
        kwargs.pop("warmup_steps", None)
    else:
        kwargs.pop("warmup_ratio", None)
    if "eval_strategy" in params:
        kwargs.pop("evaluation_strategy", None)
    else:
        kwargs.pop("eval_strategy", None)
    kwargs = {k: v for k, v in kwargs.items() if k in params}
    return TrainingArguments(**kwargs)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--train_manifest", type=str, required=True)
    parser.add_argument("--eval_manifest", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--gate_init", type=float, default=2.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=1000)
    return parser.parse_args()


def default_data_root():
    return str(ROOT / "data")


def main():
    args = parse_args()
    data_root = args.data_root or default_data_root()
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else args.lora_rank * 2

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    report(phase="setup", message="loading processor and 4-bit model")
    processor = AutoProcessor.from_pretrained(args.model_name)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None and isinstance(getattr(image_processor, "size", None), dict):
        image_processor.size["longest_edge"] = min(
            int(image_processor.size.get("longest_edge", 1024 * 1024)),
            1024 * 1024,
        )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
    model.config.use_cache = False

    replaced = inject_gated_lora(
        model,
        rank=args.lora_rank,
        alpha=lora_alpha,
        dropout=args.lora_dropout,
        gate_init=args.gate_init,
    )
    move_adapters_to_dtype(model, torch.bfloat16)
    configure_trainable_parameters(model)
    mark_quantized_model_has_adapters(model)
    print_trainable_parameters(model)
    print_gate_summary(model)

    report(phase="data", message="loading train manifest")
    train_dataset = ManifestDataset(args.train_manifest, data_root=data_root)
    report(phase="data", message=f"train examples: {len(train_dataset):,}")
    eval_dataset = ManifestDataset(args.eval_manifest, data_root=data_root)
    report(phase="data", message=f"eval examples: {len(eval_dataset):,}")

    collator = Qwen3VLCollator(processor, max_length=args.max_length)

    training_args = build_training_args(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        bf16=True,
        fp16=False,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        warmup_steps=0.03,
        weight_decay=0.01,
        logging_steps=10,
        logging_first_step=True,
        save_steps=args.save_steps,
        eval_strategy="steps",
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_total_limit=None,
        load_best_model_at_end=False,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        report_to="none",
    )

    efficiency_path = os.path.join(args.output_dir, "efficiency_train.json")
    trainer = GatedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=[
            DriveProgressCallback(),
            EfficiencyCallback(efficiency_path, method="gated_qlora"),
        ],
    )
    last_checkpoint = last_complete_gated_checkpoint(args.output_dir)
    if last_checkpoint:
        print("Resuming from", last_checkpoint)
        report(phase="train", message=f"resuming from {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("No complete gated checkpoint; starting from step 0.")
        trainer.train(resume_from_checkpoint=False)

    final_dir = os.path.join(args.output_dir, "final_adapter")
    save_gated_adapter(
        model=trainer.model,
        processor=processor,
        output_dir=final_dir,
        config_dict={
            "base_model": args.model_name,
            "method": "gated_qlora",
            "rank": args.lora_rank,
            "alpha": lora_alpha,
            "dropout": args.lora_dropout,
            "gate_init": args.gate_init,
            "target_modules": sorted(TARGET_LEAF_NAMES),
            "vision_frozen": True,
            "language_only": True,
            "replaced_modules": replaced,
            "parameter_summary": adapter_parameter_summary(trainer.model),
        },
    )
    print_gate_summary(trainer.model)
    report(phase="save", message=f"saved gated adapter: {final_dir}", percent=100)
    print("Saved gated adapter:", final_dir)


if __name__ == "__main__":
    main()
