# ============================================================
# train_qlora.py
#
# Qwen3-VL-8B-Instruct
# 4-bit QLoRA baseline
# ============================================================

import argparse
import inspect
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint

from src.data_utils import (
    ManifestDataset,
    Qwen3VLCollator,
)
from src.profiler import EfficiencyCallback
from src.progress import report


class DriveProgressCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        step = state.global_step or 0
        max_steps = state.max_steps or 0
        percent = round(100.0 * step / max_steps, 2) if max_steps else 0
        report(
            phase="train",
            message="training started" if step == 0 else f"resumed from step {step}",
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
            message="training",
            step=step,
            max_steps=max_steps,
            epoch=logs.get("epoch", state.epoch),
            loss=logs.get("loss"),
            learning_rate=logs.get("learning_rate"),
            eval_loss=logs.get("eval_loss"),
            percent=percent,
        )

    def on_train_end(self, args, state, control, **kwargs):
        report(phase="train", message="training finished", percent=100)


def build_training_args(**kwargs):
    """Transformers v5 removed warmup_ratio and some eval arg aliases."""
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

    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
    )

    parser.add_argument(
        "--train_manifest",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--eval_manifest",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_dir",
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
        "--lora_rank",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
    )

    parser.add_argument(
        "--epochs",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--grad_accum",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--save_steps",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--eval_steps",
        type=int,
        default=1000,
    )

    return parser.parse_args()


def default_data_root():
    return str(ROOT / "data")


def main():
    args = parse_args()
    data_root = args.data_root or default_data_root()

    # --------------------------------------------------------
    # QLoRA 4-bit config
    # --------------------------------------------------------

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # --------------------------------------------------------
    # Processor
    # --------------------------------------------------------

    report(phase="setup", message="loading processor and 4-bit model")
    processor = AutoProcessor.from_pretrained(
        args.model_name,
    )

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = (
            processor.tokenizer.eos_token
        )

    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None and isinstance(getattr(image_processor, "size", None), dict):
        image_processor.size["longest_edge"] = min(
            int(image_processor.size.get("longest_edge", 1024 * 1024)),
            1024 * 1024,
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        # Turn this on if environment supports FA2:
        # attn_implementation="flash_attention_2",
    )

    # QLoRA preparation
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    model.config.use_cache = False

    # --------------------------------------------------------
    # LoRA
    # --------------------------------------------------------

    lora_alpha = (
        args.lora_alpha
        if args.lora_alpha is not None
        else args.lora_rank * 2
    )

    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(
        model,
        peft_config,
    )

    print("\nTrainable parameters:")
    model.print_trainable_parameters()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    report(phase="data", message="loading train manifest")
    train_dataset = ManifestDataset(
        args.train_manifest,
        data_root=data_root,
    )
    report(
        phase="data",
        message=f"train examples: {len(train_dataset):,}",
        train_examples=len(train_dataset),
    )

    report(phase="data", message="loading eval manifest")
    eval_dataset = ManifestDataset(
        args.eval_manifest,
        data_root=data_root,
    )
    report(
        phase="data",
        message=f"eval examples: {len(eval_dataset):,}",
        eval_examples=len(eval_dataset),
    )

    collator = Qwen3VLCollator(
        processor,
        max_length=args.max_length,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=[
            DriveProgressCallback(),
            EfficiencyCallback(
                os.path.join(args.output_dir, "efficiency_train.json"),
                method="qlora",
            ),
        ],
    )

    last_checkpoint = get_last_checkpoint(args.output_dir)
    if last_checkpoint:
        print("Resuming from", last_checkpoint)
        report(phase="train", message=f"resuming from {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # --------------------------------------------------------
    # Save final LoRA adapter
    # --------------------------------------------------------

    final_dir = os.path.join(
        args.output_dir,
        "final_adapter"
    )

    trainer.model.save_pretrained(
        final_dir
    )

    processor.save_pretrained(
        final_dir
    )

    report(phase="save", message=f"saved adapter: {final_dir}", percent=100)
    print("\nSaved final adapter:")
    print(final_dir)


if __name__ == "__main__":
    main()
