# ============================================================
# Reference dump. The runnable gated method now lives in:
#   src/gated_lora.py
#   scripts/train_gated_lora.py
#   scripts/eval_qlora.py --adapter_type gated
#
# Delta: y = Wx + (alpha/r) B [ g .* (A x) ]
# ============================================================

import os
import re
import gc
import json
import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    set_seed,
)

from peft import prepare_model_for_kbit_training


# ============================================================
# CONFIG / ARGUMENTS
# ============================================================

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
        "--rank",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=64.0,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--gate_init",
        type=float,
        default=2.0,
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
        default=1,
    )

    parser.add_argument(
        "--grad_accum",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


# ============================================================
# DATASET
# ============================================================

def answer_to_text(answer):

    if answer is None:
        return ""

    if isinstance(answer, str):
        return answer.strip()

    if isinstance(answer, (int, float)):
        return str(answer)

    if isinstance(answer, dict):

        for key in [
            "answer",
            "text",
            "label",
            "target",
        ]:

            if key in answer:
                return answer_to_text(answer[key])

        return str(answer)

    if isinstance(answer, list):

        if len(answer) == 0:
            return ""

        # list[str]
        if all(
            isinstance(x, str)
            for x in answer
        ):
            return answer[0].strip()

        # list[dict]
        for item in answer:

            if isinstance(item, dict):

                for key in [
                    "answer",
                    "text",
                    "label",
                ]:

                    if key in item:
                        return str(
                            item[key]
                        ).strip()

        return str(answer[0])

    return str(answer)


def get_question(record):

    q = record.get("question")

    if q is not None:
        return str(q).strip()

    meta = record.get(
        "metadata",
        {}
    )

    for key in [
        "question",
        "query",
        "prompt",
        "instruction",
        "sentence",
        "text",
    ]:

        if key in meta:
            return str(
                meta[key]
            ).strip()

    return ""


def get_answer(record):

    answer = record.get("answer")

    if answer is not None:
        return answer_to_text(
            answer
        )

    meta = record.get(
        "metadata",
        {}
    )

    for key in [
        "answer",
        "answers",
        "label",
        "labels",
        "target",
        "response",
    ]:

        if key in meta:
            return answer_to_text(
                meta[key]
            )

    return ""


class ManifestDataset(Dataset):

    def __init__(
        self,
        manifest_path,
    ):

        self.records = []

        manifest_path = Path(
            manifest_path
        )

        with manifest_path.open(
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                record = json.loads(
                    line
                )

                image_path = record.get(
                    "image"
                )

                question = get_question(
                    record
                )

                answer = get_answer(
                    record
                )

                if not image_path:
                    continue

                if not question:
                    continue

                if not answer:
                    continue

                self.records.append(
                    record
                )

        print(
            f"Loaded {len(self.records):,} examples "
            f"from {manifest_path}"
        )

    def __len__(self):
        return len(
            self.records
        )

    def __getitem__(
        self,
        idx,
    ):

        record = self.records[
            idx
        ]

        image_path = record[
            "image"
        ]

        if isinstance(
            image_path,
            list,
        ):
            image_path = image_path[0]

        return {
            "id": record.get("id"),
            "dataset": record.get(
                "dataset",
                "unknown",
            ),
            "image_path": str(
                image_path
            ),
            "question": get_question(
                record
            ),
            "answer": get_answer(
                record
            ),
        }


# ============================================================
# COLLATOR
#
# batch_size=1 is deliberate for the first stable version.
# It avoids complicated multimodal batching.
# Gradient accumulation supplies the effective batch size.
# ============================================================

SYSTEM_PROMPT = (
    "Answer the question based on the image. "
    "Give a concise answer without additional explanation."
)


class Qwen3VLCollator:

    def __init__(
        self,
        processor,
        max_length=2048,
    ):

        self.processor = processor
        self.max_length = max_length

    def __call__(
        self,
        examples,
    ):

        if len(examples) != 1:

            raise ValueError(
                "This initial gated-Qwen implementation "
                "uses per_device_train_batch_size=1. "
                "Use gradient accumulation for larger "
                "effective batch size."
            )

        ex = examples[0]

        image_path = ex[
            "image_path"
        ]

        question = ex[
            "question"
        ]

        answer = ex[
            "answer"
        ]

        # ----------------------------------------------------
        # prompt only
        # ----------------------------------------------------

        prompt_messages = [

            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },

            {
                "role": "user",
                "content": [

                    {
                        "type": "image",
                        "image": image_path,
                    },

                    {
                        "type": "text",
                        "text": question,
                    },

                ],
            },

        ]

        prompt_inputs = (
            self.processor
            .apply_chat_template(

                prompt_messages,

                tokenize=True,

                add_generation_prompt=True,

                return_dict=True,

                return_tensors="pt",

            )
        )

        prompt_len = (
            prompt_inputs[
                "input_ids"
            ].shape[1]
        )

        # ----------------------------------------------------
        # full training conversation
        # ----------------------------------------------------

        full_messages = [

            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },

            {
                "role": "user",
                "content": [

                    {
                        "type": "image",
                        "image": image_path,
                    },

                    {
                        "type": "text",
                        "text": question,
                    },

                ],
            },

            {
                "role": "assistant",
                "content": [

                    {
                        "type": "text",
                        "text": answer,
                    }

                ],
            },

        ]

        batch = (
            self.processor
            .apply_chat_template(

                full_messages,

                tokenize=True,

                add_generation_prompt=False,

                return_dict=True,

                return_tensors="pt",

            )
        )

        # ----------------------------------------------------
        # truncate text sequence if necessary
        # ----------------------------------------------------

        seq_len = batch[
            "input_ids"
        ].shape[1]

        if seq_len > self.max_length:

            batch[
                "input_ids"
            ] = batch[
                "input_ids"
            ][
                :,
                :self.max_length
            ]

            if (
                "attention_mask"
                in batch
            ):

                batch[
                    "attention_mask"
                ] = batch[
                    "attention_mask"
                ][
                    :,
                    :self.max_length
                ]

        # ----------------------------------------------------
        # labels
        #
        # Only assistant answer contributes to LM loss.
        # ----------------------------------------------------

        labels = batch[
            "input_ids"
        ].clone()

        effective_prompt_len = min(
            prompt_len,
            labels.shape[1],
        )

        labels[
            :,
            :effective_prompt_len
        ] = -100

        pad_id = (
            self.processor
            .tokenizer
            .pad_token_id
        )

        if pad_id is not None:

            labels[
                batch["input_ids"]
                == pad_id
            ] = -100

        batch[
            "labels"
        ] = labels

        return batch


# ============================================================
# GATED LoRA LAYER
#
# Existing frozen linear:
#
#       y_base = W x
#
# Adapter:
#
#       z = A x
#       z = g .* z
#       y_lora = B z
#
#       y = y_base + alpha/r * y_lora
#
# ============================================================

class GatedLoRALinear(
    nn.Module
):

    def __init__(
        self,
        base_layer,
        rank=32,
        alpha=64.0,
        dropout=0.05,
        gate_init=2.0,
    ):

        super().__init__()

        self.base_layer = (
            base_layer
        )

        self.rank = int(
            rank
        )

        self.alpha = float(
            alpha
        )

        self.scaling = (
            self.alpha
            / self.rank
        )

        self.in_features = (
            base_layer
            .in_features
        )

        self.out_features = (
            base_layer
            .out_features
        )

        # Freeze Qwen base layer.
        for parameter in (
            self.base_layer
            .parameters()
        ):
            parameter.requires_grad = (
                False
            )

        # A:
        #
        # [hidden] -> [rank]

        self.lora_A = nn.Linear(
            self.in_features,
            self.rank,
            bias=False,
        )

        # B:
        #
        # [rank] -> [hidden/out]

        self.lora_B = nn.Linear(
            self.rank,
            self.out_features,
            bias=False,
        )

        self.dropout = nn.Dropout(
            dropout
        )

        # One gate for each rank component.

        self.gate_logits = (
            nn.Parameter(

                torch.full(
                    (self.rank,),
                    float(
                        gate_init
                    ),
                    dtype=torch.float32,
                )

            )
        )

        # Standard LoRA-style initialization.

        nn.init.kaiming_uniform_(
            self.lora_A.weight,
            a=math.sqrt(5),
        )

        nn.init.zeros_(
            self.lora_B.weight
        )

    def gate_values(
        self,
    ):

        return torch.sigmoid(
            self.gate_logits
        )

    def forward(
        self,
        x,
    ):

        # Quantized frozen base path.

        base_output = (
            self.base_layer(x)
        )

        # Adapter operates in adapter dtype.

        adapter_dtype = (
            self.lora_A
            .weight
            .dtype
        )

        x_adapter = (
            self.dropout(
                x
            )
            .to(
                adapter_dtype
            )
        )

        z = self.lora_A(
            x_adapter
        )

        gates = (
            self.gate_values()
            .to(
                dtype=z.dtype,
                device=z.device,
            )
        )

        z = (
            z
            * gates
        )

        delta = (
            self.lora_B(
                z
            )
        )

        delta = (
            delta
            * self.scaling
        )

        delta = delta.to(
            base_output.dtype
        )

        return (
            base_output
            + delta
        )


# ============================================================
# LANGUAGE-MODEL TARGET MODULES
#
# We deliberately do NOT adapt model.visual.
# ============================================================

TARGET_LEAF_NAMES = {

    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",

    "gate_proj",
    "up_proj",
    "down_proj",

}


def get_parent_module(
    model,
    module_name,
):

    parts = module_name.split(
        "."
    )

    child_name = parts[-1]

    parent_name = ".".join(
        parts[:-1]
    )

    if parent_name:

        parent = (
            model
            .get_submodule(
                parent_name
            )
        )

    else:
        parent = model

    return (
        parent,
        child_name,
    )


def is_language_target(
    module_name,
    module,
):

    leaf_name = (
        module_name
        .split(".")[-1]
    )

    if (
        leaf_name
        not in TARGET_LEAF_NAMES
    ):
        return False

    # Explicitly exclude vision tower.

    if (
        "visual"
        in module_name.lower()
    ):
        return False

    # Qwen3-VL HF structure:
    # model.language_model....

    if (
        "language_model"
        not in module_name
    ):
        return False

    if not hasattr(
        module,
        "in_features",
    ):
        return False

    if not hasattr(
        module,
        "out_features",
    ):
        return False

    return True


def inject_gated_lora(
    model,
    rank=32,
    alpha=64.0,
    dropout=0.05,
    gate_init=2.0,
):

    # Snapshot first because we mutate
    # the model during replacement.

    module_list = list(
        model.named_modules()
    )

    targets = []

    for (
        module_name,
        module,
    ) in module_list:

        if is_language_target(
            module_name,
            module,
        ):

            targets.append(
                (
                    module_name,
                    module,
                )
            )

    if len(targets) == 0:

        print(
            "\nNo targets found."
        )

        print(
            "\nCandidate q_proj modules:"
        )

        for name, module in module_list:

            if name.endswith(
                "q_proj"
            ):
                print(
                    name,
                    type(module),
                )

        raise RuntimeError(
            "Could not locate Qwen3-VL "
            "language-model projection layers."
        )

    print(
        f"\nFound {len(targets)} "
        f"language linear layers "
        f"for gated LoRA."
    )

    for (
        module_name,
        base_module,
    ) in targets:

        (
            parent,
            child_name,
        ) = get_parent_module(
            model,
            module_name,
        )

        gated_layer = (
            GatedLoRALinear(

                base_layer=base_module,

                rank=rank,

                alpha=alpha,

                dropout=dropout,

                gate_init=gate_init,

            )
        )

        setattr(
            parent,
            child_name,
            gated_layer,
        )

    print(
        "\nFirst replaced modules:"
    )

    for (
        name,
        _
    ) in targets[:20]:

        print(
            "  ",
            name,
        )

    return [
        name
        for (
            name,
            _
        ) in targets
    ]


# ============================================================
# FREEZE BASE MODEL
# ============================================================

def configure_trainable_parameters(
    model,
):

    for (
        name,
        parameter,
    ) in model.named_parameters():

        trainable = (

            ".lora_A."
            in name

            or ".lora_B."
            in name

            or name.endswith(
                "gate_logits"
            )

        )

        parameter.requires_grad = (
            trainable
        )


def print_trainable_parameters(
    model,
):

    total = 0
    trainable = 0

    adapter = 0
    gate_params = 0

    for (
        name,
        parameter,
    ) in model.named_parameters():

        n = parameter.numel()

        total += n

        if parameter.requires_grad:

            trainable += n

            if (
                "gate_logits"
                in name
            ):

                gate_params += n

            else:

                adapter += n

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PARAMETER SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Total parameters:     "
        f"{total:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable:,}"
    )

    print(
        f"A/B parameters:       "
        f"{adapter:,}"
    )

    print(
        f"Gate parameters:      "
        f"{gate_params:,}"
    )

    print(
        f"Trainable percentage: "
        f"{100.0 * trainable / total:.6f}%"
    )


# ============================================================
# GATE STATISTICS
# ============================================================

def get_gate_statistics(
    model,
):

    rows = []

    for (
        name,
        module,
    ) in model.named_modules():

        if isinstance(
            module,
            GatedLoRALinear,
        ):

            gates = (
                module
                .gate_values()
                .detach()
                .float()
                .cpu()
            )

            rows.append(
                {
                    "module": name,
                    "rank": module.rank,
                    "mean_gate":
                        float(
                            gates.mean()
                        ),
                    "min_gate":
                        float(
                            gates.min()
                        ),
                    "max_gate":
                        float(
                            gates.max()
                        ),
                    "gates":
                        gates.tolist(),
                }
            )

    return rows


def print_gate_summary(
    model,
):

    stats = get_gate_statistics(
        model
    )

    if not stats:
        return

    all_gates = torch.tensor(
        [
            gate
            for row in stats
            for gate in row["gates"]
        ],
        dtype=torch.float32,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "GATE SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Number of gated layers: "
        f"{len(stats)}"
    )

    print(
        f"Mean gate: "
        f"{all_gates.mean().item():.6f}"
    )

    print(
        f"Min gate:  "
        f"{all_gates.min().item():.6f}"
    )

    print(
        f"Max gate:  "
        f"{all_gates.max().item():.6f}"
    )

    for threshold in [
        0.1,
        0.25,
        0.5,
        0.75,
    ]:

        active = (
            all_gates
            >= threshold
        ).float().mean()

        print(
            f"Fraction gate >= "
            f"{threshold:.2f}: "
            f"{active.item():.4f}"
        )


# ============================================================
# SAVE ONLY GATED ADAPTER
#
# We don't want to save the whole 8B base model every time.
# ============================================================

def gated_adapter_state_dict(
    model,
):

    state = {}

    for (
        name,
        tensor,
    ) in model.state_dict().items():

        if (

            ".lora_A."
            in name

            or ".lora_B."
            in name

            or name.endswith(
                "gate_logits"
            )

        ):

            state[
                name
            ] = (
                tensor
                .detach()
                .cpu()
            )

    return state


def save_gated_adapter(
    model,
    processor,
    output_dir,
    config_dict,
):

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    adapter_path = (
        output_dir
        / "gated_adapter.pt"
    )

    torch.save(
        gated_adapter_state_dict(
            model
        ),
        adapter_path,
    )

    with (
        output_dir
        / "gated_config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            config_dict,
            f,
            indent=2,
        )

    gate_stats = (
        get_gate_statistics(
            model
        )
    )

    with (
        output_dir
        / "gate_statistics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            gate_stats,
            f,
            indent=2,
        )

    processor.save_pretrained(
        output_dir
    )

    print(
        "\nSaved gated adapter to:"
    )

    print(
        adapter_path
    )


# ============================================================
# LOAD GATED ADAPTER
# ============================================================

def load_gated_adapter(
    model,
    adapter_path,
):

    adapter_path = Path(
        adapter_path
    )

    state = torch.load(
        adapter_path,
        map_location="cpu",
        weights_only=True,
    )

    missing, unexpected = (
        model.load_state_dict(
            state,
            strict=False,
        )
    )

    # Missing base-model keys are expected.
    # Unexpected adapter keys are not.

    if unexpected:

        raise RuntimeError(
            f"Unexpected gated adapter keys:\n"
            f"{unexpected[:20]}"
        )

    print(
        f"\nLoaded gated adapter:"
        f"\n{adapter_path}"
    )


# ============================================================
# CALLBACK-LIKE TRAINER
#
# No new loss.
# outputs.loss is exactly the normal LM loss.
# ============================================================

class GatedTrainer(
    Trainer
):

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):

        outputs = model(
            **inputs
        )

        loss = outputs.loss

        if return_outputs:
            return (
                loss,
                outputs,
            )

        return loss


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train_gated_lora(
    args,
):

    set_seed(
        args.seed
    )

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "LOAD PROCESSOR"
    )

    print(
        "=" * 70
    )

    processor = (
        AutoProcessor
        .from_pretrained(
            args.model_name
        )
    )

    if (
        processor
        .tokenizer
        .pad_token_id
        is None
    ):

        processor.tokenizer.pad_token = (
            processor
            .tokenizer
            .eos_token
        )

    # --------------------------------------------------------
    # 4-bit QLoRA base
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "LOAD QWEN3-VL 4-BIT"
    )

    print(
        "=" * 70
    )

    quant_config = (
        BitsAndBytesConfig(

            load_in_4bit=True,

            bnb_4bit_quant_type="nf4",

            bnb_4bit_use_double_quant=True,

            bnb_4bit_compute_dtype=(
                torch.bfloat16
            ),

        )
    )

    model = (
        Qwen3VLForConditionalGeneration
        .from_pretrained(

            args.model_name,

            quantization_config=(
                quant_config
            ),

            dtype=torch.bfloat16,

            device_map="auto",

        )
    )

    # Official Qwen3-VL uses the same
    # Qwen3VLForConditionalGeneration +
    # AutoProcessor interface.
    #
    # prepare_model_for_kbit_training is
    # the standard PEFT preparation step
    # for 4-bit adapter training.

    model = (
        prepare_model_for_kbit_training(

            model,

            use_gradient_checkpointing=True,

        )
    )

    model.config.use_cache = (
        False
    )

    # --------------------------------------------------------
    # Insert gated LoRA
    # --------------------------------------------------------

    replaced_modules = (
        inject_gated_lora(

            model,

            rank=args.rank,

            alpha=args.alpha,

            dropout=args.dropout,

            gate_init=args.gate_init,

        )
    )

    # Make adapter weights BF16.

    for module in (
        model.modules()
    ):

        if isinstance(
            module,
            GatedLoRALinear,
        ):

            module.lora_A = (
                module.lora_A.to(
                    dtype=torch.bfloat16
                )
            )

            module.lora_B = (
                module.lora_B.to(
                    dtype=torch.bfloat16
                )
            )

    configure_trainable_parameters(
        model
    )

    print_trainable_parameters(
        model
    )

    print_gate_summary(
        model
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = (
        ManifestDataset(
            args.train_manifest
        )
    )

    eval_dataset = (
        ManifestDataset(
            args.eval_manifest
        )
    )

    collator = (
        Qwen3VLCollator(

            processor=processor,

            max_length=(
                args.max_length
            ),

        )
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    training_args = (
        TrainingArguments(

            output_dir=(
                args.output_dir
            ),

            num_train_epochs=(
                args.epochs
            ),

            max_steps=(
                args.max_steps
            ),

            per_device_train_batch_size=(
                args.batch_size
            ),

            per_device_eval_batch_size=1,

            gradient_accumulation_steps=(
                args.grad_accum
            ),

            learning_rate=(
                args.learning_rate
            ),

            lr_scheduler_type="cosine",

            warmup_ratio=0.03,

            weight_decay=0.01,

            optim="paged_adamw_8bit",

            bf16=True,

            fp16=False,

            gradient_checkpointing=True,

            logging_steps=(
                args.logging_steps
            ),

            eval_strategy="steps",

            eval_steps=(
                args.eval_steps
            ),

            save_strategy="steps",

            save_steps=(
                args.save_steps
            ),

            save_total_limit=2,

            remove_unused_columns=False,

            dataloader_num_workers=2,

            dataloader_pin_memory=True,

            report_to="none",

        )
    )

    trainer = (
        GatedTrainer(

            model=model,

            args=training_args,

            train_dataset=(
                train_dataset
            ),

            eval_dataset=(
                eval_dataset
            ),

            data_collator=(
                collator
            ),

        )
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "START TRAINING"
    )

    print(
        "=" * 70
    )

    train_result = (
        trainer.train()
    )

    # --------------------------------------------------------
    # Trainer metrics
    # --------------------------------------------------------

    trainer.log_metrics(
        "train",
        train_result.metrics,
    )

    trainer.save_metrics(
        "train",
        train_result.metrics,
    )

    # --------------------------------------------------------
    # Final loss-based eval
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL EVAL LOSS"
    )

    print(
        "=" * 70
    )

    eval_metrics = (
        trainer.evaluate()
    )

    trainer.log_metrics(
        "eval",
        eval_metrics,
    )

    trainer.save_metrics(
        "eval",
        eval_metrics,
    )

    # --------------------------------------------------------
    # Gate results
    # --------------------------------------------------------

    print_gate_summary(
        model
    )

    # --------------------------------------------------------
    # Save adapter
    # --------------------------------------------------------

    final_dir = (
        Path(
            args.output_dir
        )
        / "final_adapter"
    )

    config_dict = {

        "base_model":
            args.model_name,

        "method":
            "gated_qlora",

        "rank":
            args.rank,

        "alpha":
            args.alpha,

        "dropout":
            args.dropout,

        "gate_init":
            args.gate_init,

        "target_modules":
            sorted(
                TARGET_LEAF_NAMES
            ),

        "vision_frozen":
            True,

        "language_only":
            True,

        "system_prompt":
            SYSTEM_PROMPT,

        "replaced_modules":
            replaced_modules,

    }

    save_gated_adapter(

        model=model,

        processor=processor,

        output_dir=final_dir,

        config_dict=config_dict,

    )

    return (
        model,
        processor,
        trainer,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    args = parse_args()

    if args.batch_size != 1:

        raise ValueError(
            "--batch_size must currently be 1. "
            "Use --grad_accum to increase "
            "effective batch size."
        )

    train_gated_lora(
        args
    )