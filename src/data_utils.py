# ============================================================
# data_utils.py
# Dataset + collator for Qwen3-VL SFT
# ============================================================

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image

DATASETS = (
    "chartqa",
    "textvqa",
    "gqa",
    "docvqa",
    "visual_genome",
)

# Keep vision tokens inside max_length=2048. Qwen3-VL default longest_edge
# is huge, which produced 2100+ image tokens and a token/feature mismatch.
MAX_IMAGE_SIDE = 1024


def answer_to_text(answer):
    """
    Normalize different VQA answer formats into one target string.
    """
    if answer is None:
        return ""

    if isinstance(answer, str):
        return answer

    if isinstance(answer, (int, float)):
        return str(answer)

    if isinstance(answer, list):
        if len(answer) == 0:
            return ""

        # VQA may contain list[str]
        if all(isinstance(x, str) for x in answer):
            # use first answer for SFT
            return answer[0]

        # Sometimes list[dict]
        if isinstance(answer[0], dict):
            for key in ["answer", "text", "label"]:
                if key in answer[0]:
                    return str(answer[0][key])

        return str(answer[0])

    if isinstance(answer, dict):
        for key in ["answer", "text", "label"]:
            if key in answer:
                return str(answer[key])

    return str(answer)


def question_from_record(record):
    """Read question from the top-level field, then metadata fallbacks."""
    question = record.get("question")
    if question is not None:
        return str(question).strip()

    meta = record.get("metadata") or {}
    for key in ["question", "query", "prompt", "instruction"]:
        if meta.get(key) is not None:
            return str(meta[key]).strip()

    return ""


def answer_from_record(record):
    """Read answer from the top-level field, then metadata fallbacks."""
    answer = record.get("answer")
    if answer is not None:
        return answer

    meta = record.get("metadata") or {}
    for key in ["answer", "answers", "label", "labels", "target"]:
        if meta.get(key) is not None:
            return meta[key]

    return None


def image_path_candidates(image_path, data_root=None):
    """
    Ordered paths to try. Remapped data/ locations come first.

    Manifests store Colab paths like:
      /content/drive/MyDrive/gated_lora/gqa/images/...

    Files actually live under:
      <repo>/data/gqa/images/...
    """
    if image_path is None:
        return []

    if isinstance(image_path, list):
        image_path = image_path[0]

    original = Path(str(image_path))
    posix = str(image_path).replace("\\", "/")
    candidates = []

    if data_root is not None:
        data_root = Path(data_root)
        for dataset in DATASETS:
            token = f"/{dataset}/"
            padded = f"/{posix}" if not posix.startswith("/") else posix
            if token in padded:
                rel = padded.split(token, 1)[1]
                candidates.append(data_root / dataset / rel)
                break

    if "gated_lora/" in posix and "gated_lora/data/" not in posix:
        idx = posix.find("gated_lora/") + len("gated_lora/")
        candidates.append(Path(posix[:idx] + "data/" + posix[idx:]))

    candidates.append(original)

    unique = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def resolve_image_path(image_path, data_root=None):
    """Return the first existing candidate, else the remapped data/ path."""
    candidates = image_path_candidates(image_path, data_root)
    if not candidates:
        return None
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def open_rgb_image(image_path, data_root=None):
    last_error = None
    candidates = image_path_candidates(image_path, data_root)
    if not candidates:
        raise FileNotFoundError(f"No image path in record: {image_path}")
    for candidate in candidates:
        try:
            with Image.open(candidate) as image:
                image.load()
                rgb = image.convert("RGB")
            rgb.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
            return rgb, candidate
        except (FileNotFoundError, OSError, Image.UnidentifiedImageError) as exc:
            last_error = exc
    raise FileNotFoundError(
        f"Image not readable. Tried: {[str(c) for c in candidates]}"
    ) from last_error


class ManifestDataset(Dataset):

    def __init__(self, manifest_path, data_root=None):
        self.manifest_path = Path(manifest_path)
        self.data_root = Path(data_root) if data_root else None
        self.skipped_images = 0

        self.records = []

        with self.manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                r = json.loads(line)

                question = question_from_record(r)
                answer = answer_from_record(r)
                image = r.get("image")

                if not question or answer is None or image is None:
                    continue

                self.records.append(r)

        print(
            f"Loaded {len(self.records):,} usable examples "
            f"from {self.manifest_path}"
        )

    def __len__(self):
        return len(self.records)

    def _example_from_record(self, r):
        image, _resolved = open_rgb_image(r["image"], self.data_root)
        question = question_from_record(r)
        answer = answer_to_text(answer_from_record(r)).strip()
        return {
            "id": r.get("id"),
            "dataset": r.get("dataset"),
            "image": image,
            "question": question,
            "answer": answer,
        }

    def __getitem__(self, idx):
        n = len(self.records)
        if n == 0:
            raise IndexError("ManifestDataset is empty")
        last_error = None
        for offset in range(n):
            r = self.records[(idx + offset) % n]
            try:
                return self._example_from_record(r)
            except (FileNotFoundError, OSError, Image.UnidentifiedImageError) as exc:
                last_error = exc
                self.skipped_images += 1
                print(
                    f"Skipping missing image ({self.skipped_images}): "
                    f"{r.get('id')} {r.get('image')}"
                )
        raise FileNotFoundError(
            "Every training image was unreadable"
        ) from last_error


class Qwen3VLCollator:
    """
    Builds multimodal Qwen3-VL SFT batches.

    Loss is applied only to assistant answer tokens.
    """

    def __init__(
        self,
        processor,
        max_length=2048,
    ):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, examples):
        batch_inputs = []
        examples = [x for x in examples if x is not None]

        for x in examples:
            # Complete training conversation
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": x["image"],
                        },
                        {
                            "type": "text",
                            "text": x["question"],
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": x["answer"],
                        }
                    ],
                },
            ]

            encoded = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
            )

            # Remove singleton batch dimension.
            # image_grid_thw must stay [num_images, 3], not [3].
            sample = {}

            for k, v in encoded.items():
                if torch.is_tensor(v):
                    sample[k] = self._squeeze_feature(k, v)
                else:
                    sample[k] = v

            # ------------------------------------------------
            # Construct labels
            #
            # Better than training on prompt/image tokens.
            # Find answer boundary from prompt-only encoding.
            # ------------------------------------------------

            prompt_messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": x["image"],
                        },
                        {
                            "type": "text",
                            "text": x["question"],
                        },
                    ],
                }
            ]

            prompt_encoded = self.processor.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )

            prompt_len = prompt_encoded["input_ids"].shape[1]
            seq_len = sample["input_ids"].shape[0]

            # Never cut image tokens. That leaves extra pixel_values and
            # crashes with "Image features and image tokens do not match".
            if prompt_len > self.max_length:
                print(
                    f"Skipping example {x.get('id')}: "
                    f"prompt+image is {prompt_len} tokens > max_length={self.max_length}"
                )
                continue

            if seq_len > self.max_length:
                for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
                    if key in sample and torch.is_tensor(sample[key]):
                        sample[key] = sample[key][:self.max_length]

            labels = sample["input_ids"].clone()
            labels[:prompt_len] = -100
            sample["labels"] = labels
            batch_inputs.append(sample)

        if not batch_inputs:
            raise RuntimeError(
                "Every example in this batch exceeded max_length. "
                "Lower image size or raise --max_length."
            )

        return self._pad(batch_inputs)

    @staticmethod
    def _squeeze_feature(key, value):
        if value.ndim >= 2 and value.shape[0] == 1:
            value = value.squeeze(0)
        if key == "image_grid_thw" and value.ndim == 1:
            value = value.unsqueeze(0)
        return value

    @staticmethod
    def _stack_vision(key, vals):
        if key == "image_grid_thw":
            vals = [
                v.unsqueeze(0) if v.ndim == 1 else v
                for v in vals
            ]
        try:
            return torch.cat(vals, dim=0)
        except Exception:
            return torch.stack(vals)

    def _pad(self, examples):
        max_len = max(x["input_ids"].shape[0] for x in examples)
        max_len = min(max_len, self.max_length)

        pad_id = self.processor.tokenizer.pad_token_id

        input_ids = []
        attention_masks = []
        labels = []
        token_types = []

        for x in examples:
            ids = x["input_ids"][:max_len]
            mask = x["attention_mask"][:max_len]
            lab = x["labels"][:max_len]
            pad_len = max_len - len(ids)

            ids = torch.cat([
                ids,
                torch.full((pad_len,), pad_id, dtype=ids.dtype),
            ])
            mask = torch.cat([
                mask,
                torch.zeros(pad_len, dtype=mask.dtype),
            ])
            lab = torch.cat([
                lab,
                torch.full((pad_len,), -100, dtype=lab.dtype),
            ])

            input_ids.append(ids)
            attention_masks.append(mask)
            labels.append(lab)

            if x.get("mm_token_type_ids") is not None:
                tt = x["mm_token_type_ids"]
                if tt.ndim == 0:
                    tt = tt.unsqueeze(0)
                tt = tt[:max_len]
                tt = torch.cat([
                    tt,
                    torch.zeros(max_len - len(tt), dtype=tt.dtype),
                ])
                token_types.append(tt)

        batch = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(labels),
        }
        if token_types:
            batch["mm_token_type_ids"] = torch.stack(token_types)

        # Vision fields are concatenated across images, not batched like text.
        for key in ("pixel_values", "image_grid_thw"):
            vals = [
                x[key]
                for x in examples
                if key in x and x[key] is not None
            ]
            if vals:
                batch[key] = self._stack_vision(key, vals)

        return batch
