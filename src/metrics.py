# ============================================================
# QWEN3-VL EVALUATION METRICS
# Dataset-specific metrics:
#   GQA            -> normalized exact match
#   TextVQA        -> VQA consensus accuracy
#   ChartQA        -> relaxed accuracy
#   DocVQA         -> ANLS
#   Visual Genome  -> normalized exact match
# ============================================================

import re
import string


METRIC_NAMES = {
    "gqa": "Accuracy",
    "textvqa": "VQA Accuracy",
    "chartqa": "Relaxed Accuracy",
    "docvqa": "ANLS",
    "visual_genome": "Accuracy",
}


def basic_normalize(x):
    if x is None:
        return ""

    x = str(x).strip().lower()

    # collapse whitespace
    x = re.sub(r"\s+", " ", x)

    # remove punctuation
    x = x.translate(
        str.maketrans("", "", string.punctuation)
    )

    return x.strip()


def extract_answers(gt):
    if gt is None:
        return []

    if isinstance(gt, str):
        return [gt]

    if isinstance(gt, (int, float)):
        return [str(gt)]

    if isinstance(gt, dict):
        for key in [
            "answer",
            "answers",
            "text",
            "label",
            "labels",
        ]:
            if key in gt:
                return extract_answers(gt[key])

        return [str(gt)]

    if isinstance(gt, list):
        answers = []

        for item in gt:
            if isinstance(item, dict):
                for key in [
                    "answer",
                    "text",
                    "label",
                ]:
                    if key in item:
                        answers.append(
                            str(item[key])
                        )
                        break
            else:
                answers.append(str(item))

        return answers

    return [str(gt)]


def gqa_score(prediction, gt):
    pred = basic_normalize(prediction)

    answers = extract_answers(gt)

    answers = [
        basic_normalize(x)
        for x in answers
    ]

    return float(pred in answers)


def visual_genome_score(prediction, gt):
    pred = basic_normalize(prediction)

    answers = [
        basic_normalize(x)
        for x in extract_answers(gt)
    ]

    return float(pred in answers)


def textvqa_score(prediction, gt):
    """
    VQA-style consensus accuracy.

    Standard VQA formulation:
      accuracy = min(#matching human answers / 3, 1)
    """
    pred = basic_normalize(prediction)

    answers = [
        basic_normalize(x)
        for x in extract_answers(gt)
    ]

    if len(answers) == 0:
        return 0.0

    matches = sum(
        pred == ans
        for ans in answers
    )

    return min(matches / 3.0, 1.0)


def parse_number(text):
    if text is None:
        return None

    s = str(text).strip().lower()

    # remove commas
    s = s.replace(",", "")

    # percentage
    percent = "%" in s
    s = s.replace("%", "")

    # strip common currency symbols
    s = re.sub(r"[$€£¥]", "", s)

    # find first number
    match = re.search(
        r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
        s
    )

    if not match:
        return None

    try:
        value = float(match.group())

        if percent:
            # Keep percentage scale as written.
            # e.g. 25% -> 25
            value = value

        return value

    except ValueError:
        return None


def chartqa_single_score(prediction, answer):
    pred_num = parse_number(prediction)
    gt_num = parse_number(answer)

    # numeric relaxed matching
    if pred_num is not None and gt_num is not None:
        if gt_num == 0:
            return float(
                abs(pred_num - gt_num) <= 1e-6
            )

        relative_error = abs(
            pred_num - gt_num
        ) / abs(gt_num)

        return float(relative_error <= 0.05)

    # string matching fallback
    return float(
        basic_normalize(prediction)
        ==
        basic_normalize(answer)
    )


def chartqa_score(prediction, gt):
    answers = extract_answers(gt)

    if len(answers) == 0:
        return 0.0

    return max(
        chartqa_single_score(
            prediction,
            answer
        )
        for answer in answers
    )


def levenshtein_distance(a, b):
    if a == b:
        return 0

    if len(a) == 0:
        return len(b)

    if len(b) == 0:
        return len(a)

    previous = list(range(len(b) + 1))

    for i, ca in enumerate(a, start=1):
        current = [i]

        for j, cb in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            substitute_cost = (
                previous[j - 1]
                + (ca != cb)
            )

            current.append(
                min(
                    insert_cost,
                    delete_cost,
                    substitute_cost,
                )
            )

        previous = current

    return previous[-1]


def docvqa_single_anls(prediction, answer):
    pred = str(prediction).strip().lower()
    gt = str(answer).strip().lower()

    if pred == gt:
        return 1.0

    if len(pred) == 0 or len(gt) == 0:
        return 0.0

    dist = levenshtein_distance(
        pred,
        gt
    )

    similarity = 1.0 - (
        dist /
        max(len(pred), len(gt))
    )

    if similarity < 0.5:
        return 0.0

    return similarity


def docvqa_score(prediction, gt):
    answers = extract_answers(gt)

    if len(answers) == 0:
        return 0.0

    return max(
        docvqa_single_anls(
            prediction,
            answer
        )
        for answer in answers
    )


def score_prediction(dataset, prediction, gt):
    dataset = dataset.lower().strip()

    if dataset == "gqa":
        return gqa_score(prediction, gt)

    if dataset == "textvqa":
        return textvqa_score(prediction, gt)

    if dataset == "chartqa":
        return chartqa_score(prediction, gt)

    if dataset == "docvqa":
        return docvqa_score(prediction, gt)

    if dataset in [
        "visual_genome",
        "visualgenome",
        "vg",
    ]:
        return visual_genome_score(prediction, gt)

    raise ValueError(f"Unknown dataset: {dataset}")
