import os
import re
import threading
from typing import Optional
from functools import partial

from verl.utils.reward_score.prime_math import compute_score
from math_verify import parse, verify

def extract_last_boxed(text: str) -> tuple[Optional[str], str]:
    pattern = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    matches = list(re.finditer(pattern, text))
    if matches:
        return matches[-1].group(1), ""
    return None, "missing \\boxed{}"


def extract_last_final_answer(text: str) -> tuple[Optional[str], str]:
    candidate_patterns = [
        r"Final Answer:\s*((?:[^<]|<[^<])*?)\n",
        r"Final Answer is:\s*((?:[^<]|<[^<])*?)\n",
        r"The answer is:\s*((?:[^<]|<[^<])*?)\n",
        r"Answer:\s*((?:[^<]|<[^<])*?)\n",
        r"Solution:\s*((?:[^<]|<[^<])*?)\n",
        r"The solution is:\s*((?:[^<]|<[^<])*?)\n",
    ]

    last_match: Optional[str] = None
    last_position = -1
    for pattern in candidate_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            if match.start() > last_position:
                last_position = match.start()
                last_match = match.group(1).strip()

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
    for stop_word in stop_words:
        if last_match and last_match.endswith(stop_word):
            last_match = last_match[: -len(stop_word)].strip()

    if last_match:
        return last_match, ""
    return None, "missing final answer markers"


def extract_solution(solution_str: str) -> tuple[Optional[str], str]:
    answer, reason = extract_last_boxed(solution_str)
    if answer:
        return answer, ""
    answer, reason = extract_last_final_answer(solution_str)
    if answer:
        return answer, ""
    return None, reason or f"no recognizable answer pattern: {solution_str[-100:]}"

def safe_math_verify(
    predict: str,
    ground_truth: str,
    parsing_timeout: int = 0.5,
    verify_timeout: int = 0.5,
    parse_kwargs: dict = {},
    verify_kwargs: dict = {},
):
    predict_strings = [f"${predict}$", predict]

    for predict_str in predict_strings:
        parsed_predict = parse(predict_str, parsing_timeout=parsing_timeout, **parse_kwargs)
        if parsed_predict is None:
            continue

        gold_strings = [f"${ground_truth}$", ground_truth]

        for gold_str in gold_strings:
            parsed_gold = parse(gold_str, parsing_timeout=parsing_timeout, **parse_kwargs)
            if parsed_gold is None:
                continue

            ok = verify(parsed_predict, parsed_gold, timeout_seconds=verify_timeout, **verify_kwargs)
            if ok:
                return True

    return False


def exact_match_verify(predict: str, ground_truth: str):
    try:
        res = re.findall(r"\\boxed{(.*)}", predict)
        res = res[-1] 
    except Exception as e:
        res = predict
    return res.strip() == ground_truth
    

def math_verify_score(
    answer,
    ground_truth,
    **kwargs
) -> str:
    answer = str(answer).strip()
    ground_truth = str(ground_truth).strip()
    if abs(len(answer) - len(ground_truth)) > 50:
        return False
    verify_score = exact_match_verify(answer, ground_truth)
    if verify_score != 1:
        verify_score = safe_math_verify(answer, ground_truth, parsing_timeout=0.5, verify_timeout=1.0)
    if verify_score != 1:
        verify_score, format_score, extracted_answer = compute_score(answer, ground_truth, timeout=1.0)
    return verify_score

if __name__ == "__main__":
    # Example usage
    pred = "\\left( 3, \\frac{\\pi}{2} \\right)"
    gt = "(3, \\frac{\\pi}{2})"
    score = math_verify_score(pred, gt)
    print(f"Score: {score}")
    print(pred)
    print(gt)
