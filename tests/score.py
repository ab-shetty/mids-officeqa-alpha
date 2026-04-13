"""Local copy of the green judge's scoring logic.

Ported verbatim from officeqa-agentbeats/judge/src/agent.py so our local
sample-harness accuracy matches what the real evaluation would produce.
Only the imports and dataclasses are trimmed — the `score_answer` surface
and all helpers are byte-identical.
"""
from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    if not text:
        raise ValueError("Cannot normalize empty or None text")
    normalized = text.replace('\u2212', '-').replace('−', '-')
    return normalized


def extract_numbers_with_context(text: str) -> list[tuple[float, str, bool, bool]]:
    if not text:
        raise ValueError("Cannot extract numbers from empty text")
    text = normalize_text(text)
    text_no_commas = text.replace(',', '')
    numbers_with_context = []
    pattern = r'-?\d+\.?\d*%?'
    for match in re.finditer(pattern, text_no_commas):
        matched_text = match.group()
        if not matched_text or matched_text == '-':
            continue
        has_percent = matched_text.endswith('%')
        num_text = matched_text.rstrip('%')
        is_negative = num_text.startswith('-')
        try:
            num = float(num_text)
        except ValueError:
            continue
        start = max(0, match.start() - 20)
        end = min(len(text_no_commas), match.end() + 20)
        context = text_no_commas[start:end].lower()
        numbers_with_context.append((num, context, has_percent, is_negative))
    return numbers_with_context


def detect_unit_in_context(context: str) -> tuple[str | None, float]:
    c = context.lower()
    if re.search(r'\btrillions?\b', c): return ('trillion', 1e12)
    if re.search(r'\bbillions?\b', c) or re.search(r'\bb\b', c): return ('billion', 1e9)
    if re.search(r'\bmillions?\b', c) or re.search(r'\bm\b', c): return ('million', 1e6)
    if re.search(r'\bthousands?\b', c) or re.search(r'\bk\b', c): return ('thousand', 1e3)
    return (None, 1.0)


def normalize_number_with_units(number: float, context: str) -> tuple[float, str | None]:
    unit_name, _ = detect_unit_in_context(context)
    return (number, unit_name)


def is_likely_year(num: float) -> bool:
    return 1900 <= num <= 2100 and num == int(num)


def has_significant_text(text: str) -> tuple[bool, str]:
    if not text: return False, ""
    cleaned = normalize_text(text).lower()
    cleaned = re.sub(r'-?\d+\.?\d*%?', '', cleaned)
    cleaned = re.sub(r'[,]', '', cleaned)
    for unit in ['trillion','trillions','billion','billions','million','millions','thousand','thousands','hundred','hundreds','percent','percentage','%']:
        cleaned = re.sub(r'\b' + unit + r'\b', '', cleaned)
    cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return (len(cleaned) >= 2, cleaned)


def check_text_overlap(gt_text: str, pred_text: str) -> tuple[bool, str]:
    if not gt_text or not pred_text: return False, "empty"
    gt_has, gt_clean = has_significant_text(gt_text)
    pred_has, pred_clean = has_significant_text(pred_text)
    if not gt_has: return True, "gt purely numeric"
    if not pred_has: return False, "pred purely numeric, gt has text"
    if gt_clean in pred_clean: return True, "gt in pred"
    if pred_clean in gt_clean: return True, "pred in gt"
    return False, "text mismatch"


def extract_final_answer(text: str) -> str:
    m = re.search(r'<FINAL_ANSWER>\s*(.*?)\s*</FINAL_ANSWER>', text, re.DOTALL | re.IGNORECASE)
    if not m: raise ValueError("No FINAL_ANSWER tags found")
    content = m.group(1).strip()
    if not content: raise ValueError("FINAL_ANSWER tags are empty")
    if len(content) > 500: raise ValueError(f"FINAL_ANSWER too long ({len(content)})")
    return content


def contains_multiple_candidates(gt: str, pred: str) -> tuple[bool, str]:
    try:
        gt_n = extract_numbers_with_context(gt)
        pred_n = extract_numbers_with_context(pred)
    except ValueError:
        return False, ""
    if len(gt_n) != 1: return False, ""
    gt_val, _, _, _ = gt_n[0]
    gt_is_year = is_likely_year(gt_val)
    cands = set()
    for pv, _, _, _ in pred_n:
        if gt_is_year:
            if is_likely_year(pv): cands.add(int(pv))
        else:
            if not is_likely_year(pv): cands.add(round(pv, 2))
    if len(cands) > 1:
        return True, f"Hedged: {len(cands)} candidates"
    return False, ""


def fuzzy_match_answer(gt: str, pred: str, tolerance: float = 0.0) -> tuple[bool, str]:
    if not gt or not pred: return False, "empty"
    hedged, hr = contains_multiple_candidates(gt, pred)
    if hedged: return False, hr
    gt_nums = [(n, c) for n, c, _, _ in extract_numbers_with_context(gt)]
    pred_nums = [(n, c) for n, c, _, _ in extract_numbers_with_context(pred)]

    if gt_nums and pred_nums:
        if len(gt_nums) > 1:
            pred_non_years = [(n, c) for n, c in pred_nums
                              if not is_likely_year(n) or any(is_likely_year(g) for g, _ in gt_nums)]
            matched = 0
            for gv, gc in gt_nums:
                gb, _ = normalize_number_with_units(gv, gc)
                for pv, pc in pred_non_years:
                    pb, _ = normalize_number_with_units(pv, pc)
                    if gb == 0:
                        if pb == 0 and check_text_overlap(gt, pred)[0]:
                            matched += 1; break
                    else:
                        if abs(gb - pb) / abs(gb) <= tolerance and check_text_overlap(gt, pred)[0]:
                            matched += 1; break
            return (matched == len(gt_nums)), f"{matched}/{len(gt_nums)}"

        gv, gc = gt_nums[0]
        gb, gu = normalize_number_with_units(gv, gc)
        gt_has_text, _ = has_significant_text(gt)
        filter_years = not (is_likely_year(gv) or gt_has_text)
        best_diff = float('inf'); best_pb = None
        for pv, pc in pred_nums:
            if filter_years and is_likely_year(pv): continue
            pb, _ = normalize_number_with_units(pv, pc)
            if gb == 0:
                if pb == 0 and check_text_overlap(gt, pred)[0]:
                    return True, "zero match"
                continue
            diff = abs(gb - pb) / abs(gb)
            if diff < best_diff:
                best_diff = diff; best_pb = pb
            if diff <= tolerance and check_text_overlap(gt, pred)[0]:
                return True, f"match gb={gb} pb={pb}"
        if best_pb is not None:
            return False, f"no-match gb={gb} closest={best_pb} diff={best_diff*100:.2f}%"
        return False, "no valid numbers"

    g = gt.strip().lower().strip('"').strip("'")
    p = pred.strip().lower().strip('"').strip("'")
    g = re.sub(r'\([^)]*\)', '', g).strip()
    p = re.sub(r'\([^)]*\)', '', p).strip()
    if g in p: return True, "text in pred"
    if g == p: return True, "text exact"
    return False, "no match"


def score_answer(ground_truth: str, predicted: str, tolerance: float = 0.0) -> tuple[bool, str]:
    try:
        ans = extract_final_answer(predicted)
    except ValueError as e:
        return False, str(e)
    if ans.strip().lower() == "no answer found":
        return False, "no answer found"
    return fuzzy_match_answer(ground_truth, ans, tolerance)
