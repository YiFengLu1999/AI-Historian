import re
from typing import Any, Dict, List, Tuple

SEASON_MONTHS = {
    "春": "春",
    "春天": "春",
    "春季": "春",
    "夏": "夏",
    "夏天": "夏",
    "夏季": "夏",
    "秋": "秋",
    "秋天": "秋",
    "秋季": "秋",
    "冬": "冬",
    "冬天": "冬",
    "冬季": "冬",
}

PSEUDO_TIME_EXACT = {
    "在即墨",
    "夜",
    "夜间",
    "明天早晨",
    "过了几天",
    "等到秦军投降了诸侯军",
    "沛公在山东时",
    "秦朝的御史到泗水郡督察郡的工作时",
}

VAGUE_TIME_EXACT = {
    "秦末",
    "秦末汉初",
    "汉王时期",
    "秦王子婴时期",
    "秦朝末年，刘邦起事做沛公时",
}


def _strip_noise(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"\s+", "", text)
    text = text.strip("，,。；;：:")
    text = re.sub(r"间$", "", text)
    return text


def _latest_regnal_prefix(previous_tm: str) -> str:
    text = str(previous_tm or "")
    matches = re.findall(r"((?:汉高祖|汉|秦二世|秦始皇|秦王子婴)[元一二三四五六七八九十]+年)", text)
    return matches[-1] if matches else ""


def _latest_regnal_title(previous_tm: str) -> str:
    text = str(previous_tm or "")
    matches = re.findall(r"((?:汉高祖|汉|秦二世|秦始皇|秦王子婴))[元一二三四五六七八九十]+年", text)
    return matches[-1] if matches else ""


def normalize_experiment1_tm(tm: str, previous_tm: str = "") -> str:
    raw_text = str(tm or "").strip()
    text = _strip_noise(raw_text)
    if not text or text == "无":
        return ""

    if text in PSEUDO_TIME_EXACT or text in VAGUE_TIME_EXACT:
        return raw_text

    text = re.sub(r"公元前206年十二月（[^）]*）", "汉元年十二月", text)
    text = text.replace("公元前206年十一月", "汉元年十一月")
    text = text.replace("公元前206年十二月", "汉元年十二月")
    text = text.replace("公元前206年十月", "汉元年十月")
    text = text.replace("公元前206年八月", "汉高祖元年八月")
    text = text.replace("公元前206年四月", "汉高祖元年四月")
    text = text.replace("公元前206年正月", "汉高祖元年正月")
    text = re.sub(r"^汉王([元一二三四五六七八九十]+年)", r"汉高祖\1", text)
    text = re.sub(r"^汉元年", "汉高祖元年", text)
    text = re.sub(r"([元一二三四五六七八九十]+年)的?(春天|春季|春)", r"\1春", text)
    text = re.sub(r"([元一二三四五六七八九十]+年)的?(夏天|夏季|夏)", r"\1夏", text)
    text = re.sub(r"([元一二三四五六七八九十]+年)的?(秋天|秋季|秋)", r"\1秋", text)
    text = re.sub(r"([元一二三四五六七八九十]+年)的?(冬天|冬季|冬)", r"\1冬", text)
    text = re.sub(
        r"^((?:汉高祖|汉|秦二世|秦始皇|秦王子婴)[元一二三四五六七八九十]+年[正一二三四五六七八九十冬腊]+月)(?:早晨|中午|午间|傍晚|晚上|夜间|夜|早晨、中午|早晨和中午|早晨至中午).*$",
        r"\1",
        text,
    )

    bare_month = re.fullmatch(
        r"([正一二三四五六七八九十冬腊]+月)(?:[,，、和至]*(?:早晨|中午|午间|傍晚|晚上|夜间|夜))*",
        text,
    )
    if bare_month:
        prefix = _latest_regnal_prefix(previous_tm)
        return f"{prefix}{bare_month.group(1)}" if prefix else bare_month.group(1)

    def repl_short_bce(match: re.Match) -> str:
        month = match.group(1)
        return f"汉高祖元年{month}月" if month else "汉高祖元年"

    text = re.sub(r"(?<!公元)前206年([正一二三四五六七八九十冬腊]+)月", repl_short_bce, text)
    text = re.sub(r"(?<!公元)前206年(?![一-龥月])", "汉高祖元年", text)

    bare_year = re.fullmatch(r"([元一二三四五六七八九十]+年)", text)
    if bare_year:
        title = _latest_regnal_title(previous_tm)
        return f"{title}{bare_year.group(1)}" if title else raw_text

    if text in SEASON_MONTHS:
        prefix = _latest_regnal_prefix(previous_tm)
        return f"{prefix}{SEASON_MONTHS[text]}" if prefix else raw_text

    return text


def is_non_anchor_tm(tm: str) -> bool:
    text = _strip_noise(tm)
    return text in PSEUDO_TIME_EXACT or text in VAGUE_TIME_EXACT or text in SEASON_MONTHS


def is_context_dependent_tm(tm: str) -> bool:
    text = _strip_noise(tm)
    if text in SEASON_MONTHS:
        return True
    return bool(re.fullmatch(
        r"[正一二三四五六七八九十冬腊]+月(?:[,，、和至]*(?:早晨|中午|午间|傍晚|晚上|夜间|夜))*",
        text,
    ))


def canonicalize_timeblock_sequence(tmb: List[Dict[str, Any]]) -> Tuple[int, int]:
    changed = 0
    demoted = 0
    previous_anchor_tm = ""

    for obj in tmb:
        if not isinstance(obj, dict):
            continue

        original_tm = str(obj.get("TM", "") or "").strip()
        canonical_tm = normalize_experiment1_tm(original_tm, previous_anchor_tm)
        if canonical_tm != original_tm:
            obj["TM"] = canonical_tm
            changed += 1
            obj.pop("iso", None)
            obj.pop("iso_range", None)

        if original_tm and (not canonical_tm or is_non_anchor_tm(canonical_tm) or is_context_dependent_tm(canonical_tm)):
            if canonical_tm and canonical_tm != original_tm:
                obj["TM"] = canonical_tm
                changed += 1
            if str(obj.get("Granularity", "")).strip() != "0":
                obj["Granularity"] = "0"
                demoted += 1
            obj.pop("iso", None)
            obj.pop("iso_range", None)
            continue

        if canonical_tm and str(obj.get("Granularity", "")).strip() != "0":
            previous_anchor_tm = canonical_tm

    return changed, demoted
