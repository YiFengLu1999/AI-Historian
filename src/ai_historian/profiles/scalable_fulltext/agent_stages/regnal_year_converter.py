"""Deterministic Chinese regnal-year to ISO-date conversion for A9.

This module is intentionally narrow: it handles explicit regnal expressions
such as "洪武三年", "永乐二十二年八月", and "建文元年春". Context-dependent
expressions such as "明年", "是岁", "太祖时", or "洪武间" remain outside this
converter and should be handled by upstream context propagation or the LLM
fallback in Step11.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

ERA_REQUIRED_COLUMNS = ("皇帝名称", "年号/纪元", "开始的第一天是ISO 8601的那一天")

CN_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}

MONTH_ALIASES = {
    "正": 1,
    "元": 1,
    "冬": 11,
    "腊": 12,
    "臘": 12,
}

SEASON_MONTHS = {
    "春": 3,
    "夏": 6,
    "秋": 9,
    "冬": 12,
}


@dataclass(frozen=True)
class EraRecord:
    emperor: str
    era: str
    start_iso: str
    start: Tuple[int, int, int]
    dynasty_key: str
    next_start: Optional[Tuple[int, int, int]] = None


@dataclass(frozen=True)
class RegnalConversion:
    iso: str
    source: str
    era: str
    emperor: str
    regnal_year: int


def parse_iso_date(value: str) -> Optional[Tuple[int, int, int]]:
    match = re.match(r"^([+-]?\d{4,})-(\d{2})-(\d{2})$", str(value or "").strip())
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def format_iso_date(year: int, month: int, day: int = 1) -> str:
    if year < 0:
        year_text = f"-{abs(year):04d}"
    else:
        year_text = f"{year:04d}"
    return f"{year_text}-{month:02d}-{day:02d}"


def parse_chinese_integer(text: str) -> Optional[int]:
    value = str(text or "").strip()
    if not value:
        return None
    if value == "元":
        return 1
    if value.isdigit():
        number = int(value)
        return number if number > 0 else None
    if value.startswith("廿"):
        rest = value[1:]
        return 20 + (CN_DIGITS.get(rest, 0) if rest else 0)
    if value.startswith("卅"):
        rest = value[1:]
        return 30 + (CN_DIGITS.get(rest, 0) if rest else 0)
    if value in CN_DIGITS:
        number = CN_DIGITS[value]
        return number if number > 0 else None
    if "十" in value:
        left, _, right = value.partition("十")
        tens = CN_DIGITS.get(left, 1) if left else 1
        ones = CN_DIGITS.get(right, 0) if right else 0
        number = tens * 10 + ones
        return number if number > 0 else None
    return None


def parse_month(suffix: str) -> Optional[int]:
    text = str(suffix or "")
    match = re.search(r"(?:闰|閏)?(?P<month>正|元|冬|腊|臘|\d{1,2}|[一二三四五六七八九十廿卅两]{1,3})月", text)
    if not match:
        return None
    month_text = match.group("month")
    if month_text in MONTH_ALIASES:
        return MONTH_ALIASES[month_text]
    month = parse_chinese_integer(month_text)
    if month is None or not (1 <= month <= 12):
        return None
    return month


def parse_season_month(suffix: str) -> Optional[int]:
    text = str(suffix or "")
    for season, month in SEASON_MONTHS.items():
        if season in text:
            return month
    return None


def read_era_rows(path: Path) -> List[EraRecord]:
    rows: List[Tuple[str, str, str, Tuple[int, int, int]]] = []
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            with path.open(encoding=encoding, newline="") as fh:
                reader = csv.DictReader(fh)
                if not reader.fieldnames or not all(col in reader.fieldnames for col in ERA_REQUIRED_COLUMNS):
                    return []
                for row in reader:
                    start_iso = str(row.get("开始的第一天是ISO 8601的那一天", "")).strip()
                    start = parse_iso_date(start_iso)
                    emperor = str(row.get("皇帝名称", "")).strip()
                    era = str(row.get("年号/纪元", "")).strip()
                    if emperor and era and start:
                        rows.append((emperor, era, start_iso, start))
            break
        except UnicodeDecodeError:
            rows = []
            continue

    records: List[EraRecord] = []
    for index, (emperor, era, start_iso, start) in enumerate(rows):
        dynasty = dynasty_key(emperor)
        next_start = None
        if index + 1 < len(rows) and dynasty_key(rows[index + 1][0]) == dynasty:
            next_start = rows[index + 1][3]
        records.append(
            EraRecord(
                emperor=emperor,
                era=era,
                start_iso=start_iso,
                start=start,
                dynasty_key=dynasty,
                next_start=next_start,
            )
        )
    return records


def dynasty_key(emperor: str) -> str:
    text = str(emperor or "").strip()
    for prefix in ("东汉", "西汉", "北魏", "东魏", "西魏", "北周", "北齐", "南齐", "刘宋", "南宋"):
        if text.startswith(prefix):
            return prefix
    return text[:1]


class RegnalYearConverter:
    """Convert explicit Chinese regnal-year strings using a reference CSV."""

    def __init__(self, records: Iterable[EraRecord], source: str = ""):
        self.records = list(records)
        self.source = source
        self.by_era: Dict[str, List[EraRecord]] = {}
        for record in self.records:
            self.by_era.setdefault(record.era, []).append(record)
        self.era_names = sorted(self.by_era, key=len, reverse=True)

    @classmethod
    def from_csv(cls, path: Path) -> "RegnalYearConverter":
        return cls(read_era_rows(path), source=str(path))

    def convert(self, tm: str) -> Optional[RegnalConversion]:
        text = str(tm or "").strip()
        if not text:
            return None

        for era in self.era_names:
            start_index = text.find(era)
            if start_index < 0:
                continue
            after_era = text[start_index + len(era):]
            match = re.search(r"(?P<year>元|\d{1,3}|[〇零一二两三四五六七八九十廿卅]{1,5})年", after_era)
            if not match:
                continue
            regnal_year = parse_chinese_integer(match.group("year"))
            if not regnal_year:
                continue
            suffix = after_era[match.end():]
            record = self._select_record(era, text)
            if record is None:
                return None
            month = parse_month(suffix)
            if month is None:
                month = parse_season_month(suffix)
            if month is None:
                month = 10
            year = record.start[0] + regnal_year - 1
            candidate = (year, month, 1)
            if self._outside_era_bounds(candidate, record):
                return None
            return RegnalConversion(
                iso=format_iso_date(year, month, 1),
                source=self.source,
                era=record.era,
                emperor=record.emperor,
                regnal_year=regnal_year,
            )
        return None

    def _select_record(self, era: str, text: str) -> Optional[EraRecord]:
        candidates = self.by_era.get(era, [])
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        mentioned = [record for record in candidates if record.emperor and record.emperor in text]
        if len(mentioned) == 1:
            return mentioned[0]
        return None

    @staticmethod
    def _outside_era_bounds(candidate: Tuple[int, int, int], record: EraRecord) -> bool:
        if candidate < (record.start[0], 1, 1):
            return True
        if record.next_start and candidate >= record.next_start:
            return True
        return False


def load_default_converter(project_root: Path) -> RegnalYearConverter:
    from ai_historian.resources import CHINESE_ERAS

    return RegnalYearConverter.from_csv(CHINESE_ERAS)
