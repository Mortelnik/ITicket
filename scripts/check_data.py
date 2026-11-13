"""Проверка датасета и сохранённого разбиения без повторного обучения.

Запуск из корня: uv run python scripts/check_data.py
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

from ticket_router.data import DataValidationError, combine_text, load_dataset, sha256_file, validate


def check_data(data_path: Path, manifest_path: Path, splits_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = sha256_file(data_path)
    if digest != manifest["sha256"]:
        raise DataValidationError("SHA256 датасета не совпадает с manifest.json")
    df = load_dataset(data_path)
    if len(df) != manifest["rows"]:
        raise DataValidationError("Количество строк не совпадает с manifest.json")
    recorded = json.loads(splits_path.read_text(encoding="utf-8"))
    indices = {name: recorded[f"{name}_idx"] for name in ("train", "val", "test")}
    flat = [idx for group in indices.values() for idx in group]
    if len(flat) != len(df) or len(set(flat)) != len(flat) or set(flat) != set(df.index):
        raise DataValidationError("Индексы split должны покрывать датасет ровно один раз")
    if any(not group for group in indices.values()):
        raise DataValidationError("Выборки train/val/test не должны быть пустыми")
    texts = combine_text(df)
    normalized = texts.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    overlaps = {}
    problems = validate(df)
    for left, right in combinations(indices, 2):
        exact_count = len(set(texts.loc[indices[left]]) & set(texts.loc[indices[right]]))
        normalized_count = len(set(normalized.loc[indices[left]]) & set(normalized.loc[indices[right]]))
        overlaps[f"{left}_{right}"] = {"exact": exact_count, "normalized": normalized_count}
        if exact_count:
            raise DataValidationError(f"Тексты повторяются между {left} и {right}: {exact_count}")
        if normalized_count:
            problems.append(f"{left}/{right}: совпадающих текстов после lower/whitespace: {normalized_count}")
    return {
        "hard_checks_passed": True,
        "data_sha256": digest,
        "rows": len(df),
        "missing_subject": int(df["subject"].isna().sum()),
        "missing_body": int(df["body"].isna().sum()),
        "languages": df["language"].value_counts().to_dict(),
        "queues": df["queue"].value_counts().to_dict(),
        "split_sizes": {name: len(group) for name, group in indices.items()},
        "text_overlaps": overlaps,
        "warnings": problems,
        "note": "Нормализованные совпадения — остаточный риск; near-duplicates этим аудитом не исключаются.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/aa_dataset-tickets-multi-lang-5-2-50-version.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.json"))
    parser.add_argument("--splits", type=Path, default=Path("experiments/splits.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check_data(args.data, args.manifest, args.splits)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
