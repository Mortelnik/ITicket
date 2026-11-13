"""Повторная проверка выбранного артефакта на сохранённом test, без обучения.

Не использовать этот test для выбора новой конфигурации или подбора порогов.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, f1_score

from check_data import check_data
from ticket_router.data import combine_text, load_dataset, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/aa_dataset-tickets-multi-lang-5-2-50-version.csv"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--metadata", type=Path, default=Path("artifacts/model_metadata.json"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.json"))
    parser.add_argument("--splits", type=Path, default=Path("experiments/splits.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    audit = check_data(args.data, args.manifest, args.splits)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    if metadata["data_sha256"] != audit["data_sha256"]:
        raise RuntimeError("Модель обучалась на другом датасете")
    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    test = load_dataset(args.data).loc[splits["test_idx"]]
    model = joblib.load(args.model)
    if list(model.classes_) != metadata["classes"]:
        raise RuntimeError("Классы модели не совпадают с метаданными")
    texts = combine_text(test)
    predicted = model.predict(texts)
    measured = {
        "test_macro_f1": round(f1_score(test["queue"], predicted, average="macro"), 4),
        "test_weighted_f1": round(f1_score(test["queue"], predicted, average="weighted"), 4),
        "test_accuracy": round(accuracy_score(test["queue"], predicted), 4),
    }
    for key, value in measured.items():
        if value != metadata[key]:
            raise RuntimeError(f"{key}: измерено {value}, в метаданных {metadata[key]}")
    pipeline = getattr(model, "best_estimator_", model)
    features = pipeline[:-1].transform(texts)
    unsupported = texts.str.contains(r"[\u0400-\u052f]", regex=True) | (features.getnnz(axis=1) == 0)
    over_limit = test["subject"].fillna("").str.len().gt(300) | test["body"].fillna("").str.len().gt(20000)
    result = {
        "recorded_metrics_match": True,
        "data_sha256": audit["data_sha256"],
        "model_sha256": sha256_file(args.model),
        "test_rows": len(test),
        **measured,
        "by_language": {
            lang: {
                "n": int((test["language"] == lang).sum()),
                "macro_f1": round(f1_score(test.loc[test["language"] == lang, "queue"], predicted[test["language"] == lang], average="macro", zero_division=0), 4),
            }
            for lang in sorted(test["language"].unique())
        },
        "rejected_by_text_policy": int(unsupported.sum()),
        "rejected_by_length_policy": int(over_limit.sum()),
        "eligible_for_api": int((~unsupported & ~over_limit).sum()),
        "note": "Проверка фиксированного артефакта; метрики рассчитаны на всём test, отдельно показано покрытие API.",
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
