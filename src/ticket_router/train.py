"""Обучение и сравнение экспериментов.

Запуск: uv run ticket-router-train --data data/<csv> --out experiments/

Схема честного сравнения:
1. одно фиксированное разбиение train/val/test (splits.json);
2. модели обучаются только на train;
3. выбор лучшей конфигурации — по val (macro-F1);
4. финальная оценка лучшей модели — на test, один раз.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from .data import combine_text, load_dataset, make_splits, save_splits, sha256_file
from .experiments import EXPERIMENTS

DEFAULT_DATA = Path("data/aa_dataset-tickets-multi-lang-5-2-50-version.csv")


def run_experiments(data_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset(data_path)
    data_hash = sha256_file(data_path)
    print(f"Датасет: {data_path} | строк: {len(df)} | sha256={data_hash[:12]}…")

    train_df, val_df, test_df = make_splits(df)
    save_splits(train_df, val_df, test_df, out_dir)
    print(f"Разбиение: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    X_train = combine_text(train_df)
    X_val = combine_text(val_df)
    X_test = combine_text(test_df)
    y_train = train_df["queue"]
    y_val = val_df["queue"]
    y_test = test_df["queue"]

    results = []
    models: dict[str, object] = {}
    for name, spec in EXPERIMENTS.items():
        model = spec["make"]()
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - t0
        pred_val = model.predict(X_val)
        val_f1 = f1_score(y_val, pred_val, average="macro")
        results.append(
            {
                "experiment": name,
                "description": spec["description"],
                "val_macro_f1": round(val_f1, 4),
                "val_weighted_f1": round(f1_score(y_val, pred_val, average="weighted"), 4),
                "val_accuracy": round(accuracy_score(y_val, pred_val), 4),
                "fit_seconds": round(fit_seconds, 2),
            }
        )
        models[name] = model
        print(f"{name:24s} val_macro_f1={val_f1:.4f} ({fit_seconds:.1f}s)")

    results_df = pd.DataFrame(results).sort_values("val_macro_f1", ascending=False)
    results_df.to_csv(out_dir / "results.csv", index=False)
    print("\nresults.csv:\n", results_df.to_string(index=False))

    # Финальная оценка лучшей модели на test (один раз, после выбора по val).
    best_name = results_df.iloc[0]["experiment"]
    best_model = models[best_name]
    pred_test = best_model.predict(X_test)
    report = classification_report(y_test, pred_test, output_dict=True, zero_division=0)
    test_metrics = {
        "best_experiment": best_name,
        "test_macro_f1": round(f1_score(y_test, pred_test, average="macro"), 4),
        "test_weighted_f1": round(f1_score(y_test, pred_test, average="weighted"), 4),
        "test_accuracy": round(accuracy_score(y_test, pred_test), 4),
        "classification_report": report,
    }
    with open(out_dir / "best_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

    # Метрики лучшей модели на val и test по языкам.
    lang_report = {}
    for split_name, X, y in (("val", X_val, y_val), ("test", X_test, y_test)):
        mask_val = val_df["language"].values if split_name == "val" else test_df["language"].values
        for lang in sorted(set(mask_val)):
            m = mask_val == lang
            lang_report[f"{split_name}_{lang}"] = {
                "n": int(m.sum()),
                "macro_f1": round(f1_score(y[m], best_model.predict(X[m]), average="macro", zero_division=0), 4),
            }
    with open(out_dir / "best_by_language.json", "w", encoding="utf-8") as f:
        json.dump(lang_report, f, ensure_ascii=False, indent=2)

    # Сохранение артефакта: единый pipeline (векторизация + классификатор).
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    joblib.dump(best_model, artifacts_dir / "model.joblib")
    metadata = {
        "best_experiment": best_name,
        "data_sha256": data_hash,
        "data_source": "Kaggle: tobiasbueck/multilingual-customer-support-tickets v10 (зеркало GitHub: CHANDRADEEP2007/feedback-to-backlog-ai-copilot)",
        "classes": list(best_model.classes_),
        "n_train": len(train_df),
        "random_state": 42,
        **{k: v for k, v in test_metrics.items() if k != "classification_report"},
    }
    with open(artifacts_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Confusion matrix на test для лучшей модели.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        fig, ax = plt.subplots(figsize=(10, 8))
        ConfusionMatrixDisplay.from_predictions(y_test, pred_test, ax=ax, xticks_rotation=45)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_title(f"Confusion matrix (test): {best_name}")
        fig.tight_layout()
        (out_dir / "figures").mkdir(exist_ok=True)
        fig.savefig(out_dir / "figures" / "confusion_matrix_test.png", dpi=150)
        plt.close(fig)
    except ImportError:
        print("matplotlib недоступен — матрица ошибок не построена")

    print(f"\nЛучшая модель: {best_name} | test_macro_f1={test_metrics['test_macro_f1']}")
    return test_metrics


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Обучение экспериментов ticket-router")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=Path("experiments"))
    args = parser.parse_args()
    run_experiments(args.data, args.out)


if __name__ == "__main__":
    main()
