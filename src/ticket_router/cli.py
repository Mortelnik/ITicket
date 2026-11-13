"""Клиент командной строки: uv run ticket-router 'subject' 'body'."""

from __future__ import annotations

from .predict import load_model, predict_queue


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Предсказание queue для тикета")
    parser.add_argument("subject", help="Тема обращения")
    parser.add_argument("body", help="Текст обращения")
    args = parser.parse_args()
    load_model()
    print(predict_queue(args.subject, args.body))


if __name__ == "__main__":
    main()
