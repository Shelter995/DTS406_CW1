"""Entry point for deep learning classification experiments."""

from __future__ import annotations

from config import parse_args
from train import run_experiments


def main() -> None:
    args = parse_args()
    run_experiments(args)


if __name__ == "__main__":
    main()
