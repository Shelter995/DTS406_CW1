"""Entry point for traditional TF-IDF classification experiments."""

from __future__ import annotations

from config import parse_args
from trian import run_experiments


def main() -> None:
    args = parse_args()
    run_experiments(
        project_root=args.project_root,
        max_features=args.max_features,
        min_df=args.min_df,
    )


if __name__ == "__main__":
    main()
