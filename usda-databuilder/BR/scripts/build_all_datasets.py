try:
    from .process_mt_imea import run_mt_imea_pipeline
    from .process_pr_sima_prices import run_pr_sima_pipeline
    from .merge_mt_and_pr_data import run_merge_pipeline
except ImportError:  # pragma: no cover - fallback when executed as a script
    from process_mt_imea import run_mt_imea_pipeline
    from process_pr_sima_prices import run_pr_sima_pipeline
    from merge_mt_and_pr_data import run_merge_pipeline


def main() -> None:
    """Run the full ETL pipeline: MT costs -> PR SIMA -> merged dataset."""
    run_mt_imea_pipeline()
    run_pr_sima_pipeline()
    run_merge_pipeline()


if __name__ == "__main__":
    main()
