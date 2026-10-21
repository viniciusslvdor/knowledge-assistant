import logging


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Third-party libraries are noisy at INFO and may log request details.
    for noisy in (
        "httpx", "httpx2", "httpcore", "urllib3", "chromadb", "huggingface_hub",
        "sentence_transformers", "transformers",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)
