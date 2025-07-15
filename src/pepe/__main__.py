import logging
from pepe.parse_arguments import parse_arguments

logger = logging.getLogger("src.__main__")


def main():
    args = parse_arguments()

    # Lazy import to avoid loading heavy dependencies during --help
    from pepe.model_selecter import select_model

    selected_model = select_model(args.model_name)

    embedder = selected_model(args)

    logger.info("Embedder initialized")

    embedder.run()

    logger.info("All outputs saved.")


if __name__ == "__main__":
    main()
