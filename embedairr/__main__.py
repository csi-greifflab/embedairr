import logging
from embedairr.model_selecter import select_model
from embedairr.parse_arguments import parse_arguments

logger = logging.getLogger("embedairr.__main__")


def main():
    args = parse_arguments()
    if args.batch_writing:
        if "false" not in args.extract_cdr3_attention_matrices:
            raise ValueError(
                "Batch writing is not supported with CDR3 attention matrices. Set '--batch_writing False' to disable."
            )

    selected_model = select_model(args.model_name)

    embedder = selected_model(args)

    logger.info("Embedder initialized")

    embedder.run()

    logger.info("All outputs saved.")


if __name__ == "__main__":
    main()
