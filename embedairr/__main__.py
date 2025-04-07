from embedairr.model_selecter import select_model
from embedairr.parse_arguments import parse_arguments


def main():
    args = parse_arguments()
    if args.batch_writing:
        if "false" not in args.extract_cdr3_attention_matrices:
            raise ValueError(
                "Batch writing is not supported with CDR3 attention matrices. Set '--batch_writing False' to disable."
            )

    selected_model = select_model(args.model_name)

    embedder = selected_model(args)

    print("Embedder initialized")

    embedder.run()

    print("All outputs saved.")


if __name__ == "__main__":
    import multiprocessing

    try:
        multiprocessing.set_start_method("spawn", force=True)  # for MacOS compatibility
    except RuntimeError:
        pass  # start method already set
    main()
