import logging
import sys


# Configure logging for the embedairr package
def setup_logging(level=logging.INFO):
    """Set up logging configuration for the PEPE package."""
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Get root logger for the package
    logger = logging.getLogger("src")
    logger.setLevel(level)

    # Remove existing handlers to avoid duplication
    logger.handlers.clear()

    # Add handler
    logger.addHandler(console_handler)

    return logger


# Set up default logging
setup_logging()
