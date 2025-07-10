import logging
from typing import Optional

def setup_logging(log_level: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Configures and returns a logger.

    Args:
        log_level (str): The logging level (e.g., 'INFO', 'DEBUG').
        log_file (str, optional): The path to the log file. If provided, logs are written to this file.
                                If not provided, logs are sent to the console.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger("default")
    logger.propagate = False
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers to reconfigure logger
    if logger.hasHandlers():
        logger.handlers.clear()

    if log_file:
        # Log to file with full timestamp
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        # Log to console with time-only format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger

def setup_llm_logger(log_level: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Configures and returns a logger for LLM prompts and outputs.

    Args:
        llm_log_file (str): The path to the LLM log file.

    Returns:
        logging.Logger: The configured logger instance.
    """
    llm_logger = logging.getLogger('llm_logger')
    llm_logger.propagate = False
    llm_logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers to reconfigure logger
    if llm_logger.hasHandlers():
        llm_logger.handlers.clear()

    if log_file:
        # Log to file with full timestamp
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        llm_logger.addHandler(file_handler)
    else:
        # Log to console with time-only format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        llm_logger.addHandler(stream_handler)

    return llm_logger