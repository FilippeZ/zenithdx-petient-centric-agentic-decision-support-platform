"""
Custom logging module with colorized output.

This module provides a logger with ANSI color formatting for different log levels.
It defines a `ColorFormatter` to apply colors and a `setup_logger` function to
initialize logging with colorized output.

Log Levels and Colors:
    - DEBUG: Cyan
    - INFO: Green
    - WARNING: Yellow
    - ERROR: Red
    - CRITICAL: Magenta
"""

import logging

# ANSI escape codes for colors
COLOR_MAP = {
    "DEBUG": "\033[96m",  # Cyan
    "INFO": "\033[92m",  # Green
    "WARNING": "\033[93m",  # Yellow
    "ERROR": "\033[91m",  # Red
    "CRITICAL": "\033[95m",  # Magenta
    "RESET": "\033[0m",  # Reset
}


# Custom log formatter function
class ColorFormatter(logging.Formatter):
    """
    Custom log formatter that adds color to log messages based on severity level.

    This class extends `logging.Formatter` and overrides `format` to insert ANSI
    escape codes for colored log output. The colors are mapped as follows:

    - DEBUG: Cyan
    - INFO: Green
    - WARNING: Yellow
    - ERROR: Red
    - CRITICAL: Magenta

    Example:
        log_message = ColorFormatter().format(record)
    """

    def format(self, record):
        log_color = COLOR_MAP.get(record.levelname, COLOR_MAP["RESET"])
        log_message = super().format(record)
        return f"{log_color}{log_message}{COLOR_MAP['RESET']}"


# Function to configure logging
def setup_logger():
    """
    Configures the logger with colorized output and a custom log format.

    This function sets up the logging configuration to display log messages with
    different colors based on their severity level. The log level is set to `INFO`,
    and a `StreamHandler` is used to print messages to the console.

    Returns:
        logger: The configured logger instance.

    Example:
        logger = setup_logger()
        logger.info("This is an info message.")
        logger.error("This is an error message.")
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # Apply colored formatter to all handlers
    logger = logging.getLogger()
    for handler in logger.handlers:
        handler.setFormatter(ColorFormatter("%(levelname)s: %(message)s"))

    return logger
