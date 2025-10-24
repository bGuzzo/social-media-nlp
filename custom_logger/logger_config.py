"""
This script provides a centralized and standardized way to create and configure loggers
for the entire project. It defines a single function, `get_logger`, that returns a
logger instance with a predefined configuration, ensuring that all log messages
across the project have a consistent format and level.

The use of a dedicated logger configuration script is a standard best practice in
software engineering. It promotes:

- **Consistency**: All log messages will have the same format, making them easier to
  read and parse.
- **Centralized Control**: The logging configuration can be easily modified in this
  single file, without having to change the code in every module that uses logging.
- **Readability**: It separates the logging configuration from the application logic,
  making the code cleaner and easier to understand.

The logger is configured to output messages to the console with a format that includes
the timestamp, logger name, log level, and the log message itself. This provides
rich contextual information for debugging and monitoring the application.
"""

import logging

def get_logger(name: str) -> logging.Logger:
    """
    Creates and configures a logger with a standardized format.

    Args:
        name (str): The name of the logger, typically the name of the module
                    in which the logger is being used.

    Returns:
        logging.Logger: The configured logger instance.
    """
    # Get a logger instance.
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create a console handler and set its level.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a formatter and set it for the handler.
    formatter = logging.Formatter("[%(asctime)s] - [%(name)s] - [%(levelname)s] - %(message)s")
    console_handler.setFormatter(formatter)

    # Add the handler to the logger.
    logger.addHandler(console_handler)

    logger.info("Logger initialized")
    return logger