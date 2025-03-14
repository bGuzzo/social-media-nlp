# Create & configure logger
import logging

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] - [%(name)s] - [%(levelname)s] - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    __LOGGER = logger
    logger.info("Logger initialized")   
    return logger
