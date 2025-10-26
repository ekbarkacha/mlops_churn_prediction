import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import logging
from datetime import datetime
from src.utils.const import LOG_DIR

os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"LOGS_{datetime.now().strftime('%Y%m%d')}.log")

def get_logger(name: str):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        console = logging.StreamHandler()
        console_fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console.setFormatter(console_fmt)
        logger.addHandler(console)

        fileh = logging.FileHandler(LOG_FILE)
        file_fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fileh.setFormatter(file_fmt)
        logger.addHandler(fileh)

        logger.propagate = False

    return logger
