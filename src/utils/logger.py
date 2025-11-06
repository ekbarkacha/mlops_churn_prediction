"""
Module: logger.py
=================

Author: AIMS-AMMI STUDENT
Created: October/November 2025
Description:
------------
This module provides a standardized logger configuration for all components in 
the MLOps project. It ensures that all scripts, pipelines, and API services 
write consistent, timestamped logs to both the console and a rotating daily log file.

Features:
---------
- Logs are saved to a timestamped log file (e.g., `LOGS_20251105.log`).
- Also streams logs to the console for real-time visibility.
- Prevents duplicate handlers when multiple modules import the logger.
- Uses INFO level by default, suitable for production monitoring.

"""
# Imports and setup
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import logging
from datetime import datetime
# Custom utility imports
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
