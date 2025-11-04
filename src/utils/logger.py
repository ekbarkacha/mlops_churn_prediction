"""
Logger module - Centralized logging configuration
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
import colorlog

def get_logger(name: str, log_level: int = logging.INFO) -> logging.Logger:
    """
    Create and configure a logger with console and file handlers.
    
    Args:
        name (str): Logger name (usually __name__)
        log_level (int): Logging level (default: INFO)
    
    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler with colors
    console_handler = colorlog.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    console_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s%(reset)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(console_formatter)
    
    # File handler
    logs_dir = Path(__file__).parent.parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    log_file = logs_dir / f"churn_prediction_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level)
    
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


# Test logger
if __name__ == "__main__":
    test_logger = get_logger(__name__)
    test_logger.debug("🔍 This is a debug message")
    test_logger.info("✅ This is an info message")
    test_logger.warning("⚠️ This is a warning message")
    test_logger.error("❌ This is an error message")
    test_logger.critical("🔥 This is a critical message")