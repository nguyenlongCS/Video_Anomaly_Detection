import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logging(log_type: str = 'general') -> logging.Logger:
    """Setup logging configuration"""

    # Create logs directory structure
    log_dir = Path('logs')
    for subdir in ['training', 'inference', 'preprocessing', 'data']:
        (log_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Get current timestamp
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')

    # Setup logging format
    log_format = '%(asctime)s UTC - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # Create formatter
    formatter = logging.Formatter(log_format, date_format)

    # Create logger
    logger = logging.getLogger(f'anomaly_detection.{log_type}')
    logger.setLevel(logging.DEBUG)

    # Determine log file path
    if log_type in ['training', 'inference', 'preprocessing', 'data']:
        log_file = log_dir / log_type / f'{log_type}_{timestamp}.log'
    else:
        log_file = log_dir / f'general_{timestamp}.log'

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Log initial information
    logger.info('=' * 50)
    logger.info('Logging session started')
    logger.info(f'Log Type: {log_type}')
    logger.info(f'Created by: nguyenlongCS')
    logger.info(f'Date: 2025-02-24 12:36:01 UTC')
    logger.info('=' * 50)

    return logger