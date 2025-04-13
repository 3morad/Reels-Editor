import logging
import time
import traceback
import os
from datetime import datetime
import functools
from typing import Any, Callable

def configure_logger(name: str) -> logging.Logger:
    """Configure and return a logger with standard formatting"""
    logger = logging.getLogger(name)
    
    # Set propagate to False to prevent duplicate logs
    logger.propagate = False
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger

def timed(func: Callable) -> Callable:
    """Decorator to time function execution"""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger = logging.getLogger(func.__module__)
        logger.debug(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper

def log_exceptions(func: Callable) -> Callable:
    """Decorator to log exceptions"""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = logging.getLogger(func.__module__)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
    return wrapper

def create_log_file(base_name="video_transformer"):
    """
    Create a log file with timestamp.
    
    Args:
        base_name: Base name for the log file
        
    Returns:
        Path to the created log file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return os.path.join(log_dir, f"{base_name}_{timestamp}.log")
