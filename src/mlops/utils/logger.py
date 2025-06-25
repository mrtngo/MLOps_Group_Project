"""Centralized logging configuration for the MLOps project."""

import logging
import os
from pathlib import Path


def setup_logger(name: str = None, level: str = "INFO") -> logging.Logger:
    """
    Setup centralized logger with proper configuration.
    
    Args:
        name: Logger name, defaults to root logger
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger only once
    root_logger = logging.getLogger()
    
    # Only configure if not already configured
    if not root_logger.handlers:
        log_level = getattr(logging, level.upper(), logging.INFO)
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(log_format, datefmt=date_format)
        console_handler.setFormatter(console_formatter)
        
        # File handler
        file_handler = logging.FileHandler(log_dir / "main.log")
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(log_format, datefmt=date_format)
        file_handler.setFormatter(file_formatter)
        
        # Configure root logger
        root_logger.setLevel(log_level)
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
    
    # Return named logger if requested
    if name:
        return logging.getLogger(name)
    return root_logger
