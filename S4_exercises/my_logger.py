"""
Loguru logging example demonstrating different log levels.
"""

import sys

from loguru import logger

# Configure logging with multiple handlers
# Remove default handler
logger.remove()

# Add file handler with rotation, retention, and compression
logger.add(
    "my_log.log",
    level="DEBUG",
    rotation="100 MB",  # Rotate when file reaches 100 MB
    retention=5,  # Keep only last 5 rotated log files
    compression="gz",  # Compress rotated logs to .gz format
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {extra} | {message}",
)

# Add terminal handler - only shows WARNING and above in console
logger.add(sys.stderr, level="WARNING")


def main():
    """Demonstrate logging at different levels using loguru."""

    # DEBUG level - detailed diagnostic information
    logger.debug("Initializing application with config: {config_file}", config_file="config.yaml")

    # INFO level - general informational messages
    logger.info("Application started successfully")

    # Contextualize: Add context information to logs within a block
    # Example 1: Simulating user request processing
    with logger.contextualize(user_id="user_12345", request_id="REQ-001"):
        logger.info("Processing user request")
        logger.debug("Fetching user data from database")
        logger.info("User request completed successfully")

    # Example 2: Simulating a transaction
    with logger.contextualize(transaction_id="TXN-789", amount=150.50):
        logger.info("Starting payment transaction")
        logger.debug("Validating payment method")

    # WARNING level - potentially problematic situations
    logger.warning("Configuration file not found, using defaults")

    # ERROR level - errors that allow the program to continue
    logger.error("File not found: /path/to/data.csv")

    # CRITICAL level - critical errors that might stop the program
    logger.critical("Out of memory! Cannot continue execution")


if __name__ == "__main__":
    main()
