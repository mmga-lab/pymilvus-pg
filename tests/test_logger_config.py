"""
Test cases for logger configuration functionality.

This module tests the logging configuration and level management
features of the pymilvus-pg package.
"""

from pymilvus_pg import logger, set_logger_level


class TestLoggerConfig:
    """Test logger configuration functionality."""

    def test_logger_exists(self):
        """Test that logger instance exists and is properly configured."""
        assert logger is not None
        assert hasattr(logger, "info")
        assert hasattr(logger, "debug")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "error")

    def test_set_logger_level_info(self):
        """Test setting logger level to INFO."""
        set_logger_level("INFO")
        # Test that logger can handle INFO level messages
        logger.info("Test INFO message")
        logger.debug("Test DEBUG message")

    def test_set_logger_level_debug(self):
        """Test setting logger level to DEBUG."""
        set_logger_level("DEBUG")
        # Test that logger can handle DEBUG level messages
        logger.debug("Test DEBUG message")
        logger.info("Test INFO message")

    def test_set_logger_level_warning(self):
        """Test setting logger level to WARNING."""
        set_logger_level("WARNING")
        # Test that logger can handle WARNING level messages
        logger.warning("Test WARNING message")
        logger.error("Test ERROR message")

    def test_set_logger_level_error(self):
        """Test setting logger level to ERROR."""
        set_logger_level("ERROR")
        # Test that logger can handle ERROR level messages
        logger.error("Test ERROR message")

    def test_invalid_logger_level(self):
        """Test setting invalid logger level."""
        # This should either handle gracefully or raise appropriate error
        try:
            set_logger_level("INVALID_LEVEL")
        except (ValueError, AttributeError):
            # Expected behavior for invalid level
            pass

    def test_logger_message_formatting(self):
        """Test that logger properly formats messages."""
        test_message = "Test message with formatting: {value}"

        # These should not raise exceptions
        logger.info(test_message, value=123)
        logger.debug("Debug message with multiple params: %s, %d", "string", 42)
        logger.warning("Warning with dict: %s", {"key": "value"})

    def test_logger_with_exception(self):
        """Test logger handling of exceptions."""
        try:
            raise ValueError("Test exception for logging")
        except ValueError as e:
            logger.error("Exception occurred: %s", str(e))
            logger.exception("Exception with traceback")
