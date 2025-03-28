"""
Entry point for the Minitab-like application.
This script provides backward compatibility with the original monolithic application,
but uses the new modularized code structure.
"""

import sys
import traceback
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("minitab_app.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("minitab_app")

try:
    from PyQt6.QtWidgets import QApplication
    logger.info("PyQt6 imported successfully")
except ImportError as e:
    logger.error(f"Failed to import PyQt6: {e}")
    sys.exit(1)

try:
    from src.gui.main_window import MinitabLikeApp
    logger.info("MinitabLikeApp imported successfully")
except ImportError as e:
    logger.error(f"Failed to import MinitabLikeApp: {e}")
    traceback.print_exc()
    sys.exit(1)

def main():
    """Run the application using the modularized code."""
    logger.info("Starting Minitab-like application...")
    try:
        app = QApplication(sys.argv)
        logger.info("Creating main window...")
        window = MinitabLikeApp()
        logger.info("Showing main window...")
        window.show()
        logger.info("Entering Qt event loop...")
        sys.exit(app.exec())
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.critical(f"Unhandled exception: {str(e)}")
        traceback.print_exc()
        sys.exit(1) 