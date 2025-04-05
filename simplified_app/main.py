"""
Minitab-Like Application - Simplified Version
Main entry point
"""

import sys
from simplified_app.core.app import MinitabApp

def main():
    """Main entry point function"""
    app = MinitabApp()
    return app.run()

if __name__ == "__main__":
    sys.exit(main())
