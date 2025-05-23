#!/usr/bin/env python3
"""
Run script for the Minitab-like application
"""

import sys
import os
import argparse
from pathlib import Path
from bs4 import BeautifulSoup
import textwrap

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

try:
    from minitab_app.modules.ui.main_window import MinitabMainWindow
except ImportError as e:
    print(f"Error importing MinitabMainWindow: {e}")
    print("\nTroubleshooting steps:")
    print("1. Make sure you're in the correct directory")
    print("2. Install all dependencies:")
    print("   pip install -r requirements.txt")
    print("3. Check if all required modules exist:")
    print("   - minitab_app/modules/ui/main_window.py")
    print("   - minitab_app/core/statistics.py")
    print("   - minitab_app/core/app.py")
    sys.exit(1)

# Path to converted HTML book (update if moved)
BOOK_HTML_PATH = Path(project_root) / "pdf_output" / "html_content" / "content.html"


def extract_toc(html_path: Path):
    """Return a list of headings that look like table-of-contents entries."""
    toc = []
    if not html_path.exists():
        return toc
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup.find_all(["h1", "h2", "h3"]):
        text = tag.get_text(" ", strip=True)
        # Simple heuristic: heading starts with a digit (chapter/section number)
        if text and text[0].isdigit():
            toc.append(text)
    return toc


def get_section_text(html_path: Path, query: str, span_chars: int = 5000):
    """Return plain-text content of the section whose heading matches *query* (case-insensitive).
    We stop collecting when we hit the next heading of same or higher level.
    """
    if not html_path.exists():
        return None
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8", errors="ignore"), "html.parser")

    # Find the first heading that contains the query text
    header_tag = soup.find(
        lambda tag: tag.name in ["h1", "h2", "h3"] and query.lower() in tag.get_text(" ", strip=True).lower()
    )
    if not header_tag:
        return None

    collected = []
    for sibling in header_tag.next_siblings:
        if getattr(sibling, "name", None) in ["h1", "h2", "h3"]:
            break
        # Keep text from NavigableString or other tags
        if hasattr(sibling, "get_text"):
            collected.append(sibling.get_text(" ", strip=True))
        else:
            collected.append(str(sibling).strip())
        # Safety guard to avoid extremely long outputs
        if sum(len(x) for x in collected) > span_chars:
            break
    plain = "\n".join(x for x in collected if x)
    return textwrap.dedent(plain).strip()


def launch_gui():
    """Launch the original PyQt GUI application."""
    app = QApplication(sys.argv)
    window = MinitabMainWindow()
    window.show()
    sys.exit(app.exec())


def launch_cli(args):
    """Simple interactive/command-line interface aligned with test_guide.md."""
    if args.toc:
        toc_entries = extract_toc(BOOK_HTML_PATH)
        if not toc_entries:
            print("[Error] Could not extract TOC from HTML. Make sure the file exists at", BOOK_HTML_PATH)
            return
        print("\nTable of Contents (from book):")
        for line in toc_entries:
            print(" ", line)
        return

    if args.section:
        section_text = get_section_text(BOOK_HTML_PATH, args.section)
        if not section_text:
            print(f"[Error] Section '{args.section}' not found in HTML book.")
        else:
            print(f"\n=== {args.section} ===\n")
            print(section_text)
        return

    if args.run_example:
        app = QApplication(sys.argv)  # Create QApplication instance
        from minitab_app.modules.stats import hypothesis as hp
        hp.run_example(args.run_example)
        app.exec()  # Start the event loop
        return

    # Interactive prompt displaying menu from test_guide.md
    print("\nMinitab-like CLI (type 'help' for commands, 'exit' to quit)")
    while True:
        try:
            cmd = input("minitab> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if cmd in {"exit", "quit"}:
            break
        if cmd == "help":
            print("Commands:\n  toc            - show table of contents from book\n  section <text> - show section containing <text>\n  gui            - launch graphical interface\n  exit           - quit CLI")
            continue
        if cmd == "toc":
            for line in extract_toc(BOOK_HTML_PATH):
                print(" ", line)
            continue
        if cmd.startswith("section "):
            query = cmd[len("section "):].strip()
            if not query:
                print("Usage: section <query text>")
                continue
            txt = get_section_text(BOOK_HTML_PATH, query)
            if txt:
                print(f"\n=== {query} ===\n{txt}\n")
            else:
                print("[Not found]")
            continue
        if cmd == "gui":
            print("Launching GUI...")
            launch_gui()
            break
        print("Unknown command. Type 'help' for list of commands.")


def main():
    parser = argparse.ArgumentParser(description="Run Minitab-like application (GUI or CLI).")
    parser.add_argument("--mode", choices=["gui", "cli"], default="gui", help="Launch in GUI (default) or CLI mode")
    parser.add_argument("--toc", action="store_true", help="Print table of contents extracted from the converted book HTML")
    parser.add_argument("--section", metavar="QUERY", help="Print the section whose heading contains QUERY from the book HTML")
    parser.add_argument("--run-example", metavar="NAME", help="Run built-in example (e.g., 'one-sample', 'two-sample') in CLI mode")
    args = parser.parse_args()

    # If specific args like --toc/--section given, force CLI mode
    if args.toc or args.section or args.run_example:
        args.mode = "cli"

    if args.mode == "gui":
        launch_gui()
    else:
        launch_cli(args)


if __name__ == "__main__":
    main() 