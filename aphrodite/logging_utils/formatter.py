import logging
import os
import sys
from pathlib import Path

from aphrodite import envs

class Colors:
    """ANSI color codes for terminal output"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    DEBUG = '\033[36m'                    # Cyan
    INFO = '\033[1m\033[36m'              # Bold Cyan
    WARNING = '\033[1m\033[33m'           # Bold Yellow
    ERROR = '\033[1m\033[31m'             # Bold Red
    CRITICAL = '\033[1m\033[41m\033[37m'  # Bold White on Red

    TIME = '\033[2m'          # Dim
    PATH = '\033[2m\033[34m'  # Dim Blue


def _supports_color() -> bool:
    """Check if the terminal supports color output"""
    if os.environ.get('NO_COLOR'):
        return False

    if os.environ.get('FORCE_COLOR'):
        return True

    if not hasattr(sys.stdout, 'isatty'):
        return False
    
    if not sys.stdout.isatty():
        return False

    term = os.environ.get('TERM', '')
    return term != 'dumb'


class NewLineFormatter(logging.Formatter):
    """Adds logging prefix to newlines to align multi-line messages with optional colors."""

    def __init__(self, fmt, datefmt=None, style="%"):
        super().__init__(fmt, datefmt, style)

        self.use_relpath = envs.APHRODITE_LOGGING_LEVEL == "DEBUG"
        if self.use_relpath:
            self.root_dir = Path(__file__).resolve().parent.parent.parent

        self.use_color = _supports_color() and os.environ.get('APHRODITE_LOGGING_COLOR', '1') in ('1', 'true', 'True')

        self.level_colors = {
            'DEBUG': Colors.DEBUG,
            'INFO': Colors.INFO,
            'WARNING': Colors.WARNING,
            'ERROR': Colors.ERROR,
            'CRITICAL': Colors.CRITICAL,
        }

    def format(self, record):
        def shrink_path(relpath: Path) -> str:
            """
            Shortens a file path for logging display:
            - Removes leading 'aphrodite' folder if present.
            - If path starts with 'v1',
            keeps the first two and last two levels,
            collapsing the middle as '...'.
            - Otherwise, keeps the first and last two levels,
            collapsing the middle as '...'.
            - If the path is short, returns it as-is.
            - Examples:
            aphrodite/quantization/utils/fp8_utils.py ->
            aphrodite/.../fp8_utils.py
            aphrodite/quantization/awq.py ->
            quantization/awq.py
            aphrodite/v1/attention/backends/mla/common.py ->
            v1/attention/backends/mla/common.py

            Args:
                relpath (Path): The relative path to be shortened.
            Returns:
                str: The shortened path string for display.
            """
            parts = list(relpath.parts)
            new_parts = []
            if parts and parts[0] == "aphrodite":
                parts = parts[1:]
            if parts and parts[0] == "v1":
                new_parts += parts[:2]
                parts = parts[2:]
            elif parts:
                new_parts += parts[:1]
                parts = parts[1:]
            if len(parts) > 2:
                new_parts += ["..."] + parts[-2:]
            else:
                new_parts += parts
            return "/".join(new_parts)

        if self.use_relpath:
            abs_path = getattr(record, "pathname", None)
            if abs_path:
                try:
                    relpath = Path(abs_path).resolve().relative_to(self.root_dir)
                except Exception:
                    relpath = Path(record.filename)
            else:
                relpath = Path(record.filename)
            record.fileinfo = shrink_path(relpath)
        else:
            record.fileinfo = record.filename

        # it's a given that the fileinfo ends with .py
        if record.fileinfo.endswith('.py'):
            record.fileinfo = record.fileinfo[:-3]

        max_fileinfo_width = 15
        if len(record.fileinfo) > max_fileinfo_width:
            record.fileinfo = "..." + record.fileinfo[-(max_fileinfo_width - 3):]

        msg = super().format(record)

        # for brevity
        if 'WARNING' in msg:
            msg = msg.replace('WARNING', 'WARN', 1)

        if self.use_color:
            level_color = self.level_colors.get(record.levelname, '')

            # Format: PREFIX + LEVEL + TIME + [PATH:LINE] + MESSAGE
            level_str = 'WARN' if record.levelname == 'WARNING' else record.levelname

            if level_str in msg:
                # Color the level
                msg = msg.replace(level_str, f"{level_color}{level_str}{Colors.RESET}", 1)

            asctime = self.formatTime(record, self.datefmt)
            if asctime in msg:
                msg = msg.replace(asctime, f"{Colors.TIME}{asctime}{Colors.RESET}", 1)

            # Match the formatted fileinfo with padding (left-aligned 15 chars + : + 4 digit lineno)
            fileinfo_str = f"[{record.fileinfo:<15}:{record.lineno:>4}]"
            if fileinfo_str in msg:
                msg = msg.replace(fileinfo_str, f"{Colors.PATH}{fileinfo_str}{Colors.RESET}", 1)

        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\r\n" + parts[0])
        return msg
