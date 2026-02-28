"""
Parse recording log files that record when videos were written to disk.
Log lines: "YYYY-MM-DD HH:MM:SS.mmm - Video recording written to hard disk <filename>"
"""
import os
import re
from typing import Optional

_RECORDING_LINE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+) - Video recording written to hard disk (.+)$"
)


def parse_recording_log(log_path: str) -> dict[str, str]:
    """
    Parse a recording log file and return a mapping from filename to timestamp.

    Lines matching "Video recording written to hard disk <filename>" are captured.
    If the same filename appears multiple times, the last occurrence wins.

    Returns:
        Dict mapping filename (e.g. "1.mp4") to timestamp string (e.g. "2025-10-31 16:42:09.357").
    """
    result: dict[str, str] = {}
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                m = _RECORDING_LINE.match(line)
                if m:
                    timestamp, filename = m.groups()
                    result[filename.strip()] = timestamp
    except OSError:
        pass
    return result


def get_video_timestamps(
    video_paths: list[str], log_path: Optional[str]
) -> Optional[dict[str, str]]:
    """
    Resolve recording timestamps for a list of video paths using an optional log file.

    If log_path is None or empty, returns None (caller should use blank for all).
    If multiple videos share the same basename, prints a warning and returns None.
    Otherwise returns a dict mapping each video_path to its timestamp string (or "" if not in log).

    Returns:
        None if no log, log unreadable, or duplicate basenames in video_paths.
        Otherwise dict[video_path -> timestamp string].
    """
    if not log_path or not log_path.strip():
        return None

    basenames = [os.path.basename(p) for p in video_paths]
    if len(basenames) != len(set(basenames)):
        print(
            "The log file is not valid for the videos being selected (multiple videos share the same filename). Time will be ignored."
        )
        return None

    log_entries = parse_recording_log(log_path)
    return {
        path: log_entries.get(os.path.basename(path), "")
        for path in video_paths
    }
