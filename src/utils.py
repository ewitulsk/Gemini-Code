import os
from pathlib import Path
import re
import logging
import fnmatch

def load_raw_ignore_patterns(ignore_file_path: str) -> list[str]:
    patterns = []
    if os.path.exists(ignore_file_path):
        logging.info(f"Loading ignore patterns from: {ignore_file_path}")
        try:
            with open(ignore_file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        patterns.append(line)
                        logging.debug(f"  Loaded raw pattern: {line}")
        except Exception as e:
            logging.error(f"Error reading ignore file {ignore_file_path}: {e}")
    else:
        logging.info(
            f"No .indexignore file found at {ignore_file_path}. No files will be ignored by pattern."
        )
    print(f"Ignore Patterns: {patterns}")
    return patterns
