import os
from pathlib import Path
import re
import logging
import fnmatch


def get_directory_structure(root_dir):
    structure = f"Directory structure for: {os.path.abspath(root_dir)}\n---\n"
    root_path = Path(root_dir)
    if not root_path.is_dir():
        return f"Error: Not a valid directory: {root_dir}"
    exclude_dirs = {'.git', '__pycache__', '.venv', 'node_modules', '.DS_Store'}
    for path_object in root_path.rglob('*'):
        try:
            relative_path = path_object.relative_to(root_path)
            depth = len(relative_path.parts) - 1
            if any(part in exclude_dirs for part in relative_path.parts):
                continue
            indent = '  ' * depth
            prefix = "|-- "
            if path_object.is_dir():
                if path_object.name in exclude_dirs:
                    continue
                structure += f'{indent}{prefix}{path_object.name}/\n'
            elif path_object.is_file():
                if path_object.name in exclude_dirs:
                    continue
                structure += f'{indent}{prefix}{path_object.name}\n'
        except ValueError:
            continue
    structure += "---\n"
    return structure


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
