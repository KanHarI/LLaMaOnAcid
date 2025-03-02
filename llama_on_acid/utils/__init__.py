"""
Utility functions for the LLaMa on Acid project.
"""

import subprocess
from typing import Optional


def get_git_commit_hash() -> Optional[str]:
    """
    Get the current git commit hash at runtime.

    Returns:
        The git commit hash as a string, or None if it couldn't be retrieved
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        # Git command failed or git is not installed
        return None
