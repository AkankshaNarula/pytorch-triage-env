"""
VirtualFileSystem: stateful in-memory file store for one episode.
"""
from __future__ import annotations
import difflib
from typing import Dict, List, Optional


class VirtualFileSystem:
    def __init__(self, initial_files: Dict[str, str]):
        self._original: Dict[str, str] = dict(initial_files)
        self._current:  Dict[str, str] = dict(initial_files)

    def read(self, filename: str) -> str:
        if filename not in self._current:
            available = ", ".join(sorted(self._current.keys()))
            return f"ERROR: File '{filename}' not found.\nAvailable files: {available}"
        return self._current[filename]

    def edit(self, filename: str, old_str: str, new_str: str) -> str:
        if filename not in self._current:
            return f"ERROR: File '{filename}' not found."
        content = self._current[filename]
        if old_str not in content:
            lines = content.split("\n")
            preview = "\n".join(lines[:8]) + ("\n..." if len(lines) > 8 else "")
            return (
                f"ERROR: old_str not found in '{filename}'.\n"
                f"You must copy old_str EXACTLY from the file content.\n\n"
                f"File preview (first 8 lines):\n{preview}"
            )
        self._current[filename] = content.replace(old_str, new_str, 1)
        return f"✓ Edit applied to {filename}."

    def git_diff(self, filename: Optional[str] = None) -> str:
        targets = [filename] if filename else list(self._original.keys())
        diffs = []
        for fname in targets:
            orig = self._original.get(fname, "").splitlines(keepends=True)
            curr = self._current.get(fname, "").splitlines(keepends=True)
            diff = list(difflib.unified_diff(orig, curr, fromfile=f"a/{fname}", tofile=f"b/{fname}"))
            if diff:
                diffs.extend(diff)
        return "".join(diffs) if diffs else "No changes from original."

    @property
    def files(self) -> Dict[str, str]:
        return dict(self._current)

    @property
    def changed_files(self) -> List[str]:
        return [f for f in self._current if self._current[f] != self._original[f]]

    def reset(self, initial_files: Dict[str, str]) -> None:
        self._original = dict(initial_files)
        self._current  = dict(initial_files)
