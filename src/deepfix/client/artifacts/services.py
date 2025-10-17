from __future__ import annotations

import hashlib


class ChecksumService:
    def compute_sha256(self, file_path: str) -> str:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
