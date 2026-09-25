# Chamelion modifications are licensed under GPL-3.0-or-later.
# The original MapMOS copyright and MIT license notice follow.
#
# MIT License
#
# Copyright (c) 2023 Benedikt Mersch, Tiziano Guadagnino, Ignacio Vizzo, Cyrill Stachniss
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import functools as ft
import hashlib
import io
import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path

import numpy as np
import torch


class ArrayCache:
    """Process-safe SQLite cache of numeric NPZ payloads, never Python objects.

    A separate directory keeps legacy pickle caches untouched. Connections are
    short-lived so DataLoader workers never inherit live SQLite connections.
    The size limit applies to live serialized payloads; SQLite reuses freed pages.
    """

    def __init__(self, directory, size_limit):
        self.directory = str(Path(directory) / "numeric-npz-v1")
        self.size_limit = int(size_limit)
        Path(self.directory).mkdir(parents=True, exist_ok=True)
        self.path = str(Path(self.directory) / "arrays.sqlite3")
        with closing(self._connect()) as db, db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS entries "
                "(key TEXT PRIMARY KEY, payload BLOB NOT NULL, "
                "nbytes INTEGER NOT NULL, accessed REAL NOT NULL)"
            )
            db.execute("CREATE INDEX IF NOT EXISTS access_order ON entries(accessed)")

    def _connect(self):
        return sqlite3.connect(self.path, timeout=60)

    def get(self, key, default=None, retry=True):
        with closing(self._connect()) as db, db:
            row = db.execute("SELECT payload FROM entries WHERE key=?", (key,)).fetchone()
            if row is None:
                return default
            db.execute("UPDATE entries SET accessed=? WHERE key=?", (time.time(), key))
        with np.load(io.BytesIO(row[0]), allow_pickle=False) as data:
            names = [f"arr_{i}" for i in range(8)]
            if set(data.files) != set(names):
                raise ValueError("Invalid cache payload: expected eight numeric arrays")
            arrays = [data[name] for name in names]
            if any(a.dtype != np.float32 or a.ndim != 2 for a in arrays):
                raise ValueError("Invalid cache array type or shape")
            return tuple(torch.from_numpy(a) for a in arrays)

    def set(self, key, value):
        if not isinstance(value, tuple) or len(value) != 8:
            raise ValueError("Cache values must contain eight CPU float32 tensors")
        arrays = []
        for tensor in value:
            if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
                    or tensor.dtype != torch.float32 or tensor.ndim != 2
                    or tensor.requires_grad):
                raise ValueError("Cache accepts only 2D CPU float32 tensors without gradients")
            arrays.append(tensor.numpy())
        buffer = io.BytesIO()
        # Numeric dtypes are enforced above; no object arrays can be serialized.
        np.savez(buffer, *arrays)
        payload = buffer.getvalue()
        if len(payload) > self.size_limit:
            return False
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "INSERT OR REPLACE INTO entries VALUES (?, ?, ?, ?)",
                (key, payload, len(payload), time.time()),
            )
            total = db.execute("SELECT COALESCE(SUM(nbytes), 0) FROM entries").fetchone()[0]
            while total > self.size_limit:
                oldest, size = db.execute(
                    "SELECT key, nbytes FROM entries ORDER BY accessed LIMIT 1"
                ).fetchone()
                db.execute("DELETE FROM entries WHERE key=?", (oldest,))
                total -= size
        return True

    def close(self):
        """No persistent connection is held."""


def get_cache(directory, size_limit: int = 500):
    return ArrayCache(directory, size_limit=size_limit * 2**30)


def memoize():
    """Cache dataset results using numeric payloads and JSON-derived keys."""
    def decorator(func):
        @ft.wraps(func)
        def wrapper(self, *args, **kwargs):
            if not self.use_cache:
                return func(self, *args, **kwargs)
            key = wrapper.__cache_key__(self, *args, **kwargs)
            missing = object()
            result = self.cache.get(key, default=missing)
            if result is missing:
                result = func(self, *args, **kwargs)
                self.cache.set(key, result)
            return result

        def cache_key(self, *args, **kwargs):
            encoded = json.dumps(
                [func.__module__ + "." + func.__qualname__, args, kwargs],
                sort_keys=True, separators=(",", ":"), allow_nan=False,
            ).encode("utf-8")
            return hashlib.sha256(encoded).hexdigest()

        wrapper.__cache_key__ = cache_key
        return wrapper
    return decorator
