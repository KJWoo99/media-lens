"""Unified SQLite cache for image features, video hashes, and CLIP embeddings."""

import os
import logging
import sqlite3
import hashlib
import threading
import numpy as np
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)

_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_cache")


class CacheManager:
    """Unified cache using SQLite for all media analysis results.
    Thread-safe via WAL mode and per-call connections with timeout."""

    def __init__(self, db_name="media_cache.db"):
        os.makedirs(_DB_DIR, exist_ok=True)
        self.db_path = os.path.join(_DB_DIR, db_name)
        self._lock = threading.Lock()
        self._init_db()

    @contextmanager
    def _conn(self):
        """Thread-safe connection context manager with WAL mode."""
        conn = sqlite3.connect(self.db_path, timeout=10, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        with self._conn() as conn:
            c = conn.cursor()

            # Video cache table
            c.execute('''CREATE TABLE IF NOT EXISTS video_cache (
                file_key TEXT PRIMARY KEY,
                file_path TEXT,
                file_size INTEGER,
                modified_time REAL,
                partial_hash TEXT,
                width INTEGER, height INTEGER,
                fps REAL, frame_count INTEGER, duration REAL,
                audio_present INTEGER,
                frame_hashes BLOB,
                cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )''')
            c.execute('CREATE INDEX IF NOT EXISTS idx_vc_size ON video_cache(file_size)')

            # Image feature cache (DINOv2)
            c.execute('''CREATE TABLE IF NOT EXISTS image_feature_cache (
                file_key TEXT PRIMARY KEY,
                file_path TEXT,
                file_size INTEGER,
                modified_time REAL,
                features BLOB,
                cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )''')

            # CLIP embedding cache
            c.execute('''CREATE TABLE IF NOT EXISTS clip_cache (
                file_key TEXT PRIMARY KEY,
                file_path TEXT,
                file_size INTEGER,
                modified_time REAL,
                model_hash TEXT,
                embedding BLOB,
                cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )''')

    @staticmethod
    def _file_key(filepath):
        """Generate unique key from full resolved path + file size + mtime."""
        p = Path(filepath)
        if not p.exists():
            return None
        st = p.stat()
        key_str = f"{st.st_size}_{st.st_mtime}_{p.resolve()}"
        return hashlib.md5(key_str.encode()).hexdigest()

    # ── Video cache ─────────────────────────────────────────────────────

    def get_video_info(self, filepath) -> Optional[Dict]:
        key = self._file_key(filepath)
        if not key:
            return None
        with self._conn() as conn:
            c = conn.cursor()
            c.execute('''SELECT file_size, modified_time, partial_hash,
                          width, height, fps, frame_count, duration,
                          audio_present, frame_hashes
                          FROM video_cache WHERE file_key=?''', (key,))
            row = c.fetchone()
        if not row:
            return None
        st = Path(filepath).stat()
        if st.st_size != row[0] or abs(st.st_mtime - row[1]) > 1.0:
            return None
        hashes = np.frombuffer(row[9], dtype=np.uint8).reshape(-1, 16)
        return {
            'path': str(filepath),
            'file_size': row[0], 'partial_hash': row[2],
            'width': row[3], 'height': row[4], 'fps': row[5],
            'frame_count': row[6], 'duration': row[7],
            'audio_present': bool(row[8]),
            'frame_hashes': [hashes[i] for i in range(len(hashes))]
        }

    def save_video_info(self, filepath, info: Dict):
        key = self._file_key(filepath)
        if not key:
            return
        st = Path(filepath).stat()
        hashes_bytes = np.array(info['frame_hashes'], dtype=np.uint8).tobytes()
        with self._conn() as conn:
            conn.execute('''INSERT OR REPLACE INTO video_cache
                (file_key, file_path, file_size, modified_time, partial_hash,
                 width, height, fps, frame_count, duration, audio_present, frame_hashes)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)''', (
                key, str(filepath), info['file_size'], st.st_mtime,
                info['partial_hash'], info['width'], info['height'],
                info['fps'], info['frame_count'], info['duration'],
                int(info['audio_present']), hashes_bytes))

    # ── Image feature cache ─────────────────────────────────────────────

    def get_image_features(self, filepath) -> Optional[np.ndarray]:
        key = self._file_key(filepath)
        if not key:
            return None
        with self._conn() as conn:
            c = conn.cursor()
            c.execute('SELECT file_size, modified_time, features FROM image_feature_cache WHERE file_key=?', (key,))
            row = c.fetchone()
        if not row:
            return None
        st = Path(filepath).stat()
        if st.st_size != row[0] or abs(st.st_mtime - row[1]) > 1.0:
            return None
        return np.frombuffer(row[2], dtype=np.float32).copy()

    def save_image_features(self, filepath, features: np.ndarray):
        key = self._file_key(filepath)
        if not key:
            return
        st = Path(filepath).stat()
        with self._conn() as conn:
            conn.execute('''INSERT OR REPLACE INTO image_feature_cache
                (file_key, file_path, file_size, modified_time, features)
                VALUES (?,?,?,?,?)''', (key, str(filepath), st.st_size, st.st_mtime,
                                         features.astype(np.float32).tobytes()))

    # ── CLIP cache ──────────────────────────────────────────────────────

    def get_clip_embedding(self, filepath, model_hash) -> Optional[np.ndarray]:
        key = self._file_key(filepath)
        if not key:
            return None
        with self._conn() as conn:
            c = conn.cursor()
            c.execute('''SELECT file_size, modified_time, embedding
                          FROM clip_cache WHERE file_key=? AND model_hash=?''', (key, model_hash))
            row = c.fetchone()
        if not row:
            return None
        st = Path(filepath).stat()
        if st.st_size != row[0] or abs(st.st_mtime - row[1]) > 1.0:
            return None
        return np.frombuffer(row[2], dtype=np.float32).copy()

    def save_clip_embedding(self, filepath, model_hash, embedding: np.ndarray):
        key = self._file_key(filepath)
        if not key:
            return
        st = Path(filepath).stat()
        with self._conn() as conn:
            conn.execute('''INSERT OR REPLACE INTO clip_cache
                (file_key, file_path, file_size, modified_time, model_hash, embedding)
                VALUES (?,?,?,?,?,?)''', (key, str(filepath), st.st_size, st.st_mtime,
                                           model_hash, embedding.astype(np.float32).tobytes()))

    # ── Batch operations ────────────────────────────────────────────────

    def _file_keys_batch(self, filepaths):
        """Generate file keys for multiple paths. Returns {path: (key, stat)} for valid files."""
        result = {}
        for fp in filepaths:
            p = Path(fp)
            try:
                if p.exists():
                    st = p.stat()
                    key_str = f"{st.st_size}_{st.st_mtime}_{p.resolve()}"
                    result[str(fp)] = (hashlib.md5(key_str.encode()).hexdigest(), st)
            except Exception:
                pass
        return result

    def get_image_features_batch(self, filepaths, expected_dim=768) -> Dict[str, np.ndarray]:
        """Batch lookup image features. Returns {path: features} for cache hits."""
        key_map = self._file_keys_batch(filepaths)
        if not key_map:
            return {}

        results = {}
        items = list(key_map.items())
        for chunk_start in range(0, len(items), 500):
            chunk = items[chunk_start:chunk_start + 500]
            keys = [key for _, (key, _) in chunk]
            path_by_key = {key: (path, st) for path, (key, st) in chunk}

            placeholders = ','.join('?' * len(keys))
            with self._conn() as conn:
                c = conn.cursor()
                c.execute(f'SELECT file_key, file_size, modified_time, features '
                          f'FROM image_feature_cache WHERE file_key IN ({placeholders})', keys)
                for row in c.fetchall():
                    fk, db_size, db_mtime, blob = row
                    if fk in path_by_key:
                        path, st = path_by_key[fk]
                        if st.st_size == db_size and abs(st.st_mtime - db_mtime) <= 1.0:
                            arr = np.frombuffer(blob, dtype=np.float32).copy()
                            if len(arr) == expected_dim:
                                results[path] = arr
        return results

    def save_image_features_batch(self, items: List[Tuple[str, np.ndarray]]):
        """Batch save image features. items: [(path, features), ...]"""
        rows = []
        for filepath, features in items:
            key = self._file_key(filepath)
            if not key:
                continue
            try:
                st = Path(filepath).stat()
                rows.append((key, str(filepath), st.st_size, st.st_mtime,
                             features.astype(np.float32).tobytes()))
            except Exception:
                pass
        if not rows:
            return
        with self._conn() as conn:
            conn.executemany('''INSERT OR REPLACE INTO image_feature_cache
                (file_key, file_path, file_size, modified_time, features)
                VALUES (?,?,?,?,?)''', rows)

    def get_clip_embeddings_batch(self, filepaths, model_hash) -> Dict[str, np.ndarray]:
        """Batch lookup CLIP embeddings. Returns {path: embedding} for cache hits."""
        key_map = self._file_keys_batch(filepaths)
        if not key_map:
            return {}

        results = {}
        items = list(key_map.items())
        for chunk_start in range(0, len(items), 500):
            chunk = items[chunk_start:chunk_start + 500]
            keys = [key for _, (key, _) in chunk]
            path_by_key = {key: (path, st) for path, (key, st) in chunk}

            placeholders = ','.join('?' * len(keys))
            params = keys + [model_hash]
            with self._conn() as conn:
                c = conn.cursor()
                c.execute(f'SELECT file_key, file_size, modified_time, embedding '
                          f'FROM clip_cache WHERE file_key IN ({placeholders}) AND model_hash=?',
                          params)
                for row in c.fetchall():
                    fk, db_size, db_mtime, blob = row
                    if fk in path_by_key:
                        path, st = path_by_key[fk]
                        if st.st_size == db_size and abs(st.st_mtime - db_mtime) <= 1.0:
                            results[path] = np.frombuffer(blob, dtype=np.float32).copy()
        return results

    def save_clip_embeddings_batch(self, items: List[Tuple[str, np.ndarray]], model_hash: str):
        """Batch save CLIP embeddings. items: [(path, embedding), ...]"""
        rows = []
        for filepath, embedding in items:
            key = self._file_key(filepath)
            if not key:
                continue
            try:
                st = Path(filepath).stat()
                rows.append((key, str(filepath), st.st_size, st.st_mtime,
                             model_hash, embedding.astype(np.float32).tobytes()))
            except Exception:
                pass
        if not rows:
            return
        with self._conn() as conn:
            conn.executemany('''INSERT OR REPLACE INTO clip_cache
                (file_key, file_path, file_size, modified_time, model_hash, embedding)
                VALUES (?,?,?,?,?,?)''', rows)

    # ── Maintenance ─────────────────────────────────────────────────────

    def clear_invalid(self):
        """Remove entries for files that no longer exist."""
        deleted = 0
        with self._conn() as conn:
            for table in ['video_cache', 'image_feature_cache', 'clip_cache']:
                c = conn.cursor()
                c.execute(f'SELECT file_key, file_path FROM {table}')
                for fk, fp in c.fetchall():
                    if not Path(fp).exists():
                        conn.execute(f'DELETE FROM {table} WHERE file_key=?', (fk,))
                        deleted += 1
        logger.info(f"Cache cleanup: removed {deleted} invalid entries")
        return deleted

    def get_stats(self):
        with self._conn() as conn:
            stats = {}
            for table in ['video_cache', 'image_feature_cache', 'clip_cache']:
                c = conn.cursor()
                c.execute(f'SELECT COUNT(*) FROM {table}')
                stats[table] = c.fetchone()[0]
        stats['db_size_mb'] = os.path.getsize(self.db_path) / (1024 ** 2) if os.path.exists(self.db_path) else 0
        return stats

    def get_clip_count_by_model(self):
        """Returns {model_hash: count} for clip_cache entries."""
        with self._conn() as conn:
            c = conn.cursor()
            c.execute('SELECT model_hash, COUNT(*) FROM clip_cache GROUP BY model_hash')
            return dict(c.fetchall())

    def clear_video_cache(self):
        """Delete all video cache entries."""
        with self._conn() as conn:
            conn.execute('DELETE FROM video_cache')

    def clear_image_features(self):
        """Delete all DINOv2 image feature cache entries."""
        with self._conn() as conn:
            conn.execute('DELETE FROM image_feature_cache')

    def clear_clip_cache(self, model_hash=None):
        """Delete CLIP/SigLIP2 embedding cache entries. Pass model_hash to clear one model only."""
        with self._conn() as conn:
            if model_hash:
                conn.execute('DELETE FROM clip_cache WHERE model_hash=?', (model_hash,))
            else:
                conn.execute('DELETE FROM clip_cache')

    def clear_all(self):
        """Delete all cache entries and VACUUM the DB to reclaim space."""
        with self._conn() as conn:
            for table in ['video_cache', 'image_feature_cache', 'clip_cache']:
                conn.execute(f'DELETE FROM {table}')
        # VACUUM must run outside a transaction
        conn = sqlite3.connect(self.db_path, timeout=10)
        try:
            conn.execute('VACUUM')
        finally:
            conn.close()
