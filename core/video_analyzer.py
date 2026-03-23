"""Video duplicate detection engine (improved from original 비디오분석 project).

Fixes from original:
- Multiprocessing uses module-level functions (pickle-safe)
- Cancel support via is_processing flag
- FFmpeg availability check
- Single-folder mode added
"""

import os
import logging
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from pathlib import Path

logger = logging.getLogger(__name__)
from dataclasses import dataclass
from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils.video_utils import (
    collect_videos, get_video_metadata, extract_frames_ffmpeg,
    extract_frames_opencv, check_ffmpeg
)
from core.cache_manager import CacheManager


@dataclass
class VideoInfo:
    path: str
    file_size: int
    partial_hash: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    frame_hashes: List[np.ndarray]
    audio_present: bool


# ── Module-level functions for multiprocessing ─────────────────────────

def _get_partial_hash(filepath, sample_size=4 * 1024 * 1024):
    """Compute partial file hash from start/middle/end (4MB each, 12MB total)."""
    try:
        file_size = filepath.stat().st_size
        h = hashlib.md5()
        with open(filepath, "rb") as f:
            h.update(f.read(min(sample_size, file_size)))
            if file_size > sample_size * 3:
                f.seek(file_size // 2 - sample_size // 2)
                h.update(f.read(sample_size))
            if file_size > sample_size * 2:
                f.seek(max(0, file_size - sample_size))
                h.update(f.read(sample_size))
        return h.hexdigest()
    except Exception:
        return None


def _extract_frame_hash(frame):
    """Compute perceptual hash for a single frame."""
    small = cv2.resize(frame, (8, 8), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    dct = cv2.dct(np.float32(gray))
    dct_low = dct[:4, :4]
    avg = dct_low.mean()
    return (dct_low > avg).astype(np.uint8).flatten()


def _analyze_single_video(filepath):
    """Analyze a single video file (module-level for multiprocessing)."""
    try:
        filepath = Path(filepath)
        metadata = get_video_metadata(filepath)
        if not metadata:
            return None

        fps = metadata['fps'] or 30.0
        sample_interval = max(1, int(fps))

        # Try FFmpeg first, then OpenCV
        raw_frames = extract_frames_ffmpeg(filepath, sample_interval)
        if not raw_frames:
            raw_frames = extract_frames_opencv(filepath, sample_interval)

        frame_hashes = [_extract_frame_hash(f) for f in raw_frames]

        # Skip consecutive similar frames (hamming distance <= 2)
        if len(frame_hashes) > 2:
            filtered = [frame_hashes[0]]
            for h in frame_hashes[1:]:
                if np.sum(filtered[-1] != h) > 2:
                    filtered.append(h)
            if len(filtered) >= 2:
                frame_hashes = filtered

        file_size = filepath.stat().st_size
        duration = metadata['duration']
        video_bitrate = (file_size * 8) / duration if duration > 0 else 0
        expected = metadata['width'] * metadata['height'] * fps * 0.1
        audio_present = video_bitrate > expected * 1.2

        return VideoInfo(
            path=str(filepath),
            file_size=file_size,
            partial_hash=_get_partial_hash(filepath),
            width=metadata['width'],
            height=metadata['height'],
            fps=fps,
            frame_count=metadata['frame_count'],
            duration=duration,
            frame_hashes=frame_hashes,
            audio_present=audio_present
        )
    except Exception as e:
        logger.warning(f"Video analysis error ({filepath}): {e}")
        return None


class VideoAnalyzer:
    """Video duplicate detection engine."""

    def __init__(self, use_gpu=True, num_workers=None, use_cache=True):
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.num_workers = num_workers or max(1, os.cpu_count() - 2)
        self.cache = CacheManager() if use_cache else None
        self.is_processing = False

        # Check FFmpeg
        self.ffmpeg_ok, self.ffmpeg_msg = check_ffmpeg()

    def set_use_cache(self, enabled):
        """Enable or disable cache usage."""
        if enabled and self.cache is None:
            self.cache = CacheManager()
        elif not enabled:
            self.cache = None

    def analyze_video(self, filepath):
        """Analyze single video with cache support."""
        filepath = Path(filepath)

        if self.cache:
            cached = self.cache.get_video_info(filepath)
            if cached:
                return VideoInfo(**cached)

        info = _analyze_single_video(filepath)
        if info and self.cache:
            self.cache.save_video_info(filepath, {
                'file_size': info.file_size, 'partial_hash': info.partial_hash,
                'width': info.width, 'height': info.height,
                'fps': info.fps, 'frame_count': info.frame_count,
                'duration': info.duration, 'frame_hashes': info.frame_hashes,
                'audio_present': info.audio_present
            })
        return info

    def analyze_videos(self, video_files, progress_callback=None):
        """Analyze multiple videos with multiprocessing."""
        results = []
        # Separate cached vs uncached
        uncached = []
        for vf in video_files:
            if self.cache:
                cached = self.cache.get_video_info(vf)
                if cached:
                    results.append(VideoInfo(**cached))
                    continue
            uncached.append(vf)

        if progress_callback:
            progress_callback(len(results), len(video_files))

        if not uncached:
            return results

        # Process uncached videos in parallel
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(_analyze_single_video, str(vf)): vf for vf in uncached}
            for future in as_completed(futures):
                if not self.is_processing:
                    executor.shutdown(wait=False, cancel_futures=True)
                    return results
                info = future.result()
                if info:
                    results.append(info)
                    if self.cache:
                        self.cache.save_video_info(Path(info.path), {
                            'file_size': info.file_size, 'partial_hash': info.partial_hash,
                            'width': info.width, 'height': info.height,
                            'fps': info.fps, 'frame_count': info.frame_count,
                            'duration': info.duration, 'frame_hashes': info.frame_hashes,
                            'audio_present': info.audio_present
                        })
                if progress_callback:
                    progress_callback(len(results), len(video_files))

        return results

    def calculate_frame_similarity(self, hashes1, hashes2):
        """GPU-accelerated frame hash similarity."""
        t1 = torch.from_numpy(np.array(hashes1)).float().to(self.device)
        t2 = torch.from_numpy(np.array(hashes2)).float().to(self.device)
        return F.cosine_similarity(t1.unsqueeze(1), t2.unsqueeze(0), dim=2).cpu().numpy()

    def find_partial_match(self, short_hashes, long_hashes, threshold=0.85):
        """Sliding window search for partial video match."""
        if not short_hashes or not long_hashes or len(short_hashes) > len(long_hashes):
            return None

        window = len(short_hashes)
        best_score = 0
        best_match = None
        short_t = torch.from_numpy(np.array(short_hashes)).float().to(self.device)

        for start in range(len(long_hashes) - window + 1):
            win_t = torch.from_numpy(np.array(long_hashes[start:start + window])).float().to(self.device)
            sim = F.cosine_similarity(short_t, win_t, dim=1).mean().item()
            if sim > best_score:
                best_score = sim
                best_match = {'start_idx': start, 'end_idx': start + window, 'similarity': sim}

        return best_match if best_score >= threshold else None

    def compare_videos(self, info1, info2):
        """Detailed comparison of two videos."""
        result = {
            'video1': info1.path, 'video2': info2.path,
            'match_type': None, 'confidence': 0.0, 'details': {}
        }

        # Exact duplicate
        if info1.partial_hash == info2.partial_hash and info1.file_size == info2.file_size:
            result['match_type'] = 'exact_duplicate'
            result['confidence'] = 1.0
            result['details'] = {
                'file_size': f"{info1.file_size / (1024 ** 2):.2f}MB",
                'duration': f"{info1.duration:.1f}s"
            }
            return result

        # Same content check
        if abs(info1.duration - info2.duration) < 2.0:
            if info1.width == info2.width and info1.height == info2.height:
                if info1.frame_hashes and info2.frame_hashes:
                    n = min(len(info1.frame_hashes), len(info2.frame_hashes))
                    sim = self.calculate_frame_similarity(info1.frame_hashes[:n], info2.frame_hashes[:n])
                    avg = np.mean(np.diag(sim))
                    if avg >= 0.9:
                        result['match_type'] = 'same_content'
                        result['confidence'] = float(avg)
                        result['details'] = {
                            'resolution': f"{info1.width}x{info1.height}",
                            'duration1': f"{info1.duration:.1f}s",
                            'duration2': f"{info2.duration:.1f}s",
                            'size1': f"{info1.file_size / (1024 ** 2):.2f}MB",
                            'size2': f"{info2.file_size / (1024 ** 2):.2f}MB",
                            'similarity': f"{avg * 100:.1f}%"
                        }
                        return result

        # Partial match
        for shorter, longer in [(info2, info1), (info1, info2)]:
            if longer.duration > shorter.duration * 1.5:
                match = self.find_partial_match(shorter.frame_hashes, longer.frame_hashes)
                if match:
                    fps_ratio = longer.duration / max(len(longer.frame_hashes), 1)
                    result['match_type'] = 'partial_match'
                    result['confidence'] = match['similarity']
                    result['details'] = {
                        'full_video': longer.path,
                        'partial_video': shorter.path,
                        'full_duration': f"{longer.duration:.1f}s",
                        'partial_duration': f"{shorter.duration:.1f}s",
                        'match_position': f"{match['start_idx'] * fps_ratio:.1f}s ~ {match['end_idx'] * fps_ratio:.1f}s",
                        'similarity': f"{match['similarity'] * 100:.1f}%"
                    }
                    return result

        return None

    def find_duplicates(self, folder1, folder2, progress_callback=None):
        """Find all duplicates between two folders."""
        self.is_processing = True
        try:
            v1 = collect_videos(folder1)
            v2 = collect_videos(folder2)
            if not v1 or not v2:
                return [], len(v1), len(v2)

            def p1(done, total):
                if progress_callback:
                    progress_callback('analyze1', done, total)

            infos1 = self.analyze_videos(v1, p1)

            def p2(done, total):
                if progress_callback:
                    progress_callback('analyze2', done, total)

            infos2 = self.analyze_videos(v2, p2)

            # Compare pairs
            pairs = []
            for i1 in infos1:
                for i2 in infos2:
                    ratio = max(i1.file_size, i2.file_size) / max(min(i1.file_size, i2.file_size), 1)
                    if ratio < 10:
                        pairs.append((i1, i2))

            matches = []
            for idx, (i1, i2) in enumerate(pairs):
                if not self.is_processing:
                    break
                result = self.compare_videos(i1, i2)
                if result and result['match_type']:
                    matches.append(result)
                if progress_callback:
                    progress_callback('compare', idx + 1, len(pairs))

            return matches, len(v1), len(v2)
        finally:
            self.is_processing = False

    def find_duplicates_single_folder(self, folder, progress_callback=None):
        """Find duplicates within a single folder."""
        self.is_processing = True
        try:
            videos = collect_videos(folder)
            if len(videos) < 2:
                return [], len(videos)

            def prog(done, total):
                if progress_callback:
                    progress_callback('analyze', done, total)

            infos = self.analyze_videos(videos, prog)

            pairs = []
            for i in range(len(infos)):
                for j in range(i + 1, len(infos)):
                    ratio = max(infos[i].file_size, infos[j].file_size) / max(min(infos[i].file_size, infos[j].file_size), 1)
                    if ratio < 10:
                        pairs.append((infos[i], infos[j]))

            matches = []
            for idx, (i1, i2) in enumerate(pairs):
                if not self.is_processing:
                    break
                result = self.compare_videos(i1, i2)
                if result and result['match_type']:
                    matches.append(result)
                if progress_callback:
                    progress_callback('compare', idx + 1, len(pairs))

            return matches, len(videos)
        finally:
            self.is_processing = False

    def stop(self):
        self.is_processing = False
