"""DINOv2 feature extraction engine with TensorRT/PyTorch support."""

import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from utils.image_utils import preprocess_for_resnet, collect_images, collect_resolutions
from core.inference_engine import (
    build_dinov2_tensorrt, get_device, DINOV2_FEAT_DIM,
    TensorRTDINOv2, _safe_gpu_tag, _engine_path,
)
from core.cache_manager import CacheManager
from core.model_paths import MODEL_DIR

logger = logging.getLogger(__name__)


def _fmt_eta(seconds):
    """Format seconds to human-readable ETA string."""
    if seconds < 0:
        return ""
    m, s = divmod(int(seconds), 60)
    if m > 0:
        return f" (ETA {m}m {s:02d}s)"
    return f" (ETA {s}s)"


def _get_gpu_batch_size():
    """Determine optimal batch size based on GPU VRAM.
    Targets RTX 3060 Ti (8GB), RTX 4070 (12GB) class GPUs."""
    if not torch.cuda.is_available():
        return 32  # CPU fallback
    try:
        total = torch.cuda.get_device_properties(0).total_memory
        total_gb = total / (1024 ** 3)
        if total_gb >= 10:   # RTX 4070 (12GB), RTX 3080, etc.
            return 128
        elif total_gb >= 7:  # RTX 3060 Ti (8GB), RTX 4060, etc.
            return 96
        elif total_gb >= 5:  # RTX 3060 (6GB-ish after overhead)
            return 64
        else:
            return 32       # Low VRAM GPUs
    except Exception:
        return 32


class ResNetEngine:
    """DINOv2-Base feature extraction with TensorRT > PyTorch fallback.
    Uses single-process batched inference only (avoids CUDA multiprocessing issues on Windows).
    Class name kept as ResNetEngine for UI compatibility."""

    IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff', '*.webp']

    def __init__(self, batch_size=None):
        self.device = get_device()
        self.batch_size = batch_size or _get_gpu_batch_size()
        self.model = None
        self.trt_model = None
        self.use_tensorrt = False
        self.initialized = False
        self.is_processing = False
        self._scan_id = 0  # incremented each scan to prevent stale finally blocks
        self.cache = CacheManager()
        # Thread pool for CPU-bound image preprocessing (overlaps with GPU inference)
        self._preprocess_pool = ThreadPoolExecutor(max_workers=4)

    def initialize(self, progress_callback=None):
        """Initialize model. TensorRT first, PyTorch fallback."""
        try:
            trt = build_dinov2_tensorrt(progress_callback)
            if trt is not None:
                self.trt_model = trt
                self.use_tensorrt = True
                # TensorRT FP16 uses less VRAM per image, allow larger batches
                self.batch_size = max(self.batch_size, int(self.batch_size * 1.5))
                self.initialized = True
                logger.info(f"TensorRT FP16 ready, batch_size={self.batch_size}")
                return "TensorRT FP16"
        except Exception as e:
            logger.warning(f"TensorRT init failed: {e}")

        torch.hub.set_dir(MODEL_DIR)
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True)
        self.model = self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.initialized = True
        return "PyTorch"

    def backend_name(self):
        if self.use_tensorrt:
            return "TensorRT FP16"
        return "PyTorch"

    def extract_features(self, image_paths, progress_callback=None, status_callback=None):
        """Extract features for all images. Uses batch cache lookup.
        progress_callback: (done_batches, total_batches, extracted, total_imgs, eta_str)
        status_callback: (pct, msg) for general status updates"""
        if status_callback:
            status_callback(0, f"Checking cache... ({len(image_paths)} images)")
        features = self.cache.get_image_features_batch(image_paths, expected_dim=DINOV2_FEAT_DIM)
        uncached = [p for p in image_paths if p not in features]

        if not uncached:
            if status_callback:
                status_callback(0, f"All {len(image_paths)} images cached")
            return features

        if status_callback:
            status_callback(0, f"Cache: {len(features)}/{len(image_paths)}, extracting {len(uncached)} remaining...")

        new_feats = self._extract_batched(uncached, progress_callback)

        # Batch save to cache
        if new_feats:
            if status_callback:
                status_callback(0, f"Saving {len(new_feats)} features to cache...")
            self.cache.save_image_features_batch(list(new_feats.items()))

        features.update(new_feats)
        return features

    def _submit_preprocess(self, batch_paths):
        """Submit preprocessing tasks to thread pool (non-blocking).
        Returns list of (future, path) pairs."""
        return [(self._preprocess_pool.submit(preprocess_for_resnet, p), p)
                for p in batch_paths]

    def _collect_preprocess(self, future_pairs):
        """Collect preprocessing results from submitted futures.
        Returns (tensors, valid_paths) for GPU inference."""
        tensors, valid = [], []
        for future, p in future_pairs:
            try:
                t = future.result()
                if t is not None:
                    tensors.append(t)
                    valid.append(p)
            except Exception as e:
                logger.warning(f"Preprocess failed: {p} - {e}")
        return tensors, valid

    def _extract_batched(self, paths, progress_callback=None):
        """Pipelined batch extraction: threaded CPU preprocessing overlaps with GPU inference."""
        features = {}
        batches = [paths[i:i + self.batch_size] for i in range(0, len(paths), self.batch_size)]
        total = len(batches)
        batch_start = time.monotonic()

        # Submit first batch preprocessing immediately (all 4 threads work on it)
        pending_futures = self._submit_preprocess(batches[0]) if batches else []

        for idx, batch in enumerate(batches):
            if not self.is_processing:
                return features

            # Collect preprocessed data (futures were submitted in previous iteration or init)
            tensors, valid = self._collect_preprocess(pending_futures)

            # Submit next batch preprocessing while GPU works (pipeline overlap)
            if idx + 1 < total:
                pending_futures = self._submit_preprocess(batches[idx + 1])

            # GPU inference
            if tensors:
                bt = torch.stack(tensors).to(self.device)
                if self.use_tensorrt:
                    feats_np = self.trt_model(bt)
                else:
                    with torch.no_grad():
                        out = self.model.forward_features(bt)
                        feats = out["x_norm_clstoken"]
                    feats_np = feats.cpu().numpy()
                for i, p in enumerate(valid):
                    features[p] = feats_np[i].flatten()

            if progress_callback:
                done_batches = idx + 1
                elapsed = time.monotonic() - batch_start
                eta = (elapsed / done_batches * (total - done_batches)) if done_batches > 0 else 0
                progress_callback(done_batches, total, len(features), len(paths), _fmt_eta(eta))

            # Periodic VRAM cleanup (less frequent on RTX due to more headroom)
            if self.device.type == "cuda" and idx % 20 == 19:
                torch.cuda.empty_cache()

        return features

    @staticmethod
    def similarity_matrix_1d(features_dict):
        """Single folder: N x N similarity matrix."""
        paths = list(features_dict.keys())
        matrix = np.array([features_dict[p] for p in paths])
        return paths, cosine_similarity(matrix)

    @staticmethod
    def similarity_matrix_2d(feats1_list, feats2_list):
        """Two folders: M x N similarity matrix."""
        if not feats1_list or not feats2_list:
            return np.array([])
        return cosine_similarity(np.array(feats1_list), np.array(feats2_list))

    def find_duplicates_one_folder(self, folder, threshold=0.95,
                                    progress_callback=None, status_callback=None,
                                    recursive=False):
        """Find duplicates within a single folder."""
        if status_callback:
            status_callback(0, "Scanning folder...")
        images = collect_images(folder, recursive=recursive)
        if not images:
            return []

        if status_callback:
            status_callback(0, f"Found {len(images)} images")

        self._scan_id += 1
        my_scan = self._scan_id
        self.is_processing = True
        try:
            features = self.extract_features(images, progress_callback, status_callback)
            if not features:
                return []

            if status_callback:
                status_callback(0, "Collecting resolutions...")
            resolutions = collect_resolutions(list(features.keys()))
            if status_callback:
                status_callback(0, "Computing similarity matrix...")
            paths, sim_matrix = self.similarity_matrix_1d(features)
            effective = 0.999999 if threshold >= 1.0 else threshold

            duplicates = []
            for i in range(len(paths)):
                if not self.is_processing:
                    return duplicates
                for j in range(i + 1, len(paths)):
                    if sim_matrix[i][j] >= effective:
                        p1, p2 = paths[i], paths[j]
                        r1, r2 = resolutions.get(p1), resolutions.get(p2)
                        duplicates.append({
                            'img1_path': p1, 'img2_path': p2,
                            'img1_name': os.path.basename(p1),
                            'img2_name': os.path.basename(p2),
                            'similarity': float(sim_matrix[i][j]),
                            'same_resolution': r1 == r2,
                            'resolution1': r1, 'resolution2': r2,
                        })
            return duplicates
        finally:
            if self._scan_id == my_scan:
                self.is_processing = False

    def find_duplicates_two_folders(self, folder1, folder2, threshold=0.95,
                                     progress_callback=None, cancel_check=None,
                                     recursive=False):
        """Find duplicates between two folders."""
        if progress_callback:
            progress_callback(0, "Scanning folders...")
        imgs1 = collect_images(folder1, recursive=recursive)
        imgs2 = collect_images(folder2, recursive=recursive)
        if not imgs1 or not imgs2:
            return []

        if progress_callback:
            progress_callback(0, f"Found {len(imgs1)} + {len(imgs2)} images")

        self._scan_id += 1
        my_scan = self._scan_id
        self.is_processing = True
        try:
            def status1(pct, msg):
                if progress_callback:
                    progress_callback(pct, f"Folder 1: {msg}")

            def prog1(done, total, extracted, total_imgs, eta_str=""):
                if progress_callback:
                    pct = int(done / total * 35)
                    progress_callback(pct, f"Folder 1 features: {extracted}/{total_imgs}{eta_str}")

            feats1 = self.extract_features(imgs1, prog1, status1)
            if cancel_check and cancel_check():
                return []

            def status2(pct, msg):
                if progress_callback:
                    progress_callback(35 + pct, f"Folder 2: {msg}")

            def prog2(done, total, extracted, total_imgs, eta_str=""):
                if progress_callback:
                    pct = 35 + int(done / total * 30)
                    progress_callback(pct, f"Folder 2 features: {extracted}/{total_imgs}{eta_str}")

            feats2 = self.extract_features(imgs2, prog2, status2)
            if cancel_check and cancel_check():
                return []

            valid1 = [(p, feats1[p]) for p in imgs1 if p in feats1]
            valid2 = [(p, feats2[p]) for p in imgs2 if p in feats2]
            if not valid1 or not valid2:
                return []

            if progress_callback:
                progress_callback(70, "Collecting resolutions...")
            all_paths = [p for p, _ in valid1] + [p for p, _ in valid2]
            resolutions = collect_resolutions(all_paths)

            if progress_callback:
                progress_callback(75, "Computing similarity...")
            sim = self.similarity_matrix_2d(
                [f for _, f in valid1], [f for _, f in valid2])

            effective = 0.999999 if threshold >= 1.0 else threshold
            duplicates = []
            total_comp = len(valid1) * len(valid2)
            done = 0
            tc_start = time.monotonic()

            for i, (p1, _) in enumerate(valid1):
                if cancel_check and cancel_check():
                    break
                for j, (p2, _) in enumerate(valid2):
                    s = float(sim[i, j])
                    if s >= effective:
                        r1, r2 = resolutions.get(p1), resolutions.get(p2)
                        duplicates.append({
                            'img1_path': p1, 'img2_path': p2,
                            'img1_name': os.path.basename(p1),
                            'img2_name': os.path.basename(p2),
                            'similarity': s,
                            'same_resolution': r1 == r2,
                            'resolution1': r1, 'resolution2': r2,
                        })
                    done += 1
                    if progress_callback and done % 5000 == 0:
                        pct = 75 + int(done / total_comp * 25)
                        elapsed = time.monotonic() - tc_start
                        eta = (elapsed / done * (total_comp - done)) if done > 0 else 0
                        progress_callback(pct, f"Comparing: {done:,}/{total_comp:,}{_fmt_eta(eta)}")

            if progress_callback:
                progress_callback(100, f"Done: {len(duplicates)} pairs found")
            return duplicates
        finally:
            if self._scan_id == my_scan:
                self.is_processing = False

    def to_cpu(self):
        """Offload model to CPU to free GPU VRAM."""
        if not self.initialized or self.device.type == "cpu":
            return
        if self.use_tensorrt and self.trt_model is not None:
            self._had_trt = True
            gpu_tag = _safe_gpu_tag()
            if gpu_tag:
                self._trt_engine_file = _engine_path(gpu_tag)
            del self.trt_model
            self.trt_model = None
        if self.model is not None:
            self.model.cpu()
        torch.cuda.empty_cache()
        logger.info("DINOv2 offloaded to CPU")

    def to_gpu(self):
        """Move model back to GPU."""
        if not self.initialized or self.device.type == "cpu":
            return
        if self.model is not None:
            self.model.to(self.device)
        if getattr(self, '_had_trt', False) and self.trt_model is None:
            try:
                self.trt_model = TensorRTDINOv2(self._trt_engine_file)
            except Exception as e:
                logger.warning(f"DINOv2 TRT reload failed, falling back to PyTorch: {e}")
                self.use_tensorrt = False
                self._had_trt = False
        logger.info("DINOv2 loaded to GPU")

    def stop(self):
        self.is_processing = False
