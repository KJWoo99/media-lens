"""CLIP-based text-to-image search engine."""

import os
import time
import logging
import hashlib
from concurrent.futures import ThreadPoolExecutor
import torch
import numpy as np
from pathlib import Path
from PIL import Image, ImageFile

from core.translation import KoreanTranslator
from core.cache_manager import CacheManager
from core.model_paths import MODEL_DIR
from core.inference_engine import CACHE_DIR, _safe_gpu_tag, _get_trt_logger, _TRT_BUILD_LOCK, _build_trt_subprocess

logger = logging.getLogger(__name__)

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Register HEIC/HEIF support via pillow-heif if available
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


def _fmt_eta(seconds):
    if seconds <= 0:
        return ""
    m, s = divmod(int(seconds), 60)
    return f" (ETA {m}m {s:02d}s)" if m > 0 else f" (ETA {s}s)"

DEFAULT_MODEL = "apple/DFN5B-CLIP-ViT-H-14-378"
IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tif', '.tiff',
    '.heic', '.heif', '.avif',
}

# DFN5B-CLIP-ViT-H-14-378: projection_dim=1024, image_size=378
CLIP_FEAT_DIM = 1024
CLIP_IMG_SIZE = 378


# ── TensorRT CLIP image encoder ────────────────────────────────────────────

class _CLIPVisionWrapper(torch.nn.Module):
    """Wraps CLIP vision_model + visual_projection for ONNX export."""

    def __init__(self, vision_model, visual_projection):
        super().__init__()
        self.vision_model = vision_model
        self.visual_projection = visual_projection

    def forward(self, pixel_values):
        outputs = self.vision_model(pixel_values=pixel_values)
        feats = outputs.pooler_output
        feats = self.visual_projection(feats)
        return feats / feats.norm(dim=-1, keepdim=True)


def _trt_clip_engine_path(gpu_tag):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"clip_dfn5b_vith_fp16_{gpu_tag}.engine")


def _trt_clip_onnx_path():
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, "clip_dfn5b_vith_image_encoder.onnx")


def _export_clip_onnx(onnx_file, model):
    """Export CLIP vision encoder + projection to ONNX (CPU).

    Loads a dedicated eager-attention CLIPVisionModel for export because SDPA
    (default in transformers 5.x + PyTorch 2.1.1+) is not exportable to ONNX
    opset 14-19. visual_projection is a plain linear layer with no attention —
    it is reused from the production model (already on CPU at call time).
    dynamo=False forces the legacy TorchScript exporter (PyTorch 2.9+ defaults
    to dynamo=True).
    """
    from transformers import CLIPVisionModel
    try:
        vision_export = CLIPVisionModel.from_pretrained(
            DEFAULT_MODEL,
            cache_dir=MODEL_DIR, local_files_only=True,
            attn_implementation="eager",
            output_hidden_states=False,
            output_attentions=False,
        ).eval()
    except Exception:
        vision_export = CLIPVisionModel.from_pretrained(
            DEFAULT_MODEL,
            cache_dir=MODEL_DIR,
            attn_implementation="eager",
            output_hidden_states=False,
            output_attentions=False,
        ).eval()
    try:
        # visual_projection has no attention; reuse from production model (on CPU)
        proj_cpu = model.visual_projection
        wrapper = _CLIPVisionWrapper(vision_export, proj_cpu)
        wrapper.eval()
        dummy = torch.randn(1, 3, CLIP_IMG_SIZE, CLIP_IMG_SIZE)
        with torch.no_grad():
            torch.onnx.export(
                wrapper, dummy, onnx_file,
                input_names=["pixel_values"],
                output_names=["features"],
                dynamic_axes={"pixel_values": {0: "batch"}, "features": {0: "batch"}},
                opset_version=17,
                dynamo=False,
            )
        logger.info(f"CLIP ONNX exported → {onnx_file}")
    finally:
        del vision_export


def _build_clip_trt(onnx_file, engine_file):
    import tensorrt as trt
    trt_logger = _get_trt_logger()
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)
    if not parser.parse_from_file(str(onnx_file)):
        for i in range(parser.num_errors):
            logger.error(str(parser.get_error(i)))
        raise RuntimeError("ONNX parsing failed for CLIP")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    profile.set_shape("pixel_values",
                      (1, 3, CLIP_IMG_SIZE, CLIP_IMG_SIZE),
                      (8, 3, CLIP_IMG_SIZE, CLIP_IMG_SIZE),
                      (48, 3, CLIP_IMG_SIZE, CLIP_IMG_SIZE))
    config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed for CLIP")
    with open(engine_file, "wb") as f:
        f.write(serialized)
    logger.info(f"CLIP TRT engine built")


def _build_clip_all(onnx_file, engine_file):
    """Load CLIP model, export ONNX (if needed), build TRT engine.

    Designed to run entirely inside a subprocess so that TracerWarnings and
    any native TRT crash are isolated from the main application process.
    """
    if not os.path.exists(onnx_file):
        from transformers import CLIPModel
        try:
            model = CLIPModel.from_pretrained(
                DEFAULT_MODEL, cache_dir=MODEL_DIR, local_files_only=True,
                output_hidden_states=False, output_attentions=False,
            ).eval()
        except OSError:
            model = CLIPModel.from_pretrained(
                DEFAULT_MODEL, cache_dir=MODEL_DIR,
                output_hidden_states=False, output_attentions=False,
            ).eval()
        _export_clip_onnx(onnx_file, model)
        del model
        import gc; gc.collect()
    _build_clip_trt(onnx_file, engine_file)


class TensorRTCLIPVision:
    """TensorRT FP16 CLIP image encoder wrapper."""

    def __init__(self, engine_file):
        import tensorrt as trt
        runtime = trt.Runtime(_get_trt_logger())
        with open(engine_file, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self._stream = torch.cuda.Stream()
        # Query max batch from engine profile; fallback for older engines
        try:
            self._max_batch = self.engine.get_tensor_profile_shape("pixel_values", 0)[2][0]
        except Exception:
            self._max_batch = 24

    def __call__(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values: CUDA float32 (N,3,378,378) → cpu tensor (N,1024)
        Always runs at max_batch size (padding) so Myelin graph stays loaded."""
        n = pixel_values.shape[0]
        dev = pixel_values.device

        if n > self._max_batch:
            chunks = [self._infer_padded(pixel_values[i:i + self._max_batch], dev)
                      for i in range(0, n, self._max_batch)]
            return torch.cat(chunks, dim=0)

        return self._infer_padded(pixel_values, dev)

    def _infer_padded(self, chunk: torch.Tensor, dev) -> torch.Tensor:
        """Pad chunk to max_batch, infer, return only the real rows."""
        real_n = chunk.shape[0]
        if real_n < self._max_batch:
            pad = torch.zeros(self._max_batch - real_n, 3, CLIP_IMG_SIZE, CLIP_IMG_SIZE,
                              dtype=chunk.dtype, device=dev)
            chunk = torch.cat([chunk, pad], dim=0)
        chunk = chunk.contiguous()
        output = torch.empty((self._max_batch, CLIP_FEAT_DIM), dtype=torch.float32, device=dev)
        self.context.set_input_shape("pixel_values", (self._max_batch, 3, CLIP_IMG_SIZE, CLIP_IMG_SIZE))
        self.context.set_tensor_address("pixel_values", chunk.data_ptr())
        self.context.set_tensor_address("features", output.data_ptr())
        self.context.execute_async_v3(stream_handle=self._stream.cuda_stream)
        self._stream.synchronize()
        return output.cpu()[:real_n]


def _get_clip_batch_size():
    """Determine CLIP batch size based on GPU VRAM.
    ViT-H-14-378 is larger than ResNet50, so batches are smaller.
    Targets RTX 3060 Ti (8GB), RTX 4070 (12GB)."""
    if not torch.cuda.is_available():
        return 4  # CPU
    try:
        total = torch.cuda.get_device_properties(0).total_memory
        total_gb = total / (1024 ** 3)
        if total_gb >= 10:   # 12GB+ (RTX 4070, 3080, etc.)
            return 24
        elif total_gb >= 7:  # 8GB (RTX 3060 Ti, 4060, etc.)
            return 16
        elif total_gb >= 5:  # 6GB
            return 8
        else:
            return 4         # Low VRAM
    except Exception:
        return 4


class CLIPEngine:
    """CLIP model for text-image semantic search with caching."""

    def __init__(self, model_name=DEFAULT_MODEL):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model_hash = hashlib.md5(model_name.encode()).hexdigest()[:8]
        self.model = None
        self.processor = None
        self.trt_vision = None
        self.translator = KoreanTranslator()
        self.cache = CacheManager()
        self.initialized = False
        self.batch_size = _get_clip_batch_size()
        self.is_processing = False
        self._load_id = 0

        # In-memory cache for current session
        self._mem_cache = {}
        # Thread pool for CPU-bound image loading
        self._load_pool = ThreadPoolExecutor(max_workers=4)

    def initialize(self, progress_callback=None):
        """Load CLIP model."""
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        logging.getLogger("transformers").setLevel(logging.ERROR)

        if progress_callback:
            progress_callback("Loading CLIP model...")

        from transformers import CLIPProcessor, CLIPModel

        if self.device == "cuda":
            torch.cuda.empty_cache()

        # Try local models/ dir first, download only on first run
        try:
            self.model = CLIPModel.from_pretrained(
                self.model_name,
                output_hidden_states=False,
                output_attentions=False,
                cache_dir=MODEL_DIR,
                local_files_only=True
            ).to("cpu").eval()
            self.processor = CLIPProcessor.from_pretrained(
                self.model_name, cache_dir=MODEL_DIR, local_files_only=True)
        except OSError:
            self.model = CLIPModel.from_pretrained(
                self.model_name,
                output_hidden_states=False,
                output_attentions=False,
                cache_dir=MODEL_DIR
            ).to("cpu").eval()
            self.processor = CLIPProcessor.from_pretrained(
                self.model_name, cache_dir=MODEL_DIR)

        for p in self.model.parameters():
            p.requires_grad = False

        trt_ok = self._init_trt(progress_callback)
        if trt_ok:
            self.batch_size = max(self.batch_size, int(self.batch_size * 1.5))

        self.initialized = True
        backend = "TRT FP16" if trt_ok else "PyTorch"
        if progress_callback:
            progress_callback(f"CLIP ready ({self.device}, {backend})")
        logger.info(f"CLIP ready ({self.device}, {backend}), batch_size={self.batch_size}")
        return True

    def _init_trt(self, progress_callback=None):
        try:
            import tensorrt  # noqa
        except ImportError:
            return False

        gpu_tag = _safe_gpu_tag()
        if gpu_tag is None:
            return False

        engine_file = _trt_clip_engine_path(gpu_tag)
        onnx_file = _trt_clip_onnx_path()

        # Fast path: load already-built engine (no lock needed)
        if os.path.exists(engine_file):
            if progress_callback:
                progress_callback("Loading CLIP TRT engine...")
            try:
                self.trt_vision = TensorRTCLIPVision(engine_file)
                self._trt_engine_file = engine_file
                return True
            except Exception as e:
                logger.warning(f"CLIP TRT load failed: {e}")
                os.remove(engine_file)

        # Build path: serialize with global lock (one TRT build at a time)
        with _TRT_BUILD_LOCK:
            # Re-check after acquiring lock
            if os.path.exists(engine_file):
                try:
                    self.trt_vision = TensorRTCLIPVision(engine_file)
                    self._trt_engine_file = engine_file
                    return True
                except Exception:
                    os.remove(engine_file)

            try:
                if progress_callback:
                    progress_callback(f"Building CLIP TRT engine ({gpu_tag})... first time only (3-5 min)")
                # ONNX export + TRT build both run in subprocess — keeps main process safe
                if not _build_trt_subprocess(onnx_file, engine_file,
                                             'core.clip_engine', '_build_clip_all'):
                    raise RuntimeError("CLIP TRT build failed in subprocess")
                self.trt_vision = TensorRTCLIPVision(engine_file)
                self._trt_engine_file = engine_file
                return True
            except Exception as e:
                logger.warning(f"CLIP TRT failed, falling back to PyTorch: {e}")
                # Clean up partial ONNX so next run retries from scratch
                if os.path.exists(onnx_file):
                    try:
                        os.remove(onnx_file)
                    except Exception:
                        pass
                return False

    def _encode_trt(self, images):
        """Encode a list of PIL images via TRT. Returns cpu tensor (N, 1024)."""
        pixel_values = self.processor(images=images, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(self.device).float()
        return self.trt_vision(pixel_values)

    def _encode_pytorch(self, images):
        """Encode a list of PIL images via PyTorch. Returns cpu tensor (N, feat_dim)."""
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.vision_model(**inputs)
            feats = outputs.pooler_output if outputs.pooler_output is not None else outputs.last_hidden_state.mean(dim=1)
            if hasattr(self.model, 'visual_projection'):
                feats = self.model.visual_projection(feats)
            feats = feats / torch.norm(feats, dim=-1, keepdim=True)
        return feats.cpu()

    def _compute_image_features(self, image):
        """Compute features for a single PIL image."""
        w, h = image.size
        aspect = w / h

        if aspect > 3.0:
            return self._process_panorama(image, aspect)

        if max(w, h) > 1024:
            ratio = 1024 / max(w, h)
            image = image.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

        if self.trt_vision is not None:
            return self._encode_trt([image])
        return self._encode_pytorch([image])

    def _process_panorama(self, image, aspect_ratio):
        """Handle ultra-wide panorama images by segmenting.
        Returns all segment embeddings (n_seg, dim) for max-similarity matching."""
        w, h = image.size
        n_seg = max(2, int(aspect_ratio // 1.5))
        seg_w = w // n_seg
        all_feats = []

        for i in range(n_seg):
            left = i * seg_w
            right = w if i == n_seg - 1 else left + seg_w
            seg = image.crop((left, 0, right, h))
            if self.trt_vision is not None:
                feats = self._encode_trt([seg])
            else:
                feats = self._encode_pytorch([seg])
            all_feats.append(feats)
            if self.device == "cuda" and i % 3 == 2:
                torch.cuda.empty_cache()

        return torch.cat(all_feats, dim=0)  # (n_seg, dim)

    def compute_text_features(self, text):
        """Compute text embedding, auto-translating Korean."""
        translated = self.translator.translate(text)
        inputs = self.processor(text=[translated], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.text_model(**inputs)
            feats = outputs.pooler_output if outputs.pooler_output is not None else outputs.last_hidden_state[:, -1, :]
            if hasattr(self.model, 'text_projection'):
                feats = self.model.text_projection(feats)
            feats = feats / torch.norm(feats, dim=-1, keepdim=True)
        return feats

    def _get_embed_dim(self):
        """Get the CLIP embedding dimension from the model."""
        if hasattr(self.model, 'visual_projection'):
            return self.model.visual_projection.out_features
        return self.model.config.projection_dim

    def get_image_embedding(self, path):
        """Get embedding for a single image (cache-aware).
        Returns (1, dim) for normal images, (n_seg, dim) for panoramas."""
        path_str = str(path)

        # Memory cache
        if path_str in self._mem_cache:
            return self._mem_cache[path_str]

        # Disk cache
        cached = self.cache.get_clip_embedding(path_str, self.model_hash)
        if cached is not None:
            dim = self._get_embed_dim()
            tensor = torch.from_numpy(cached).reshape(-1, dim)
            self._mem_cache[path_str] = tensor
            return tensor

        # Compute
        try:
            img = Image.open(path_str).convert("RGB")
            feats = self._compute_image_features(img)
            self._mem_cache[path_str] = feats
            self.cache.save_clip_embedding(path_str, self.model_hash, feats.numpy().flatten())
            return feats
        except Exception as e:
            logger.warning(f"[CLIP] Failed: {path_str} - {e}")
            return None

    @staticmethod
    def _load_and_resize(path_str):
        """Load and resize a single image (CPU-bound, thread-safe)."""
        try:
            img = Image.open(path_str).convert("RGB")
            w, h = img.size
            if max(w, h) > 1024:
                ratio = 1024 / max(w, h)
                img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
            return path_str, img
        except Exception as e:
            return path_str, None

    def get_image_embeddings_batch(self, paths):
        """Batch compute embeddings for multiple images (GPU-efficient).
        Uses threaded image loading and VRAM-aware batch sizes."""
        results = {}
        to_load_paths = []

        # Check memory cache first
        mem_miss = []
        for path_str in paths:
            if path_str in self._mem_cache:
                results[path_str] = self._mem_cache[path_str]
            else:
                mem_miss.append(path_str)

        # Batch disk cache lookup for memory misses
        if mem_miss:
            dim = self._get_embed_dim()
            disk_hits = self.cache.get_clip_embeddings_batch(mem_miss, self.model_hash)
            for path_str, cached in disk_hits.items():
                tensor = torch.from_numpy(cached).reshape(-1, dim)
                self._mem_cache[path_str] = tensor
                results[path_str] = tensor

        to_load_paths = [p for p in mem_miss if p not in results]

        if not to_load_paths:
            return results

        # Threaded image loading (overlap I/O with processing)
        loaded = list(self._load_pool.map(self._load_and_resize, to_load_paths))
        to_compute = [(p, img) for p, img in loaded if img is not None]
        for p, img in loaded:
            if img is None:
                logger.warning(f"[CLIP] Failed to load: {p}")

        # Batch inference with dynamic batch size
        for i in range(0, len(to_compute), self.batch_size):
            batch = to_compute[i:i + self.batch_size]
            batch_paths = [p for p, _ in batch]
            batch_imgs = [img for _, img in batch]
            try:
                if self.trt_vision is not None:
                    feats_cpu = self._encode_trt(batch_imgs)
                else:
                    feats_cpu = self._encode_pytorch(batch_imgs)
                to_save = []
                for j, path_str in enumerate(batch_paths):
                    single_feat = feats_cpu[j:j+1]
                    self._mem_cache[path_str] = single_feat
                    to_save.append((path_str, single_feat.numpy().flatten()))
                    results[path_str] = single_feat
                self.cache.save_clip_embeddings_batch(to_save, self.model_hash)
            except torch.cuda.OutOfMemoryError:
                # OOM: halve batch size and retry this batch
                logger.warning(f"[CLIP] OOM at batch_size={self.batch_size}, halving")
                torch.cuda.empty_cache()
                self.batch_size = max(2, self.batch_size // 2)
                for p_str in batch_paths:
                    emb = self.get_image_embedding(p_str)
                    if emb is not None:
                        results[p_str] = emb
            except Exception as e:
                logger.warning(f"[CLIP] Batch failed, falling back to single: {e}")
                for path_str in batch_paths:
                    emb = self.get_image_embedding(path_str)
                    if emb is not None:
                        results[path_str] = emb

            # Only clean VRAM periodically, not every batch (RTX has headroom)
            if self.device == "cuda" and i % (self.batch_size * 4) == 0 and i > 0:
                torch.cuda.empty_cache()

        return results

    def process_folder(self, folder, status_callback=None, recursive=False):
        """Pre-compute embeddings for all images in folder.
        status_callback(pct, msg) reports progress and ETA."""
        if status_callback:
            status_callback(0, "Scanning folder...")

        folder = Path(folder)
        if recursive:
            files = [f for f in folder.rglob("*")
                     if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
        else:
            files = [f for f in folder.iterdir()
                     if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]

        if not files:
            return 0

        if status_callback:
            status_callback(0, f"Found {len(files)} images")

        self._load_id += 1
        my_id = self._load_id
        self.is_processing = True

        try:
            # Cache check: memory first, then disk batch
            if status_callback:
                status_callback(0, f"Checking cache... ({len(files)} images)")

            all_paths = [str(f) for f in files]
            mem_miss = [p for p in all_paths if p not in self._mem_cache]

            if mem_miss:
                dim = self._get_embed_dim()
                disk_hits = self.cache.get_clip_embeddings_batch(mem_miss, self.model_hash)
                for path_str, cached in disk_hits.items():
                    self._mem_cache[path_str] = torch.from_numpy(cached).reshape(-1, dim)

            uncached = [p for p in all_paths if p not in self._mem_cache]
            cached_count = len(all_paths) - len(uncached)

            if not uncached:
                if status_callback:
                    status_callback(100, f"All {len(files)} images cached")
                return len(files)

            if status_callback:
                status_callback(0, f"Cache: {cached_count}/{len(all_paths)}, extracting {len(uncached)} remaining...")

            total = len(uncached)
            processed = 0
            batch_start = time.monotonic()

            # Show initial status once before first batch starts
            if status_callback:
                status_callback(0, f"Extracting: 0/{total}...")

            for i in range(0, total, self.batch_size):
                if not self.is_processing:
                    break

                batch = uncached[i:i + self.batch_size]
                self.get_image_embeddings_batch(batch)
                processed += len(batch)

                if status_callback:
                    elapsed = time.monotonic() - batch_start
                    eta = (elapsed / processed * (total - processed)) if processed > 0 else 0
                    pct = int(processed / total * 100)
                    status_callback(pct, f"Extracting: {processed}/{total}{_fmt_eta(eta)}")

            return len(files)
        finally:
            if self._load_id == my_id:
                self.is_processing = False

    def stop(self):
        self.is_processing = False

    def search(self, query, folder, top_k=10, recursive=False):
        """Search images by text query. Returns [(path, score), ...].
        Panorama images use max similarity across segments."""
        folder = Path(folder)
        if recursive:
            files = [f for f in folder.rglob("*")
                     if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
        else:
            files = [f for f in folder.iterdir()
                     if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]

        if not files:
            return []

        text_feats = self.compute_text_features(query)

        results = []
        for f in files:
            emb = self.get_image_embedding(str(f))
            if emb is not None:
                emb_dev = emb.to(self.device)
                sim = torch.matmul(text_feats, emb_dev.T).squeeze(0)
                max_sim = sim.max().item() if sim.dim() > 0 else sim.item()
                results.append((str(f), max_sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def search_by_image(self, query_image_path, folder, top_k=10, recursive=False):
        """Search images by a query image. Returns [(path, score), ...]."""
        folder = Path(folder)
        if recursive:
            files = [f for f in folder.rglob("*")
                     if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
        else:
            files = [f for f in folder.iterdir()
                     if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]

        if not files:
            return []

        query_emb = self.get_image_embedding(str(query_image_path))
        if query_emb is None:
            raise ValueError(f"Failed to load query image: {query_image_path}")

        query_emb_dev = query_emb.to(self.device)  # (n_seg, dim)
        query_path_str = str(query_image_path)

        results = []
        for f in files:
            path_str = str(f)
            if path_str == query_path_str:
                continue
            emb = self.get_image_embedding(path_str)
            if emb is not None:
                emb_dev = emb.to(self.device)
                sim = torch.matmul(query_emb_dev, emb_dev.T)
                max_sim = sim.max().item()
                results.append((path_str, max_sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def to_cpu(self):
        """Offload model to CPU to free GPU VRAM."""
        if not self.initialized or self.device == "cpu":
            return
        if self.model is not None:
            self.model.cpu()
        if self.trt_vision is not None:
            self._had_trt = True
            # _trt_engine_file was saved in _init_trt — no need to recompute
            del self.trt_vision
            self.trt_vision = None
        # Offload KoreanTranslator to free its VRAM
        if self.translator.model is not None and self.translator.device == "cuda":
            self.translator.model.cpu()
        torch.cuda.empty_cache()
        logger.info("CLIP offloaded to CPU")

    def to_gpu(self):
        """Move model back to GPU."""
        if not self.initialized or self.device == "cpu":
            return
        if self.model is not None:
            self.model.to(self.device)
        if getattr(self, '_had_trt', False) and self.trt_vision is None:
            try:
                self.trt_vision = TensorRTCLIPVision(self._trt_engine_file)
            except Exception as e:
                logger.warning(f"CLIP TRT reload failed: {e}")
                self._had_trt = False
        # Reload KoreanTranslator to GPU
        if self.translator.model is not None and self.translator.device == "cuda":
            self.translator.model.to(self.translator.device)
        logger.info("CLIP loaded to GPU")

    def clear_memory_cache(self):
        self._mem_cache.clear()
        if self.device == "cuda":
            torch.cuda.empty_cache()
