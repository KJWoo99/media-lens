"""SigLIP2-based text-to-image search engine.

Image encoder: TensorRT FP16 (falls back to PyTorch if unavailable)
Text encoder:  PyTorch — Korean/English supported natively (no translation needed)
Cache:         Shared clip_cache table, unique model_hash key
"""

import os
import time
import logging
import hashlib
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
from pathlib import Path
from PIL import Image, ImageFile

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

MODEL_ID   = "google/siglip2-so400m-patch14-384"
MODEL_HASH = hashlib.md5(MODEL_ID.encode()).hexdigest()[:8]
FEAT_DIM   = 1152
IMG_SIZE   = 384

# transformers 5.x bug: TOKENIZER_MAPPING_NAMES['siglip'] is None, causing
# AutoTokenizer to call None.replace() when resolving the tokenizer class.
# SigLIP2 uses GemmaTokenizerFast — patch the registry at import time.
# Note: transformers 5.2.0+ uses plain strings (not tuples) as dict values.
try:
    from transformers.models.auto import tokenization_auto as _ta
    val = _ta.TOKENIZER_MAPPING_NAMES.get("siglip")
    if val is None or isinstance(val, tuple):
        _ta.TOKENIZER_MAPPING_NAMES["siglip"] = "GemmaTokenizerFast"
except Exception:
    pass
IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tif', '.tiff',
    '.heic', '.heif', '.avif',
}


def _fmt_eta(seconds):
    if seconds <= 0:
        return ""
    m, s = divmod(int(seconds), 60)
    return f" (ETA {m}m {s:02d}s)" if m > 0 else f" (ETA {s}s)"


def _get_batch_size(trt=False):
    if not torch.cuda.is_available():
        return 8
    try:
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if trt:
            if total_gb >= 10: return 48
            elif total_gb >= 7: return 32
            elif total_gb >= 5: return 16
            else:               return 8
        else:
            if total_gb >= 10: return 24
            elif total_gb >= 7: return 16
            elif total_gb >= 5: return 8
            else:               return 4
    except Exception:
        return 16 if trt else 8


# ── TensorRT image encoder ─────────────────────────────────────────────────

def _trt_engine_path(gpu_tag):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"siglip2_so400m_fp16_{gpu_tag}.engine")


def _trt_onnx_path():
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, "siglip2_so400m_image_encoder.onnx")


class _SigLIP2VisionWrapper(torch.nn.Module):
    """Extracts pooler_output from SigLIP2 vision model for ONNX export."""

    def __init__(self, vision_model):
        super().__init__()
        self.vision_model = vision_model

    def forward(self, pixel_values):
        outputs = self.vision_model(pixel_values=pixel_values)
        return outputs.pooler_output  # (B, 1152)


def _export_siglip2_onnx(onnx_file):
    """Export SigLIP2 vision encoder to ONNX (CPU).

    Loads a dedicated eager-attention model for export because:
      1. attn_implementation must be set at load time (not changeable after).
      2. SigLIP2 uses SDPA (F.scaled_dot_product_attention) by default in
         transformers 5.x + PyTorch 2.1.1+.
      3. SDPA is not exportable to ONNX opset 14-19.
      4. PyTorch 2.9+ defaults torch.onnx.export to dynamo=True; dynamo=False
         forces the stable legacy TorchScript exporter.
    """
    from transformers.models.siglip.modeling_siglip import SiglipVisionModel
    try:
        vision_export = SiglipVisionModel.from_pretrained(
            MODEL_ID, cache_dir=MODEL_DIR, local_files_only=True,
            attn_implementation="eager",
        ).eval()
    except Exception:
        vision_export = SiglipVisionModel.from_pretrained(
            MODEL_ID, cache_dir=MODEL_DIR,
            attn_implementation="eager",
        ).eval()
    try:
        wrapper = _SigLIP2VisionWrapper(vision_export)
        wrapper.eval()
        dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        with torch.no_grad():
            torch.onnx.export(
                wrapper, dummy, onnx_file,
                input_names=["pixel_values"],
                output_names=["pooler_output"],
                dynamic_axes={"pixel_values": {0: "batch"}, "pooler_output": {0: "batch"}},
                opset_version=17,
                dynamo=False,
            )
        logger.info(f"SigLIP2 ONNX exported → {onnx_file}")
    finally:
        del vision_export


def _build_siglip2_trt(onnx_file, engine_file):
    import tensorrt as trt
    trt_logger = _get_trt_logger()
    builder  = trt.Builder(trt_logger)
    network  = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser   = trt.OnnxParser(network, trt_logger)
    if not parser.parse_from_file(str(onnx_file)):
        for i in range(parser.num_errors):
            logger.error(str(parser.get_error(i)))
        raise RuntimeError("ONNX parsing failed for SigLIP2")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    profile.set_shape("pixel_values", (1, 3, IMG_SIZE, IMG_SIZE), (32, 3, IMG_SIZE, IMG_SIZE), (64, 3, IMG_SIZE, IMG_SIZE))
    config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed for SigLIP2")
    with open(engine_file, "wb") as f:
        f.write(serialized)
    logger.info(f"SigLIP2 TRT engine built")


def _build_siglip2_all(onnx_file, engine_file):
    """Export SigLIP2 ONNX (if needed) + build TRT engine.

    Designed to run entirely inside a subprocess so that TracerWarnings and
    any native TRT crash are isolated from the main application process.
    """
    if not os.path.exists(onnx_file):
        _export_siglip2_onnx(onnx_file)
    _build_siglip2_trt(onnx_file, engine_file)


class TensorRTSigLIP2Vision:
    """TensorRT FP16 SigLIP2 image encoder wrapper."""

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
            self._max_batch = 96

    def __call__(self, pixel_values: torch.Tensor) -> np.ndarray:
        """pixel_values: CUDA float32 (N,3,384,384) → numpy (N,1152)
        Always runs at max_batch size (padding) so Myelin graph stays loaded."""
        n = pixel_values.shape[0]
        dev = pixel_values.device

        if n > self._max_batch:
            chunks = [self._infer_padded(pixel_values[i:i + self._max_batch], dev)
                      for i in range(0, n, self._max_batch)]
            return np.concatenate(chunks, axis=0)

        return self._infer_padded(pixel_values, dev)

    def _infer_padded(self, chunk: torch.Tensor, dev) -> np.ndarray:
        """Pad chunk to max_batch, infer, return only the real rows."""
        real_n = chunk.shape[0]
        if real_n < self._max_batch:
            pad = torch.zeros(self._max_batch - real_n, 3, IMG_SIZE, IMG_SIZE,
                              dtype=chunk.dtype, device=dev)
            chunk = torch.cat([chunk, pad], dim=0)
        chunk = chunk.contiguous()
        output = torch.empty((self._max_batch, FEAT_DIM), dtype=torch.float32, device=dev)
        self.context.set_input_shape("pixel_values", (self._max_batch, 3, IMG_SIZE, IMG_SIZE))
        self.context.set_tensor_address("pixel_values", chunk.data_ptr())
        self.context.set_tensor_address("pooler_output", output.data_ptr())
        self.context.execute_async_v3(stream_handle=self._stream.cuda_stream)
        self._stream.synchronize()
        return output.cpu().numpy()[:real_n]


# ── Main engine ────────────────────────────────────────────────────────────

class SigLIP2Engine:
    """SigLIP2 text-to-image search engine.

    Native multilingual support — no translation model required.
    Image encoder runs as TensorRT FP16 when available.
    """

    def __init__(self):
        self.device      = "cuda" if torch.cuda.is_available() else "cpu"
        self.model       = None
        self.processor   = None
        self.trt_vision  = None       # TRT image encoder (None = PyTorch fallback)
        self.cache       = CacheManager()
        self.model_hash  = MODEL_HASH
        self.initialized = False
        self.is_processing = False
        self.backend_info  = ""
        self.batch_size    = _get_batch_size(trt=False)
        self._load_id      = 0
        self._mem_cache    = {}
        self._load_pool    = ThreadPoolExecutor(max_workers=4)

    # ── Initialization ──────────────────────────────────────────────────

    def initialize(self, progress_callback=None):
        """Load model and build TRT engine. Sets self.backend_info."""
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        logging.getLogger("transformers").setLevel(logging.ERROR)

        if progress_callback:
            progress_callback("Loading SigLIP2 model...")

        from transformers.models.siglip.modeling_siglip import SiglipModel
        from transformers.models.siglip.image_processing_siglip import SiglipImageProcessor
        from transformers.models.siglip.processing_siglip import SiglipProcessor
        from transformers import AutoTokenizer

        def _build_processor(**kwargs):
            """Build SiglipProcessor with AutoTokenizer (correctly loads GemmaTokenizer).
            AutoProcessor routes through SiglipProcessor which hardcodes SiglipTokenizer —
            that class requires a SentencePiece vocab file absent from SigLIP2 caches."""
            img_proc  = SiglipImageProcessor.from_pretrained(MODEL_ID, **kwargs)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **kwargs)
            return SiglipProcessor(image_processor=img_proc, tokenizer=tokenizer)

        if self.device == "cuda":
            torch.cuda.empty_cache()

        try:
            self.model = SiglipModel.from_pretrained(
                MODEL_ID, cache_dir=MODEL_DIR, local_files_only=True
            ).to("cpu").eval()
            self.processor = _build_processor(cache_dir=MODEL_DIR, local_files_only=True)
        except (OSError, AttributeError, TypeError, ValueError):
            if progress_callback:
                progress_callback("Downloading SigLIP2 model (~900 MB)...")
            self.model = SiglipModel.from_pretrained(
                MODEL_ID, cache_dir=MODEL_DIR
            ).to("cpu").eval()
            self.processor = _build_processor(cache_dir=MODEL_DIR)

        for p in self.model.parameters():
            p.requires_grad = False

        # Build / load TRT image encoder
        trt_ok = self._init_trt(progress_callback)
        self.batch_size = _get_batch_size(trt=trt_ok)

        backend = "TRT FP16" if trt_ok else "PyTorch"
        self.backend_info = f"SigLIP2 ready ({self.device}, {backend})"
        self.initialized = True
        logger.info(f"SigLIP2 ready ({self.device}, {backend}), batch_size={self.batch_size}")

    def _init_trt(self, progress_callback=None):
        try:
            import tensorrt  # noqa
        except ImportError:
            return False

        gpu_tag = _safe_gpu_tag()
        if gpu_tag is None:
            return False

        engine_file = _trt_engine_path(gpu_tag)
        onnx_file   = _trt_onnx_path()

        # Fast path: load already-built engine (no lock needed)
        if os.path.exists(engine_file):
            if progress_callback:
                progress_callback("Loading SigLIP2 TRT engine...")
            try:
                self.trt_vision = TensorRTSigLIP2Vision(engine_file)
                self._trt_engine_file = engine_file
                return True
            except Exception as e:
                logger.warning(f"SigLIP2 TRT load failed: {e}")
                os.remove(engine_file)

        # Build path: serialize with global lock (one TRT build at a time)
        with _TRT_BUILD_LOCK:
            # Re-check after acquiring lock
            if os.path.exists(engine_file):
                try:
                    self.trt_vision = TensorRTSigLIP2Vision(engine_file)
                    self._trt_engine_file = engine_file
                    return True
                except Exception:
                    os.remove(engine_file)

            try:
                if progress_callback:
                    progress_callback(
                        f"Building SigLIP2 TRT engine ({gpu_tag})... first time only (1-3 min)")
                # ONNX export + TRT build both run in subprocess — keeps main process safe
                if not _build_trt_subprocess(onnx_file, engine_file,
                                             'core.siglip2_engine', '_build_siglip2_all'):
                    raise RuntimeError("SigLIP2 TRT build failed in subprocess")
                self.trt_vision = TensorRTSigLIP2Vision(engine_file)
                self._trt_engine_file = engine_file
                return True
            except Exception as e:
                logger.warning(f"SigLIP2 TRT failed, falling back to PyTorch: {e}")
                # Clean up partial ONNX so next run retries from scratch
                if os.path.exists(onnx_file):
                    try:
                        os.remove(onnx_file)
                    except Exception:
                        pass
            return False

    # ── Encoding ────────────────────────────────────────────────────────

    def compute_text_features(self, text: str) -> torch.Tensor:
        """Compute normalized text embedding (Korean/English natively)."""
        inputs = self.processor(
            text=[text], return_tensors="pt",
            padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            out = self.model.get_text_features(**inputs)
            # transformers 5.x may return BaseModelOutputWithPooling instead of tensor
            feats = out.pooler_output if hasattr(out, "pooler_output") else out
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def _preprocess_images(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        return inputs["pixel_values"]

    def _encode_images_trt(self, images) -> torch.Tensor:
        pixel_values = self._preprocess_images(images).to(self.device).float()
        feats_np = self.trt_vision(pixel_values)
        feats = torch.from_numpy(feats_np)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def _encode_images_pt(self, images) -> torch.Tensor:
        pixel_values = self._preprocess_images(images).to(self.device)
        with torch.no_grad():
            out = self.model.get_image_features(pixel_values=pixel_values)
            # transformers 5.x may return BaseModelOutputWithPooling instead of tensor
            feats = out.pooler_output if hasattr(out, "pooler_output") else out
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu()

    @staticmethod
    def _load_image(path_str):
        try:
            img = Image.open(path_str).convert("RGB")
            w, h = img.size
            if max(w, h) > 1024:
                ratio = 1024 / max(w, h)
                img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
            return path_str, img
        except Exception:
            return path_str, None

    # ── Batch embedding ─────────────────────────────────────────────────

    def get_image_embeddings_batch(self, paths):
        """Compute embeddings for a list of paths. Returns {path: tensor(1,1152)}."""
        results = {}

        # Memory cache hits
        mem_miss = []
        for p in paths:
            if p in self._mem_cache:
                results[p] = self._mem_cache[p]
            else:
                mem_miss.append(p)

        # Disk cache batch lookup
        if mem_miss:
            disk_hits = self.cache.get_clip_embeddings_batch(mem_miss, self.model_hash)
            for path_str, cached in disk_hits.items():
                tensor = torch.from_numpy(cached).reshape(1, FEAT_DIM)
                self._mem_cache[path_str] = tensor
                results[path_str] = tensor

        to_load = [p for p in mem_miss if p not in results]
        if not to_load:
            return results

        # Threaded image loading
        loaded     = list(self._load_pool.map(self._load_image, to_load))
        to_compute = [(p, img) for p, img in loaded if img is not None]

        # Batched GPU inference
        for i in range(0, len(to_compute), self.batch_size):
            batch       = to_compute[i:i + self.batch_size]
            batch_paths = [p for p, _ in batch]
            batch_imgs  = [img for _, img in batch]
            try:
                if self.trt_vision is not None:
                    feats = self._encode_images_trt(batch_imgs)
                else:
                    feats = self._encode_images_pt(batch_imgs)

                to_save = []
                for j, path_str in enumerate(batch_paths):
                    feat = feats[j:j+1]
                    self._mem_cache[path_str] = feat
                    results[path_str] = feat
                    to_save.append((path_str, feat.numpy().flatten()))
                self.cache.save_clip_embeddings_batch(to_save, self.model_hash)

            except torch.cuda.OutOfMemoryError:
                logger.warning(f"[SigLIP2] OOM, halving batch size from {self.batch_size}")
                torch.cuda.empty_cache()
                self.batch_size = max(4, self.batch_size // 2)
                for p_str in batch_paths:
                    try:
                        _, img = self._load_image(p_str)
                        if img is not None:
                            f = self._encode_images_pt([img])
                            self._mem_cache[p_str] = f
                            results[p_str] = f
                            self.cache.save_clip_embeddings_batch(
                                [(p_str, f.numpy().flatten())], self.model_hash)
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"[SigLIP2] Batch failed: {e}")

            if self.device == "cuda" and i % (self.batch_size * 4) == 0 and i > 0:
                torch.cuda.empty_cache()

        return results

    # ── Folder processing ───────────────────────────────────────────────

    def process_folder(self, folder, status_callback=None, recursive=False):
        """Pre-compute and cache embeddings for all images in a folder."""
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
            all_paths = [str(f) for f in files]
            mem_miss  = [p for p in all_paths if p not in self._mem_cache]

            if mem_miss:
                disk_hits = self.cache.get_clip_embeddings_batch(mem_miss, self.model_hash)
                for path_str, cached in disk_hits.items():
                    self._mem_cache[path_str] = torch.from_numpy(cached).reshape(1, FEAT_DIM)

            uncached     = [p for p in all_paths if p not in self._mem_cache]
            cached_count = len(all_paths) - len(uncached)

            if not uncached:
                if status_callback:
                    status_callback(100, f"All {len(files)} images cached")
                return len(files)

            if status_callback:
                status_callback(0,
                    f"Cache: {cached_count}/{len(all_paths)}, extracting {len(uncached)} remaining...")

            total       = len(uncached)
            processed   = 0
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

    # ── Search ──────────────────────────────────────────────────────────

    def search(self, query, folder, top_k=10, recursive=False):
        """Search images by text query. Korean/English supported natively."""
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
            path_str = str(f)
            emb = self._mem_cache.get(path_str)
            if emb is None:
                cached = self.cache.get_clip_embedding(path_str, self.model_hash)
                if cached is not None:
                    emb = torch.from_numpy(cached).reshape(1, FEAT_DIM)
                    self._mem_cache[path_str] = emb
            if emb is not None:
                emb_dev = emb.to(self.device)
                sim = torch.matmul(text_feats, emb_dev.T).squeeze()
                score = sim.item() if sim.dim() == 0 else sim.max().item()
                results.append((path_str, score))

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

        # Compute query image embedding (not cached — used once)
        _, img = self._load_image(str(query_image_path))
        if img is None:
            raise ValueError(f"Failed to load query image: {query_image_path}")

        if self.trt_vision is not None:
            query_emb = self._encode_images_trt([img])
        else:
            query_emb = self._encode_images_pt([img])
        query_emb_dev = query_emb.to(self.device)  # (1, 1152)
        query_path_str = str(query_image_path)

        results = []
        for f in files:
            path_str = str(f)
            if path_str == query_path_str:
                continue
            emb = self._mem_cache.get(path_str)
            if emb is None:
                cached = self.cache.get_clip_embedding(path_str, self.model_hash)
                if cached is not None:
                    emb = torch.from_numpy(cached).reshape(1, FEAT_DIM)
                    self._mem_cache[path_str] = emb
            if emb is not None:
                emb_dev = emb.to(self.device)
                sim = torch.matmul(query_emb_dev, emb_dev.T).squeeze()
                score = sim.item() if sim.dim() == 0 else sim.max().item()
                results.append((path_str, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    # ── Misc ────────────────────────────────────────────────────────────

    def stop(self):
        self.is_processing = False

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
        torch.cuda.empty_cache()
        logger.info("SigLIP2 offloaded to CPU")

    def to_gpu(self):
        """Move model back to GPU."""
        if not self.initialized or self.device == "cpu":
            return
        if self.model is not None:
            self.model.to(self.device)
        if getattr(self, '_had_trt', False) and self.trt_vision is None:
            try:
                self.trt_vision = TensorRTSigLIP2Vision(self._trt_engine_file)
            except Exception as e:
                logger.warning(f"SigLIP2 TRT reload failed: {e}")
                self._had_trt = False
        logger.info("SigLIP2 loaded to GPU")

    def clear_memory_cache(self):
        self._mem_cache.clear()
        if self.device == "cuda":
            torch.cuda.empty_cache()
