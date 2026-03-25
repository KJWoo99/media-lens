"""MetaCLIP2 text-to-image search engine.

Model:  facebook/metaclip-2-worldwide-huge-quickgelu
        ViT-H/14, 1024-dim, 224px, 300+ languages natively supported.
        No translation model required — Korean/English/etc. work out of the box.
Cache:  Shared clip_cache table, unique model_hash key.
"""

import os
import time
import logging
import hashlib
from concurrent.futures import ThreadPoolExecutor
import torch
from pathlib import Path
from PIL import Image, ImageFile

from core.cache_manager import CacheManager
from core.model_paths import MODEL_DIR
from core.inference_engine import CACHE_DIR, _safe_gpu_tag, _get_trt_logger, _TRT_BUILD_LOCK, _build_trt_subprocess

logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


MODEL_ID   = "facebook/metaclip-2-worldwide-huge-quickgelu"
MODEL_HASH = hashlib.md5(MODEL_ID.encode()).hexdigest()[:8]
FEAT_DIM   = 1024
IMG_SIZE   = 224

IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tif', '.tiff',
    '.heic', '.heif', '.avif',
}


def _fmt_eta(seconds):
    if seconds <= 0:
        return ""
    m, s = divmod(int(seconds), 60)
    return f" (ETA {m}m {s:02d}s)" if m > 0 else f" (ETA {s}s)"


# ── TensorRT image encoder ──────────────────────────────────────────────────

class _MetaCLIPVisionWrapper(torch.nn.Module):
    """Wraps CLIPModel vision_model + visual_projection for ONNX export."""

    def __init__(self, vision_model, visual_projection):
        super().__init__()
        self.vision_model      = vision_model
        self.visual_projection = visual_projection

    def forward(self, pixel_values):
        outputs = self.vision_model(pixel_values=pixel_values)
        feats   = outputs.pooler_output
        feats   = self.visual_projection(feats)
        return feats / feats.norm(dim=-1, keepdim=True)


def _trt_engine_path(gpu_tag):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"metaclip2_worldwide_fp16_{gpu_tag}.engine")


def _trt_onnx_path():
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, "metaclip2_worldwide_image_encoder.onnx")


def _export_metaclip_onnx(onnx_file, model):
    from transformers import CLIPVisionModel
    try:
        vision_export = CLIPVisionModel.from_pretrained(
            MODEL_ID, cache_dir=MODEL_DIR, local_files_only=True,
            attn_implementation="eager",
            output_hidden_states=False, output_attentions=False,
        ).eval()
    except Exception:
        vision_export = CLIPVisionModel.from_pretrained(
            MODEL_ID, cache_dir=MODEL_DIR,
            attn_implementation="eager",
            output_hidden_states=False, output_attentions=False,
        ).eval()
    try:
        proj_cpu = model.visual_projection
        wrapper  = _MetaCLIPVisionWrapper(vision_export, proj_cpu)
        wrapper.eval()
        dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        with torch.no_grad():
            torch.onnx.export(
                wrapper, dummy, onnx_file,
                input_names=["pixel_values"],
                output_names=["features"],
                dynamic_axes={"pixel_values": {0: "batch"}, "features": {0: "batch"}},
                opset_version=17,
                dynamo=False,
            )
        logger.info(f"MetaCLIP2 ONNX exported → {onnx_file}")
    finally:
        del vision_export


def _build_metaclip_trt(onnx_file, engine_file):
    import tensorrt as trt
    trt_logger = _get_trt_logger()
    builder  = trt.Builder(trt_logger)
    network  = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser   = trt.OnnxParser(network, trt_logger)
    if not parser.parse_from_file(str(onnx_file)):
        for i in range(parser.num_errors):
            logger.error(str(parser.get_error(i)))
        raise RuntimeError("ONNX parsing failed for MetaCLIP2")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    profile.set_shape("pixel_values",
                      (1,  3, IMG_SIZE, IMG_SIZE),
                      (8,  3, IMG_SIZE, IMG_SIZE),
                      (64, 3, IMG_SIZE, IMG_SIZE))
    config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed for MetaCLIP2")
    with open(engine_file, "wb") as f:
        f.write(serialized)
    logger.info("MetaCLIP2 TRT engine built")


def _build_metaclip_all(onnx_file, engine_file):
    """Load model, export ONNX, build TRT. Runs inside subprocess."""
    if not os.path.exists(onnx_file):
        from transformers import CLIPModel
        try:
            model = CLIPModel.from_pretrained(
                MODEL_ID, cache_dir=MODEL_DIR, local_files_only=True,
                output_hidden_states=False, output_attentions=False,
            ).eval()
        except OSError:
            model = CLIPModel.from_pretrained(
                MODEL_ID, cache_dir=MODEL_DIR,
                output_hidden_states=False, output_attentions=False,
            ).eval()
        _export_metaclip_onnx(onnx_file, model)
        del model
        import gc; gc.collect()
    _build_metaclip_trt(onnx_file, engine_file)


class TensorRTMetaCLIPVision:
    """TensorRT FP16 MetaCLIP2 image encoder."""

    def __init__(self, engine_file):
        import tensorrt as trt
        runtime = trt.Runtime(_get_trt_logger())
        with open(engine_file, "rb") as f:
            self.engine  = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self._stream = torch.cuda.Stream()
        try:
            self._max_batch = self.engine.get_tensor_profile_shape("pixel_values", 0)[2][0]
        except Exception:
            self._max_batch = 32

    def __call__(self, pixel_values: torch.Tensor) -> torch.Tensor:
        n = pixel_values.shape[0]
        dev = pixel_values.device
        if n > self._max_batch:
            chunks = [self._infer_padded(pixel_values[i:i + self._max_batch], dev)
                      for i in range(0, n, self._max_batch)]
            return torch.cat(chunks, dim=0)
        return self._infer_padded(pixel_values, dev)

    def _infer_padded(self, chunk, dev):
        real_n = chunk.shape[0]
        if real_n < self._max_batch:
            pad   = torch.zeros(self._max_batch - real_n, 3, IMG_SIZE, IMG_SIZE,
                                dtype=chunk.dtype, device=dev)
            chunk = torch.cat([chunk, pad], dim=0)
        chunk  = chunk.contiguous()
        output = torch.empty((self._max_batch, FEAT_DIM), dtype=torch.float32, device=dev)
        self.context.set_input_shape("pixel_values", (self._max_batch, 3, IMG_SIZE, IMG_SIZE))
        self.context.set_tensor_address("pixel_values", chunk.data_ptr())
        self.context.set_tensor_address("features",     output.data_ptr())
        self.context.execute_async_v3(stream_handle=self._stream.cuda_stream)
        self._stream.synchronize()
        return output.cpu()[:real_n]


def _get_batch_size(trt=False):
    if not torch.cuda.is_available():
        return 4
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
        return 8 if trt else 4


# ── Main engine ────────────────────────────────────────────────────────────

class MetaCLIPEngine:
    """MetaCLIP2 worldwide — multilingual text-to-image search (300+ languages)."""

    def __init__(self):
        self.device     = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_hash = MODEL_HASH
        self.model      = None
        self.processor  = None
        self.trt_vision = None
        self.cache      = CacheManager()
        self.initialized    = False
        self.batch_size     = _get_batch_size(trt=False)
        self.is_processing  = False
        self._load_id       = 0
        self._mem_cache     = {}
        self._load_pool     = ThreadPoolExecutor(max_workers=4)
        self.backend_info   = "MetaCLIP2 not loaded"

    def initialize(self, progress_callback=None):
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        logging.getLogger("transformers").setLevel(logging.ERROR)

        if progress_callback:
            progress_callback("Loading MetaCLIP2...")

        from transformers import CLIPModel, CLIPProcessor

        if self.device == "cuda":
            torch.cuda.empty_cache()

        try:
            self.model = CLIPModel.from_pretrained(
                MODEL_ID, cache_dir=MODEL_DIR, local_files_only=True,
                output_hidden_states=False, output_attentions=False,
            ).to("cpu").eval()
            self.processor = CLIPProcessor.from_pretrained(
                MODEL_ID, cache_dir=MODEL_DIR, local_files_only=True)
        except OSError:
            self.model = CLIPModel.from_pretrained(
                MODEL_ID, cache_dir=MODEL_DIR,
                output_hidden_states=False, output_attentions=False,
            ).to("cpu").eval()
            self.processor = CLIPProcessor.from_pretrained(
                MODEL_ID, cache_dir=MODEL_DIR)

        for p in self.model.parameters():
            p.requires_grad = False

        trt_ok = self._init_trt(progress_callback)
        if trt_ok:
            self.batch_size = max(self.batch_size, int(self.batch_size * 1.5))

        self.initialized = True
        backend = "TRT FP16" if trt_ok else "PyTorch"
        self.backend_info = f"MetaCLIP2 ready ({self.device}, {backend})"
        if progress_callback:
            progress_callback(self.backend_info)
        logger.info(f"MetaCLIP2 ready ({self.device}, {backend}), batch_size={self.batch_size}")
        return True

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

        if os.path.exists(engine_file):
            if progress_callback:
                progress_callback("Loading MetaCLIP2 TRT engine...")
            try:
                self.trt_vision      = TensorRTMetaCLIPVision(engine_file)
                self._trt_engine_file = engine_file
                return True
            except Exception as e:
                logger.warning(f"MetaCLIP2 TRT load failed: {e}")
                os.remove(engine_file)

        with _TRT_BUILD_LOCK:
            if os.path.exists(engine_file):
                try:
                    self.trt_vision      = TensorRTMetaCLIPVision(engine_file)
                    self._trt_engine_file = engine_file
                    return True
                except Exception:
                    os.remove(engine_file)
            try:
                if progress_callback:
                    progress_callback(f"Building MetaCLIP2 TRT engine ({gpu_tag})... first time only (3-5 min)")
                if not _build_trt_subprocess(onnx_file, engine_file,
                                             'core.metaclip_engine', '_build_metaclip_all'):
                    raise RuntimeError("MetaCLIP2 TRT build failed in subprocess")
                self.trt_vision      = TensorRTMetaCLIPVision(engine_file)
                self._trt_engine_file = engine_file
                return True
            except Exception as e:
                logger.warning(f"MetaCLIP2 TRT failed, falling back to PyTorch: {e}")
                if os.path.exists(onnx_file):
                    try:
                        os.remove(onnx_file)
                    except Exception:
                        pass
                return False

    def _encode_trt(self, images):
        pixel_values = self.processor(images=images, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(self.device).float()
        return self.trt_vision(pixel_values)

    def _encode_pytorch(self, images):
        actual_device = next(self.model.parameters()).device
        inputs = self.processor(images=images, return_tensors="pt").to(actual_device)
        with torch.no_grad():
            outputs = self.model.vision_model(**inputs)
            feats   = outputs.pooler_output if outputs.pooler_output is not None \
                      else outputs.last_hidden_state.mean(dim=1)
            if hasattr(self.model, 'visual_projection'):
                feats = self.model.visual_projection(feats)
            feats = feats / torch.norm(feats, dim=-1, keepdim=True)
        return feats.cpu()

    def _compute_image_features(self, image):
        w, h   = image.size
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
        w, h   = image.size
        n_seg  = max(2, int(aspect_ratio // 1.5))
        seg_w  = w // n_seg
        feats_list = []
        for i in range(n_seg):
            left = i * seg_w
            right = w if i == n_seg - 1 else left + seg_w
            seg  = image.crop((left, 0, right, h))
            feats_list.append(
                self._encode_trt([seg]) if self.trt_vision is not None
                else self._encode_pytorch([seg])
            )
            if self.device == "cuda" and i % 3 == 2:
                torch.cuda.empty_cache()
        return torch.cat(feats_list, dim=0)

    def compute_text_features(self, text: str):
        """Encode text directly — no translation (300+ languages natively)."""
        actual_device = next(self.model.parameters()).device
        inputs = self.processor(text=[text], return_tensors="pt",
                                padding=True, truncation=True).to(actual_device)
        with torch.no_grad():
            outputs = self.model.text_model(**inputs)
            feats   = outputs.pooler_output if outputs.pooler_output is not None \
                      else outputs.last_hidden_state[:, -1, :]
            if hasattr(self.model, 'text_projection'):
                feats = self.model.text_projection(feats)
            feats = feats / torch.norm(feats, dim=-1, keepdim=True)
        return feats

    def get_image_embedding(self, path):
        path_str = str(path)
        if path_str in self._mem_cache:
            return self._mem_cache[path_str]
        cached = self.cache.get_clip_embedding(path_str, self.model_hash)
        if cached is not None:
            tensor = torch.from_numpy(cached).reshape(-1, FEAT_DIM)
            self._mem_cache[path_str] = tensor
            return tensor
        try:
            img   = Image.open(path_str).convert("RGB")
            feats = self._compute_image_features(img)
            self._mem_cache[path_str] = feats
            self.cache.save_clip_embedding(path_str, self.model_hash, feats.numpy().flatten())
            return feats
        except Exception as e:
            logger.warning(f"[MetaCLIP2] Failed: {path_str} - {e}")
            return None

    @staticmethod
    def _load_and_resize(path_str):
        try:
            img = Image.open(path_str).convert("RGB")
            w, h = img.size
            if max(w, h) > 1024:
                ratio = 1024 / max(w, h)
                img   = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
            return path_str, img
        except Exception:
            return path_str, None

    def get_image_embeddings_batch(self, paths):
        results  = {}
        mem_miss = []
        for p in paths:
            if p in self._mem_cache:
                results[p] = self._mem_cache[p]
            else:
                mem_miss.append(p)

        if mem_miss:
            disk_hits = self.cache.get_clip_embeddings_batch(mem_miss, self.model_hash)
            for p, cached in disk_hits.items():
                tensor = torch.from_numpy(cached).reshape(-1, FEAT_DIM)
                self._mem_cache[p] = tensor
                results[p]         = tensor

        to_load = [p for p in mem_miss if p not in results]
        if not to_load:
            return results

        loaded     = list(self._load_pool.map(self._load_and_resize, to_load))
        to_compute = [(p, img) for p, img in loaded if img is not None]
        for p, img in loaded:
            if img is None:
                logger.warning(f"[MetaCLIP2] Failed to load: {p}")

        for i in range(0, len(to_compute), self.batch_size):
            batch       = to_compute[i:i + self.batch_size]
            batch_paths = [p for p, _ in batch]
            batch_imgs  = [img for _, img in batch]
            try:
                feats_cpu = (self._encode_trt(batch_imgs) if self.trt_vision is not None
                             else self._encode_pytorch(batch_imgs))
                to_save = []
                for j, p in enumerate(batch_paths):
                    f = feats_cpu[j:j+1]
                    self._mem_cache[p] = f
                    to_save.append((p, f.numpy().flatten()))
                    results[p] = f
                self.cache.save_clip_embeddings_batch(to_save, self.model_hash)
            except torch.cuda.OutOfMemoryError:
                logger.warning(f"[MetaCLIP2] OOM, halving batch_size from {self.batch_size}")
                torch.cuda.empty_cache()
                self.batch_size = max(2, self.batch_size // 2)
                for p in batch_paths:
                    emb = self.get_image_embedding(p)
                    if emb is not None:
                        results[p] = emb
            except Exception as e:
                logger.warning(f"[MetaCLIP2] Batch failed: {e}")
                for p in batch_paths:
                    emb = self.get_image_embedding(p)
                    if emb is not None:
                        results[p] = emb

            if self.device == "cuda" and i % (self.batch_size * 4) == 0 and i > 0:
                torch.cuda.empty_cache()

        return results

    def process_folder(self, folder, status_callback=None, recursive=False):
        if status_callback:
            status_callback(0, "Scanning folder...")

        folder = Path(folder)
        files  = (list(folder.rglob("*")) if recursive else list(folder.iterdir()))
        files  = [f for f in files if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]

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
                for p, cached in disk_hits.items():
                    self._mem_cache[p] = torch.from_numpy(cached).reshape(-1, FEAT_DIM)

            uncached      = [p for p in all_paths if p not in self._mem_cache]
            cached_count  = len(all_paths) - len(uncached)

            if not uncached:
                if status_callback:
                    status_callback(100, f"All {len(files)} images cached")
                return len(files)

            if status_callback:
                status_callback(0, f"Cache: {cached_count}/{len(all_paths)}, extracting {len(uncached)}...")

            total      = len(uncached)
            processed  = 0
            batch_start = time.monotonic()

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
        folder = Path(folder)
        files  = (list(folder.rglob("*")) if recursive else list(folder.iterdir()))
        files  = [f for f in files if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
        if not files:
            return []

        text_feats = self.compute_text_features(query)
        results    = []
        for f in files:
            emb = self.get_image_embedding(str(f))
            if emb is not None:
                emb_dev = emb.to(self.device)
                sim     = torch.matmul(text_feats, emb_dev.T).squeeze(0)
                max_sim = sim.max().item() if sim.dim() > 0 else sim.item()
                results.append((str(f), max_sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def search_by_image(self, query_image_path, folder, top_k=10, recursive=False):
        folder = Path(folder)
        files  = (list(folder.rglob("*")) if recursive else list(folder.iterdir()))
        files  = [f for f in files if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
        if not files:
            return []

        query_emb = self.get_image_embedding(str(query_image_path))
        if query_emb is None:
            raise ValueError(f"Failed to load query image: {query_image_path}")

        query_emb_dev   = query_emb.to(self.device)
        query_path_str  = str(query_image_path)
        results         = []
        for f in files:
            path_str = str(f)
            if path_str == query_path_str:
                continue
            emb = self.get_image_embedding(path_str)
            if emb is not None:
                sim     = torch.matmul(query_emb_dev, emb.to(self.device).T)
                max_sim = sim.max().item()
                results.append((path_str, max_sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def to_cpu(self):
        if not self.initialized or self.device == "cpu":
            return
        if self.model is not None:
            self.model.cpu()
        if self.trt_vision is not None:
            self._had_trt  = True
            del self.trt_vision
            self.trt_vision = None
        torch.cuda.empty_cache()
        logger.info("MetaCLIP2 offloaded to CPU")

    def to_gpu(self):
        if not self.initialized or self.device == "cpu":
            return
        if self.model is not None:
            self.model.to(self.device)
        if getattr(self, '_had_trt', False) and self.trt_vision is None:
            try:
                self.trt_vision = TensorRTMetaCLIPVision(self._trt_engine_file)
            except Exception as e:
                logger.warning(f"MetaCLIP2 TRT reload failed: {e}")
                self._had_trt = False
        logger.info("MetaCLIP2 loaded to GPU")

    def clear_memory_cache(self):
        self._mem_cache.clear()
        if self.device == "cuda":
            torch.cuda.empty_cache()
