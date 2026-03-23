"""
Unified inference engine with TensorRT > PyTorch fallback.
Manages DINOv2 TensorRT engine build/cache for image duplicate detection.
"""

import os
import sys
import logging
import subprocess
import threading
import torch
import numpy as np
from core.model_paths import MODEL_DIR

logger = logging.getLogger(__name__)

# Global lock: prevents concurrent TRT engine builds.
# TensorRT builder is NOT safe for concurrent use on the same GPU —
# simultaneous builds from multiple threads cause OOM or CUDA fatal errors.
_TRT_BUILD_LOCK = threading.Lock()

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_engine_cache")
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DINOV2_FEAT_DIM = 768  # DINOv2-Base output dimension

# ── Shared TRT logger singleton ────────────────────────────────────────────
_trt_logger_instance = None


def _get_trt_logger():
    """Return a shared TRT Logger instance (avoids TRT global registry conflicts)."""
    global _trt_logger_instance
    if _trt_logger_instance is None:
        import tensorrt as trt
        _trt_logger_instance = trt.Logger(trt.Logger.WARNING)
    return _trt_logger_instance


def get_gpu_info():
    """Return (gpu_available, gpu_name, gpu_memory_gb)."""
    if not torch.cuda.is_available():
        return False, "CPU", 0
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    return True, name, mem


def _safe_gpu_tag():
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name(0)
    return name.replace("NVIDIA GeForce ", "").replace("NVIDIA ", "").replace(" ", "_")


# ═══════════════════════════════════════════════════════════════════════════
# TRT subprocess builder — crash-safe wrapper
#
# Both ONNX export AND TRT build run inside the subprocess so that:
#   1. TracerWarnings from torch.onnx.export appear in the subprocess only
#   2. Native C++ aborts from build_serialized_network() kill the subprocess,
#      NOT the main Qt application
# ═══════════════════════════════════════════════════════════════════════════

# Preamble injected at the start of every subprocess script:
#  - Suppress all Python warnings (no TracerWarnings polluting stderr)
#  - On Windows, suppress crash dialogs so a native crash exits immediately
#    instead of blocking on a WER dialog (which would hang subprocess.run)
_SUBPROCESS_PREAMBLE = (
    "import warnings; warnings.filterwarnings('ignore')\n"
    "import sys\n"
    "if sys.platform == 'win32':\n"
    "    import ctypes\n"
    # SEM_FAILCRITICALERRORS|SEM_NOGPFAULTERRORBOX|SEM_NOOPENFILEERRORBOX = 0x8003
    "    ctypes.windll.kernel32.SetErrorMode(0x8003)\n"
)


def _build_trt_subprocess(onnx_file, engine_file, build_module, build_fn):
    """Run a TRT build function in a subprocess to protect the main app from native crashes.

    The subprocess receives both onnx_file and engine_file paths; the build
    function is responsible for ONNX export (if needed) AND TRT compilation.
    """
    code = (
        _SUBPROCESS_PREAMBLE +
        f"sys.path.insert(0, {_PROJECT_ROOT!r})\n"
        f"from {build_module} import {build_fn}\n"
        f"{build_fn}({onnx_file!r}, {engine_file!r})\n"
    )
    # On Windows: CREATE_NO_WINDOW prevents a console window from flashing up
    kwargs = {}
    if sys.platform == 'win32':
        kwargs['creationflags'] = 0x08000000  # CREATE_NO_WINDOW

    try:
        result = subprocess.run(
            [sys.executable, '-c', code],
            timeout=600,          # 10-minute limit per engine build
            stderr=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            **kwargs,
        )
        if result.returncode == 0 and os.path.exists(engine_file):
            return True
        # Log subprocess stderr so build failures are diagnosable
        if result.stderr:
            try:
                err_text = result.stderr.decode('utf-8', errors='replace').strip()
                if err_text:
                    logger.warning(f"TRT {build_fn} subprocess stderr:\n{err_text[-2000:]}")
            except Exception:
                pass
        logger.warning(f"TRT {build_fn} subprocess failed (exit {result.returncode})")
        return False
    except subprocess.TimeoutExpired as exc:
        # Kill the hung subprocess before returning
        try:
            exc.process.kill()
        except Exception:
            pass
        logger.warning(f"TRT {build_fn} timed out after 10 minutes")
        return False
    except Exception as e:
        logger.warning(f"TRT {build_fn} subprocess error: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════
# TensorRT DINOv2 engine
# ═══════════════════════════════════════════════════════════════════════════

def _engine_path(gpu_tag):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"dinov2_base_fp16_{gpu_tag}.engine")


def _onnx_path():
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, "dinov2_base_feature_extractor.onnx")


class _DINOv2Wrapper(torch.nn.Module):
    """Wrapper that extracts CLS token from DINOv2 for clean ONNX export."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        features = self.backbone.forward_features(x)
        return features["x_norm_clstoken"]


def _export_dinov2_onnx(onnx_file):
    torch.hub.set_dir(MODEL_DIR)
    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True)
    backbone.eval()
    wrapper = _DINOv2Wrapper(backbone)
    wrapper.eval()
    dummy = torch.randn(1, 3, 224, 224)
    # dynamo=False: use legacy TorchScript exporter (PyTorch 2.9+ defaults to dynamo=True)
    torch.onnx.export(
        wrapper, dummy, onnx_file,
        input_names=["input"], output_names=["features"],
        dynamic_axes={"input": {0: "batch"}, "features": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )


def _build_trt_engine(onnx_file, engine_file):
    import tensorrt as trt
    trt_logger = _get_trt_logger()
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)
    if not parser.parse_from_file(str(onnx_file)):
        for i in range(parser.num_errors):
            logger.error(str(parser.get_error(i)))
        raise RuntimeError("ONNX parsing failed")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, 3, 224, 224), (32, 3, 224, 224), (192, 3, 224, 224))
    config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    with open(engine_file, "wb") as f:
        f.write(serialized)


def _build_dinov2_all(onnx_file, engine_file):
    """Export DINOv2 ONNX (if needed) + build TRT engine.

    Designed to run entirely inside a subprocess so that TracerWarnings and
    any native TRT crash are isolated from the main application process.
    """
    if not os.path.exists(onnx_file):
        _export_dinov2_onnx(onnx_file)
    _build_trt_engine(onnx_file, engine_file)


class TensorRTDINOv2:
    """TensorRT FP16 DINOv2-Base feature extractor."""

    def __init__(self, engine_file):
        import tensorrt as trt
        runtime = trt.Runtime(_get_trt_logger())
        with open(engine_file, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self._stream = torch.cuda.Stream()
        # Query max batch from engine profile; fallback for older engines
        try:
            self._max_batch = self.engine.get_tensor_profile_shape("input", 0)[2][0]
        except Exception:
            self._max_batch = 64

    def __call__(self, batch_tensor):
        """batch_tensor: CUDA float32 (N,3,224,224) -> numpy (N,768)
        Always runs at max_batch size (padding) so Myelin graph stays loaded."""
        n = batch_tensor.shape[0]
        dev = batch_tensor.device

        if n > self._max_batch:
            # Split into max_batch chunks; each chunk is padded → same shape every time
            results = []
            for start in range(0, n, self._max_batch):
                results.append(self._infer_padded(batch_tensor[start:start + self._max_batch], dev))
            return np.concatenate(results, axis=0)

        return self._infer_padded(batch_tensor, dev)

    def _infer_padded(self, chunk, dev):
        """Pad chunk to max_batch, infer, return only the real rows."""
        real_n = chunk.shape[0]
        if real_n < self._max_batch:
            pad = torch.zeros(self._max_batch - real_n, 3, 224, 224,
                              dtype=chunk.dtype, device=dev)
            chunk = torch.cat([chunk, pad], dim=0)
        chunk = chunk.contiguous()
        output = torch.empty((self._max_batch, DINOV2_FEAT_DIM),
                             dtype=torch.float32, device=dev)
        self.context.set_input_shape("input", (self._max_batch, 3, 224, 224))
        self.context.set_tensor_address("input", chunk.data_ptr())
        self.context.set_tensor_address("features", output.data_ptr())
        self.context.execute_async_v3(stream_handle=self._stream.cuda_stream)
        self._stream.synchronize()
        return output.cpu().numpy()[:real_n]


def build_dinov2_tensorrt(progress_callback=None):
    """Build or load TensorRT DINOv2 engine. Returns TensorRTDINOv2 or None."""
    try:
        import tensorrt  # noqa: F401
    except ImportError:
        return None

    gpu_tag = _safe_gpu_tag()
    if gpu_tag is None:
        return None

    engine_file = _engine_path(gpu_tag)

    # Fast path: load already-built engine (no lock needed)
    if os.path.exists(engine_file):
        if progress_callback:
            progress_callback("Loading cached TensorRT engine...")
        try:
            ext = TensorRTDINOv2(engine_file)
            if progress_callback:
                progress_callback("TensorRT engine loaded")
            return ext
        except Exception as e:
            logger.warning(f"Cached engine load failed: {e}")
            os.remove(engine_file)

    # Build path: serialize with global lock (one TRT build at a time)
    with _TRT_BUILD_LOCK:
        # Re-check after acquiring lock — another thread may have built it while we waited
        if os.path.exists(engine_file):
            try:
                return TensorRTDINOv2(engine_file)
            except Exception:
                os.remove(engine_file)

        onnx_file = _onnx_path()

        if progress_callback:
            progress_callback(f"Building TensorRT engine ({gpu_tag})... first time only (1-3 min)")

        # ONNX export + TRT build both run in subprocess — keeps main process safe
        ok = _build_trt_subprocess(onnx_file, engine_file,
                                   'core.inference_engine', '_build_dinov2_all')
        if not ok:
            # Clean up partial ONNX so next run retries from scratch
            if os.path.exists(onnx_file):
                try:
                    os.remove(onnx_file)
                except Exception:
                    pass
            logger.warning("DINOv2 TRT build failed, falling back to PyTorch")
            return None

        if progress_callback:
            progress_callback("TensorRT engine ready")
        return TensorRTDINOv2(engine_file)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
