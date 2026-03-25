"""Background model update checker and downloader."""

import os
import glob
import json
import logging
import threading
import time
from pathlib import Path

from core.model_paths import MODEL_DIR
from core.inference_engine import CACHE_DIR

logger = logging.getLogger(__name__)

_VERSION_FILE = Path(MODEL_DIR) / ".model_versions.json"

# Models to track
HF_MODELS = [
    ("apple/DFN5B-CLIP-ViT-H-14-378", "CLIP"),
    ("Helsinki-NLP/opus-mt-ko-en", "번역 (MarianMT)"),
    ("facebook/metaclip-2-worldwide-huge-quickgelu", "MetaCLIP2"),
]
HUB_REPOS = [
    ("facebookresearch/dinov2", "DINOv2"),
]

# Mapping from model ID to TRT engine filename prefix
_TRT_ENGINE_PREFIXES = {
    "apple/DFN5B-CLIP-ViT-H-14-378": "clip_dfn5b_vith_fp16_",
    "facebook/metaclip-2-worldwide-huge-quickgelu": "metaclip2_worldwide_fp16_",
    "facebookresearch/dinov2": "dinov2_vitb14_fp16_",
}


# ── Version file helpers ────────────────────────────────────────────────────

def _load_versions():
    if _VERSION_FILE.exists():
        try:
            return json.loads(_VERSION_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_versions(versions):
    try:
        _VERSION_FILE.write_text(
            json.dumps(versions, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.warning(f"Version file save failed: {e}")


# ── Local version helpers ────────────────────────────────────────────────────

def _get_hf_local_sha(model_name):
    """Read commit SHA from HuggingFace local cache (refs/main file)."""
    safe = "models--" + model_name.replace("/", "--")
    refs_file = Path(MODEL_DIR) / safe / "refs" / "main"
    if refs_file.exists():
        return refs_file.read_text(encoding="utf-8").strip()
    return None


def _model_downloaded(model_name):
    """Check if a HuggingFace model is already downloaded to MODEL_DIR."""
    safe = "models--" + model_name.replace("/", "--")
    return (Path(MODEL_DIR) / safe).exists()


def _hub_model_downloaded(repo):
    """Check if a torch.hub repo is already downloaded to MODEL_DIR."""
    safe = repo.replace("/", "_") + "_main"
    return (Path(MODEL_DIR) / safe).exists()


# ── Remote version helpers ────────────────────────────────────────────────────

def _get_hf_remote_sha(model_name):
    """Fetch latest commit SHA from HuggingFace Hub API."""
    try:
        from huggingface_hub import model_info
        info = model_info(model_name)
        return info.sha
    except Exception as e:
        logger.debug(f"HF remote check failed [{model_name}]: {e}")
        return None


def _get_github_sha(repo):
    """Fetch latest commit SHA from GitHub API."""
    try:
        import urllib.request
        import json as _json
        org, name = repo.split("/")
        url = f"https://api.github.com/repos/{org}/{name}/commits/main"
        req = urllib.request.Request(
            url, headers={"Accept": "application/vnd.github+json",
                          "User-Agent": "media-manager-updater"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return _json.loads(resp.read())["sha"]
    except Exception as e:
        logger.debug(f"GitHub remote check failed [{repo}]: {e}")
        return None


# ── Main check function ────────────────────────────────────────────────────────

def check_for_updates():
    """Check all models for updates.
    Returns list of update dicts, empty list if all up-to-date or check fails."""
    updates = []
    versions = _load_versions()

    # HuggingFace models
    for model_id, display_name in HF_MODELS:
        if not _model_downloaded(model_id):
            continue
        local_sha = _get_hf_local_sha(model_id)
        if not local_sha:
            continue
        remote_sha = _get_hf_remote_sha(model_id)
        if not remote_sha:
            continue
        if local_sha != remote_sha:
            updates.append({
                "type": "huggingface",
                "id": model_id,
                "display": display_name,
                "local": local_sha[:8],
                "remote": remote_sha[:8],
            })

    # torch.hub models
    for repo, display_name in HUB_REPOS:
        if not _hub_model_downloaded(repo):
            continue
        local_sha = versions.get(f"hub:{repo}")
        if local_sha is None:
            # First check — record current remote as baseline
            remote_sha = _get_github_sha(repo)
            if remote_sha:
                versions[f"hub:{repo}"] = remote_sha
                _save_versions(versions)
            continue
        remote_sha = _get_github_sha(repo)
        if not remote_sha:
            continue
        if local_sha != remote_sha:
            updates.append({
                "type": "torchhub",
                "id": repo,
                "display": display_name,
                "local": local_sha[:8],
                "remote": remote_sha[:8],
            })

    return updates


# ── TRT cache invalidation ────────────────────────────────────────────────────

def _delete_trt_engines_for(model_id):
    """Delete TRT .engine files for a given model after a model update.
    Forces rebuild on next startup so the engine matches the new weights."""
    prefix = _TRT_ENGINE_PREFIXES.get(model_id)
    if not prefix:
        return
    pattern = os.path.join(CACHE_DIR, f"{prefix}*.engine")
    for f in glob.glob(pattern):
        try:
            os.remove(f)
            logger.info(f"Invalidated TRT engine: {f}")
        except Exception as e:
            logger.warning(f"Failed to delete TRT engine {f}: {e}")


# ── Download function ─────────────────────────────────────────────────────────

def download_updates(updates, progress_callback=None):
    """Download updated models. progress_callback(current, total, msg)."""
    versions = _load_versions()
    total = len(updates)

    for i, update in enumerate(updates):
        name = update["display"]
        if progress_callback:
            progress_callback(i, total, f"{name} 다운로드 중...")

        try:
            if update["type"] == "huggingface":
                from huggingface_hub import snapshot_download
                snapshot_download(update["id"], cache_dir=MODEL_DIR)
                # Invalidate TRT engine so it rebuilds against new weights
                _delete_trt_engines_for(update["id"])

            elif update["type"] == "torchhub":
                import torch
                torch.hub.set_dir(MODEL_DIR)
                torch.hub.load(update["id"], "dinov2_vitb14",
                               pretrained=True, force_reload=True)
                _delete_trt_engines_for(update["id"])
                versions[f"hub:{update['id']}"] = update["remote"]
                _save_versions(versions)

        except Exception as e:
            logger.error(f"Download failed [{name}]: {e}")
            raise RuntimeError(f"{name} 다운로드 실패: {e}")

    if progress_callback:
        progress_callback(total, total, "완료")


# ── Background checker thread ─────────────────────────────────────────────────

class ModelUpdateChecker(threading.Thread):
    """Daemon thread: waits for app to load, then checks for updates."""

    def __init__(self, on_updates_found, delay=5):
        super().__init__(daemon=True, name="ModelUpdateChecker")
        self._on_updates_found = on_updates_found
        self._delay = delay

    def run(self):
        time.sleep(self._delay)  # Let app fully initialize first
        try:
            updates = check_for_updates()
            if updates:
                self._on_updates_found(updates)
        except Exception as e:
            logger.debug(f"Update check error: {e}")
