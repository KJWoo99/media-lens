"""FFmpeg/FFprobe wrapper utilities for video processing."""

import os
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v'}


def is_video_file(path):
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def collect_videos(folder_path, recursive=True):
    """Collect video file paths from a folder."""
    folder = Path(folder_path)
    videos = []
    if recursive:
        for ext in VIDEO_EXTENSIONS:
            videos.extend(folder.rglob(f"*{ext}"))
    else:
        for ext in VIDEO_EXTENSIONS:
            videos.extend(folder.glob(f"*{ext}"))
    return sorted(videos)


def check_ffmpeg():
    """Check if ffmpeg and ffprobe are available."""
    for cmd in ['ffmpeg', 'ffprobe']:
        if shutil.which(cmd) is None:
            return False, f"'{cmd}' not found in PATH"
    return True, "OK"


def get_video_metadata(filepath):
    """Get video metadata via ffprobe, fallback to OpenCV."""
    filepath = str(filepath)
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-count_packets',
            '-show_entries', 'stream=width,height,r_frame_rate,nb_read_packets,duration',
            '-of', 'json',
            filepath
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            data = json.loads(result.stdout)
            if 'streams' in data and data['streams']:
                stream = data['streams'][0]
                fps_str = stream.get('r_frame_rate', '30/1')
                if '/' in fps_str:
                    num, den = map(int, fps_str.split('/'))
                    fps = num / den if den != 0 else 30.0
                else:
                    fps = float(fps_str)
                return {
                    'width': int(stream.get('width', 0)),
                    'height': int(stream.get('height', 0)),
                    'fps': fps,
                    'frame_count': int(stream.get('nb_read_packets', 0)),
                    'duration': float(stream.get('duration', 0))
                }
    except Exception:
        pass

    # OpenCV fallback
    try:
        cap = cv2.VideoCapture(filepath)
        if cap.isOpened():
            meta = {
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': cap.get(cv2.CAP_PROP_FPS) or 30.0,
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'duration': 0
            }
            if meta['fps'] > 0:
                meta['duration'] = meta['frame_count'] / meta['fps']
            cap.release()
            return meta
    except Exception:
        pass
    return None


def extract_frames_ffmpeg(filepath, sample_interval, timeout=300):
    """Extract frames using ffmpeg, returns list of numpy arrays (BGR)."""
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp()
        output_pattern = os.path.join(temp_dir, "frame_%04d.jpg")
        cmd = [
            'ffmpeg', '-loglevel', 'error',
            '-err_detect', 'ignore_err',
            '-i', str(filepath),
            '-vf', f'select=not(mod(n\\,{sample_interval}))',
            '-vsync', '0', '-q:v', '2',
            output_pattern
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)

        frames = []
        for frame_file in sorted(Path(temp_dir).glob("frame_*.jpg")):
            frame = cv2.imread(str(frame_file))
            if frame is not None:
                frames.append(frame)
        return frames
    except Exception:
        return []
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def extract_frames_opencv(filepath, sample_interval, max_frames=None):
    """Extract frames using OpenCV (fallback)."""
    frames = []
    try:
        cap = cv2.VideoCapture(str(filepath))
        if not cap.isOpened():
            return frames
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = 0
        while idx < frame_count:
            if max_frames and len(frames) >= max_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            idx += sample_interval
        cap.release()
    except Exception:
        pass
    return frames
