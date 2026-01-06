# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Video preprocessing cache generator for distributed execution.

This script pre-computes face detection results and affine transformation
matrices for videos, enabling faster inference when the same video is
used with different audio tracks.

Usage:
    python scripts/generate_video_cache.py \
        --video_path assets/demo1_video.mp4 \
        --output_dir cache/video \
        --resolution 512

    # Batch processing
    python scripts/generate_video_cache.py \
        --video_dir /path/to/videos \
        --output_dir cache/video \
        --resolution 512 \
        --num_workers 4
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass, asdict
from multiprocessing import Pool, cpu_count

import torch
import numpy as np
import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from latentsync.utils.util import read_video, gather_video_paths_recursively
from latentsync.utils.image_processor import ImageProcessor, load_fixed_mask


@dataclass
class VideoPreprocessCache:
    """Video preprocessing cache data structure."""
    video_hash: str
    video_path: str
    num_frames: int
    resolution: int
    # Tensor data
    faces: torch.Tensor          # (N, 3, H, W) uint8
    boxes: List[Tuple[int, int, int, int]]  # N x (x1, y1, x2, y2)
    affine_matrices: List[torch.Tensor]     # N x (1, 2, 3) float32


def compute_video_hash(video_path: str) -> str:
    """Compute SHA256 hash of video file (first 16 chars)."""
    hasher = hashlib.sha256()
    with open(video_path, 'rb') as f:
        # Read in chunks for large files
        for chunk in iter(lambda: f.read(65536), b''):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def generate_single_video_cache(
    video_path: str,
    output_dir: str,
    resolution: int = 512,
    device: str = "cuda",
    force_regenerate: bool = False,
) -> Optional[str]:
    """
    Generate preprocessing cache for a single video.

    Args:
        video_path: Path to input video file
        output_dir: Directory to save cache files
        resolution: Target resolution (default: 512)
        device: Device for face detection (default: "cuda")
        force_regenerate: Regenerate cache even if exists

    Returns:
        Path to saved cache file, or None if failed
    """
    video_path = str(video_path)

    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return None

    # Compute video hash for cache filename
    video_hash = compute_video_hash(video_path)
    cache_filename = f"{video_hash}_{resolution}.pt"
    cache_path = Path(output_dir) / cache_filename

    # Check if cache already exists
    if cache_path.exists() and not force_regenerate:
        print(f"Cache already exists: {cache_path}")
        return str(cache_path)

    try:
        # Read video (with FPS conversion to 25fps)
        print(f"Reading video: {video_path}")
        video_frames = read_video(video_path, change_fps=True, use_decord=False)
        print(f"  Frames: {len(video_frames)}, Shape: {video_frames.shape}")

        # Initialize ImageProcessor with face detector
        mask_image = load_fixed_mask(resolution)
        image_processor = ImageProcessor(
            resolution=resolution,
            device=device,
            mask_image=mask_image
        )

        # Process each frame
        faces = []
        boxes = []
        affine_matrices = []
        failed_frames = 0

        print(f"Processing {len(video_frames)} frames...")
        for idx, frame in enumerate(tqdm.tqdm(video_frames, desc="Face detection")):
            try:
                face, box, affine_matrix = image_processor.affine_transform(frame)
                faces.append(face)
                boxes.append(tuple(box))
                affine_matrices.append(affine_matrix.cpu())
            except RuntimeError as e:
                # Face not detected - use previous frame's data
                failed_frames += 1
                if len(faces) > 0:
                    faces.append(faces[-1])
                    boxes.append(boxes[-1])
                    affine_matrices.append(affine_matrices[-1])
                else:
                    raise RuntimeError(f"Face not detected in first frame: {e}")

        if failed_frames > 0:
            print(f"  Warning: Face detection failed for {failed_frames} frames")

        # Stack faces tensor
        faces_tensor = torch.stack(faces)  # (N, 3, H, W)

        # Create cache object
        cache = VideoPreprocessCache(
            video_hash=video_hash,
            video_path=os.path.abspath(video_path),
            num_frames=len(video_frames),
            resolution=resolution,
            faces=faces_tensor,
            boxes=boxes,
            affine_matrices=affine_matrices,
        )

        # Save cache
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        torch.save(asdict(cache), cache_path)

        # Print cache info
        cache_size_mb = cache_path.stat().st_size / 1024 / 1024
        print(f"Cache saved: {cache_path}")
        print(f"  Size: {cache_size_mb:.2f} MB")
        print(f"  Frames: {cache.num_frames}")

        return str(cache_path)

    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return None


def _worker_fn(args):
    """Worker function for multiprocessing."""
    video_path, output_dir, resolution, device, force_regenerate = args
    return generate_single_video_cache(
        video_path, output_dir, resolution, device, force_regenerate
    )


def generate_batch_cache(
    video_paths: List[str],
    output_dir: str,
    resolution: int = 512,
    device: str = "cuda",
    num_workers: int = 1,
    force_regenerate: bool = False,
) -> List[str]:
    """
    Generate preprocessing cache for multiple videos.

    Args:
        video_paths: List of video file paths
        output_dir: Directory to save cache files
        resolution: Target resolution
        device: Device for face detection
        num_workers: Number of parallel workers (GPU-bound, usually 1)
        force_regenerate: Regenerate cache even if exists

    Returns:
        List of successfully created cache paths
    """
    print(f"Processing {len(video_paths)} videos...")

    if num_workers <= 1:
        # Sequential processing
        results = []
        for video_path in video_paths:
            result = generate_single_video_cache(
                video_path, output_dir, resolution, device, force_regenerate
            )
            if result:
                results.append(result)
    else:
        # Parallel processing (note: GPU memory may be limiting factor)
        args_list = [
            (vp, output_dir, resolution, device, force_regenerate)
            for vp in video_paths
        ]
        with Pool(num_workers) as pool:
            results = [r for r in pool.map(_worker_fn, args_list) if r is not None]

    print(f"\nSuccessfully processed: {len(results)}/{len(video_paths)} videos")
    return results


def load_video_cache(cache_path: str) -> VideoPreprocessCache:
    """
    Load preprocessing cache from file.

    Args:
        cache_path: Path to cache file

    Returns:
        VideoPreprocessCache object
    """
    data = torch.load(cache_path, weights_only=False)
    return VideoPreprocessCache(**data)


def main():
    parser = argparse.ArgumentParser(
        description="Generate video preprocessing cache for distributed inference"
    )
    parser.add_argument(
        "--video_path",
        type=str,
        help="Path to single video file"
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        help="Directory containing videos (recursive search)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="cache/video",
        help="Output directory for cache files"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Target resolution (default: 512)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for face detection (default: cuda)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regenerate existing caches"
    )
    args = parser.parse_args()

    # Validate arguments
    if not args.video_path and not args.video_dir:
        parser.error("Either --video_path or --video_dir is required")

    if args.video_path and args.video_dir:
        parser.error("Cannot specify both --video_path and --video_dir")

    # Process videos
    if args.video_path:
        # Single video
        result = generate_single_video_cache(
            args.video_path,
            args.output_dir,
            args.resolution,
            args.device,
            args.force,
        )
        if result:
            print(f"\nCache file: {result}")
    else:
        # Batch processing
        video_paths = gather_video_paths_recursively(args.video_dir)
        if not video_paths:
            print(f"No videos found in {args.video_dir}")
            return

        results = generate_batch_cache(
            video_paths,
            args.output_dir,
            args.resolution,
            args.device,
            args.num_workers,
            args.force,
        )
        print(f"\nCache files created in: {args.output_dir}")


if __name__ == "__main__":
    main()
