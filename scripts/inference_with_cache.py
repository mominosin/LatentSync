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
Inference script with preprocessing cache support.

This script runs LatentSync inference using pre-computed face detection
and affine transformation cache, enabling faster processing when using
the same video with different audio tracks.

Usage:
    # With cache
    python scripts/inference_with_cache.py \
        --video_path assets/demo1_video.mp4 \
        --audio_path assets/demo1_audio.wav \
        --video_out_path output.mp4 \
        --cache_path cache/video/abc123_512.pt \
        --inference_ckpt_path checkpoints/latentsync_unet.pt

    # Without cache (auto-generate cache)
    python scripts/inference_with_cache.py \
        --video_path assets/demo1_video.mp4 \
        --audio_path assets/demo1_audio.wav \
        --video_out_path output.mp4 \
        --cache_dir cache/video \
        --inference_ckpt_path checkpoints/latentsync_unet.pt

    # With DeepCache for faster U-Net inference
    python scripts/inference_with_cache.py \
        --video_path assets/demo1_video.mp4 \
        --audio_path assets/demo1_audio.wav \
        --video_out_path output.mp4 \
        --cache_path cache/video/abc123_512.pt \
        --inference_ckpt_path checkpoints/latentsync_unet.pt \
        --enable_deepcache
"""

import argparse
import hashlib
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union, Callable

import numpy as np
import soundfile as sf
import torch
import torchvision
from torchvision import transforms
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler
from accelerate.utils import set_seed
from einops import rearrange
import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from latentsync.whisper.audio2feature import Audio2Feature
from latentsync.utils.util import read_video, read_audio, write_video, check_ffmpeg_installed
from latentsync.utils.image_processor import ImageProcessor, load_fixed_mask


def compute_video_hash(video_path: str) -> str:
    """Compute SHA256 hash of video file (first 16 chars)."""
    hasher = hashlib.sha256()
    with open(video_path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def find_cache_for_video(video_path: str, cache_dir: str, resolution: int) -> Optional[str]:
    """Find existing cache file for a video."""
    video_hash = compute_video_hash(video_path)
    cache_filename = f"{video_hash}_{resolution}.pt"
    cache_path = Path(cache_dir) / cache_filename
    if cache_path.exists():
        return str(cache_path)
    return None


def load_cache(cache_path: str) -> dict:
    """Load preprocessing cache from file."""
    return torch.load(cache_path, weights_only=False)


class CachedLipsyncPipeline(LipsyncPipeline):
    """Extended LipsyncPipeline with cache support."""

    def loop_video_with_cache(
        self,
        whisper_chunks: list,
        video_frames: np.ndarray,
        cached_faces: torch.Tensor,
        cached_boxes: list,
        cached_affine_matrices: list,
    ) -> Tuple[np.ndarray, torch.Tensor, list, list]:
        """
        Process video looping using cached face detection results.

        Args:
            whisper_chunks: Audio feature chunks
            video_frames: Original video frames
            cached_faces: Pre-computed face tensors
            cached_boxes: Pre-computed bounding boxes
            cached_affine_matrices: Pre-computed affine matrices

        Returns:
            Tuple of (video_frames, faces, boxes, affine_matrices)
        """
        num_video_frames = len(cached_faces)

        if len(whisper_chunks) > num_video_frames:
            # Audio is longer than video - need to loop
            num_loops = math.ceil(len(whisper_chunks) / num_video_frames)
            loop_video_frames = []
            loop_faces = []
            loop_boxes = []
            loop_affine_matrices = []

            for i in range(num_loops):
                if i % 2 == 0:
                    # Forward
                    loop_video_frames.append(video_frames)
                    loop_faces.append(cached_faces)
                    loop_boxes += list(cached_boxes)
                    loop_affine_matrices += list(cached_affine_matrices)
                else:
                    # Backward (ping-pong)
                    loop_video_frames.append(video_frames[::-1])
                    loop_faces.append(cached_faces.flip(0))
                    loop_boxes += list(reversed(cached_boxes))
                    loop_affine_matrices += list(reversed(cached_affine_matrices))

            video_frames = np.concatenate(loop_video_frames, axis=0)[: len(whisper_chunks)]
            faces = torch.cat(loop_faces, dim=0)[: len(whisper_chunks)]
            boxes = loop_boxes[: len(whisper_chunks)]
            affine_matrices = loop_affine_matrices[: len(whisper_chunks)]
        else:
            # Video is longer or equal - just trim
            video_frames = video_frames[: len(whisper_chunks)]
            faces = cached_faces[: len(whisper_chunks)]
            boxes = list(cached_boxes)[: len(whisper_chunks)]
            affine_matrices = list(cached_affine_matrices)[: len(whisper_chunks)]

        # Ensure affine_matrices are on GPU for restore_video
        affine_matrices = [m.to("cuda") if isinstance(m, torch.Tensor) else m for m in affine_matrices]

        return video_frames, faces, boxes, affine_matrices

    @torch.no_grad()
    def __call__(
        self,
        video_path: str,
        audio_path: str,
        video_out_path: str,
        num_frames: int = 16,
        video_fps: int = 25,
        audio_sample_rate: int = 16000,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.5,
        weight_dtype: Optional[torch.dtype] = torch.float16,
        eta: float = 0.0,
        mask_image_path: str = "latentsync/utils/mask.png",
        temp_dir: str = "temp",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        # Cache parameters
        cache_path: Optional[str] = None,
        cache_data: Optional[dict] = None,
        **kwargs,
    ):
        """
        Run inference with optional cache support.

        Additional Args:
            cache_path: Path to preprocessing cache file
            cache_data: Pre-loaded cache data (alternative to cache_path)
        """
        is_train = self.unet.training
        self.unet.eval()

        check_ffmpeg_installed()

        # 0. Define call parameters
        device = self._execution_device
        mask_image = load_fixed_mask(height, mask_image_path)
        self.image_processor = ImageProcessor(height, device="cuda", mask_image=mask_image)
        self.set_progress_bar_config(desc=f"Sample frames: {num_frames}")

        # 1. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 2. Check inputs
        self.check_inputs(height, width, callback_steps)

        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 4. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 5. Process audio
        whisper_feature = self.audio_encoder.audio2feat(audio_path)
        whisper_chunks = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=video_fps)

        audio_samples = read_audio(audio_path)
        video_frames = read_video(video_path, use_decord=False)

        # 6. Process video with or without cache
        if cache_data is not None:
            print("Using provided cache data...")
            cached_faces = cache_data['faces']
            cached_boxes = cache_data['boxes']
            cached_affine_matrices = cache_data['affine_matrices']
            video_frames, faces, boxes, affine_matrices = self.loop_video_with_cache(
                whisper_chunks, video_frames,
                cached_faces, cached_boxes, cached_affine_matrices
            )
        elif cache_path is not None and os.path.exists(cache_path):
            print(f"Loading cache from: {cache_path}")
            cache = load_cache(cache_path)
            cached_faces = cache['faces']
            cached_boxes = cache['boxes']
            cached_affine_matrices = cache['affine_matrices']
            video_frames, faces, boxes, affine_matrices = self.loop_video_with_cache(
                whisper_chunks, video_frames,
                cached_faces, cached_boxes, cached_affine_matrices
            )
        else:
            print("No cache found, running face detection...")
            video_frames, faces, boxes, affine_matrices = self.loop_video(whisper_chunks, video_frames)

        synced_video_frames = []
        num_channels_latents = self.vae.config.latent_channels

        # Prepare latent variables
        all_latents = self.prepare_latents(
            len(whisper_chunks),
            num_channels_latents,
            height,
            width,
            weight_dtype,
            device,
            generator,
        )

        num_inferences = math.ceil(len(whisper_chunks) / num_frames)
        for i in tqdm.tqdm(range(num_inferences), desc="Doing inference..."):
            if self.unet.add_audio_layer:
                audio_embeds = torch.stack(whisper_chunks[i * num_frames : (i + 1) * num_frames])
                audio_embeds = audio_embeds.to(device, dtype=weight_dtype)
                if do_classifier_free_guidance:
                    null_audio_embeds = torch.zeros_like(audio_embeds)
                    audio_embeds = torch.cat([null_audio_embeds, audio_embeds])
            else:
                audio_embeds = None

            inference_faces = faces[i * num_frames : (i + 1) * num_frames]
            latents = all_latents[:, :, i * num_frames : (i + 1) * num_frames]
            ref_pixel_values, masked_pixel_values, masks = self.image_processor.prepare_masks_and_masked_images(
                inference_faces, affine_transform=False
            )

            # 7. Prepare mask latent variables
            mask_latents, masked_image_latents = self.prepare_mask_latents(
                masks,
                masked_pixel_values,
                height,
                width,
                weight_dtype,
                device,
                generator,
                do_classifier_free_guidance,
            )

            # 8. Prepare image latents
            ref_latents = self.prepare_image_latents(
                ref_pixel_values,
                device,
                weight_dtype,
                generator,
                do_classifier_free_guidance,
            )

            # 9. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for j, t in enumerate(timesteps):
                    unet_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    unet_input = self.scheduler.scale_model_input(unet_input, t)
                    unet_input = torch.cat([unet_input, mask_latents, masked_image_latents, ref_latents], dim=1)

                    noise_pred = self.unet(unet_input, t, encoder_hidden_states=audio_embeds).sample

                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_audio - noise_pred_uncond)

                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    if j == len(timesteps) - 1 or ((j + 1) > num_warmup_steps and (j + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and j % callback_steps == 0:
                            callback(j, t, latents)

            # Recover the pixel values
            decoded_latents = self.decode_latents(latents)
            decoded_latents = self.paste_surrounding_pixels_back(
                decoded_latents, ref_pixel_values, 1 - masks, device, weight_dtype
            )
            synced_video_frames.append(decoded_latents)

        synced_video_frames = self.restore_video(torch.cat(synced_video_frames), video_frames, boxes, affine_matrices)

        audio_samples_remain_length = int(synced_video_frames.shape[0] / video_fps * audio_sample_rate)
        audio_samples = audio_samples[:audio_samples_remain_length].cpu().numpy()

        if is_train:
            self.unet.train()

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        write_video(os.path.join(temp_dir, "video.mp4"), synced_video_frames, fps=video_fps)
        sf.write(os.path.join(temp_dir, "audio.wav"), audio_samples, audio_sample_rate)

        command = f"ffmpeg -y -loglevel error -nostdin -i {os.path.join(temp_dir, 'video.mp4')} -i {os.path.join(temp_dir, 'audio.wav')} -c:v libx264 -crf 18 -c:a aac -q:v 0 -q:a 0 {video_out_path}"
        subprocess.run(command, shell=True)


def main(config, args):
    if not os.path.exists(args.video_path):
        raise RuntimeError(f"Video path '{args.video_path}' not found")
    if not os.path.exists(args.audio_path):
        raise RuntimeError(f"Audio path '{args.audio_path}' not found")

    # Check GPU and dtype
    is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    dtype = torch.float16 if is_fp16_supported else torch.float32

    print(f"Input video path: {args.video_path}")
    print(f"Input audio path: {args.audio_path}")
    print(f"Loaded checkpoint path: {args.inference_ckpt_path}")

    # Determine cache path
    cache_path = args.cache_path
    if cache_path is None and args.cache_dir:
        cache_path = find_cache_for_video(args.video_path, args.cache_dir, config.data.resolution)
        if cache_path:
            print(f"Found cache: {cache_path}")

    if cache_path:
        print(f"Using cache: {cache_path}")
    else:
        print("No cache found, will run face detection during inference")

    # Initialize models
    scheduler = DDIMScheduler.from_pretrained("configs")

    if config.model.cross_attention_dim == 768:
        whisper_model_path = "checkpoints/whisper/small.pt"
    elif config.model.cross_attention_dim == 384:
        whisper_model_path = "checkpoints/whisper/tiny.pt"
    else:
        raise NotImplementedError("cross_attention_dim must be 768 or 384")

    audio_encoder = Audio2Feature(
        model_path=whisper_model_path,
        device="cuda",
        num_frames=config.data.num_frames,
        audio_feat_length=config.data.audio_feat_length,
        audio_embeds_cache_dir=args.audio_cache_dir,
    )

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        args.inference_ckpt_path,
        device="cpu",
    )
    unet = unet.to(dtype=dtype)

    # Use CachedLipsyncPipeline instead of LipsyncPipeline
    pipeline = CachedLipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=scheduler,
    ).to("cuda")

    # Enable DeepCache if requested
    if args.enable_deepcache:
        from DeepCache import DeepCacheSDHelper
        helper = DeepCacheSDHelper(pipe=pipeline)
        helper.set_params(cache_interval=3, cache_branch_id=0)
        helper.enable()
        print("DeepCache enabled")

    # Set seed
    if args.seed != -1:
        set_seed(args.seed)
    else:
        torch.seed()

    print(f"Initial seed: {torch.initial_seed()}")

    # Run inference
    pipeline(
        video_path=args.video_path,
        audio_path=args.audio_path,
        video_out_path=args.video_out_path,
        num_frames=config.data.num_frames,
        num_inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        weight_dtype=dtype,
        width=config.data.resolution,
        height=config.data.resolution,
        mask_image_path=config.data.mask_image_path,
        temp_dir=args.temp_dir,
        cache_path=cache_path,
    )

    print(f"Output saved to: {args.video_out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LatentSync inference with cache support")
    parser.add_argument("--unet_config_path", type=str, default="configs/unet/stage2_512.yaml")
    parser.add_argument("--inference_ckpt_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--temp_dir", type=str, default="temp")
    parser.add_argument("--seed", type=int, default=1247)
    parser.add_argument("--enable_deepcache", action="store_true")
    # Cache options
    parser.add_argument("--cache_path", type=str, help="Path to video preprocessing cache file")
    parser.add_argument("--cache_dir", type=str, help="Directory to search for cache files")
    parser.add_argument("--audio_cache_dir", type=str, default="", help="Directory for audio embedding cache")
    args = parser.parse_args()

    config = OmegaConf.load(args.unet_config_path)
    main(config, args)
