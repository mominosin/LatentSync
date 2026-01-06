#!/usr/bin/env python3
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""
SQS Worker - Process lipsync jobs from the queue.

This script runs on each EC2 GPU instance and continuously polls
the SQS queue for jobs to process.

Usage:
    # Run worker (will continuously poll for jobs)
    python scripts/sqs_worker.py \
        --queue-url https://sqs.ap-northeast-1.amazonaws.com/123456789/latentsync-jobs \
        --work-dir /tmp/latentsync \
        --checkpoint checkpoints/latentsync_unet.pt

    # Run with systemd (see sqs_worker.service)
    sudo systemctl start latentsync-worker
"""

import argparse
import json
import os
import shutil
import signal
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import boto3

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class GracefulShutdown:
    """Handle graceful shutdown on SIGTERM/SIGINT."""
    shutdown_requested = False

    def __init__(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        self.shutdown_requested = True


def parse_s3_path(s3_path: str) -> tuple:
    """Parse s3://bucket/key into (bucket, key)."""
    parsed = urlparse(s3_path)
    if parsed.scheme != "s3":
        raise ValueError(f"Invalid S3 path: {s3_path}")
    return parsed.netloc, parsed.path.lstrip("/")


def download_from_s3(s3_client, s3_path: str, local_path: str) -> str:
    """Download file from S3 to local path."""
    bucket, key = parse_s3_path(s3_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    print(f"Downloading {s3_path} -> {local_path}")
    s3_client.download_file(bucket, key, local_path)
    return local_path


def upload_to_s3(s3_client, local_path: str, s3_path: str):
    """Upload local file to S3."""
    bucket, key = parse_s3_path(s3_path)
    print(f"Uploading {local_path} -> {s3_path}")
    s3_client.upload_file(local_path, bucket, key)


def check_s3_exists(s3_client, s3_path: str) -> bool:
    """Check if S3 object exists."""
    bucket, key = parse_s3_path(s3_path)
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except s3_client.exceptions.ClientError:
        return False


def get_cache_s3_path(video_s3_path: str, cache_s3_prefix: str, resolution: int = 512) -> str:
    """Generate cache S3 path based on video filename."""
    # Use video filename as base for cache
    video_filename = os.path.basename(video_s3_path)
    video_name = os.path.splitext(video_filename)[0]
    return f"{cache_s3_prefix}/{video_name}_{resolution}.pt"


def process_job(job: dict, args, s3_client) -> bool:
    """
    Process a single lipsync job.

    Returns True if successful, False otherwise.
    """
    job_id = job["job_id"]
    params = job["params"]

    print(f"\n{'='*60}")
    print(f"Processing job: {job_id}")
    print(f"Video: {params['video_s3_path']}")
    print(f"Audio: {params['audio_s3_path']}")
    print(f"Output: {params['output_s3_path']}")
    print(f"{'='*60}\n")

    # Create job-specific work directory
    job_work_dir = os.path.join(args.work_dir, job_id)
    os.makedirs(job_work_dir, exist_ok=True)

    try:
        # 1. Download input files from S3
        local_video = os.path.join(job_work_dir, "input_video.mp4")
        local_audio = os.path.join(job_work_dir, "input_audio.wav")
        local_output = os.path.join(job_work_dir, "output.mp4")

        download_from_s3(s3_client, params["video_s3_path"], local_video)
        download_from_s3(s3_client, params["audio_s3_path"], local_audio)

        # 2. Check/download cache if exists
        cache_s3_path = get_cache_s3_path(
            params["video_s3_path"],
            params["cache_s3_prefix"]
        )
        local_cache_dir = os.path.join(job_work_dir, "cache")
        os.makedirs(local_cache_dir, exist_ok=True)
        local_cache_path = None

        if check_s3_exists(s3_client, cache_s3_path):
            local_cache_path = os.path.join(local_cache_dir, "video_cache.pt")
            download_from_s3(s3_client, cache_s3_path, local_cache_path)
            print(f"Using cached preprocessing from {cache_s3_path}")

        # 3. Run inference
        from scripts.inference_with_cache import main as run_inference, CachedLipsyncPipeline
        from omegaconf import OmegaConf

        # Build args namespace
        class InferenceArgs:
            unet_config_path = "configs/unet/stage2_512.yaml"
            inference_ckpt_path = args.checkpoint
            video_path = local_video
            audio_path = local_audio
            video_out_path = local_output
            inference_steps = params.get("inference_steps", 20)
            guidance_scale = 1.0
            temp_dir = os.path.join(job_work_dir, "temp")
            seed = 1247
            enable_deepcache = params.get("enable_deepcache", True)
            cache_path = local_cache_path
            cache_dir = None
            audio_cache_dir = ""

        config = OmegaConf.load(InferenceArgs.unet_config_path)
        run_inference(config, InferenceArgs())

        # 4. Upload output to S3
        if os.path.exists(local_output):
            upload_to_s3(s3_client, local_output, params["output_s3_path"])
            print(f"Output uploaded to {params['output_s3_path']}")

            # 5. Upload cache if newly generated
            if local_cache_path is None:
                # Find generated cache file
                cache_files = list(Path(local_cache_dir).glob("*.pt"))
                if cache_files:
                    upload_to_s3(s3_client, str(cache_files[0]), cache_s3_path)
                    print(f"Cache uploaded to {cache_s3_path}")

            return True
        else:
            print(f"ERROR: Output file not generated")
            return False

    except Exception as e:
        print(f"ERROR processing job {job_id}: {e}")
        traceback.print_exc()
        return False

    finally:
        # Cleanup job work directory
        if os.path.exists(job_work_dir) and args.cleanup:
            shutil.rmtree(job_work_dir)
            print(f"Cleaned up {job_work_dir}")


def worker_loop(args):
    """Main worker loop - continuously poll SQS for jobs."""
    shutdown = GracefulShutdown()

    sqs = boto3.client("sqs", region_name=args.region)
    s3 = boto3.client("s3", region_name=args.region)

    print(f"Worker started")
    print(f"Queue URL: {args.queue_url}")
    print(f"Work directory: {args.work_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Polling interval: {args.poll_interval}s")
    print()

    # Create work directory
    os.makedirs(args.work_dir, exist_ok=True)

    jobs_processed = 0
    jobs_failed = 0

    while not shutdown.shutdown_requested:
        try:
            # Poll for messages
            response = sqs.receive_message(
                QueueUrl=args.queue_url,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=args.poll_interval,
                VisibilityTimeout=args.visibility_timeout,
            )

            messages = response.get("Messages", [])

            if not messages:
                continue

            for message in messages:
                if shutdown.shutdown_requested:
                    break

                try:
                    job = json.loads(message["Body"])

                    # Process the job
                    success = process_job(job, args, s3)

                    if success:
                        jobs_processed += 1
                        # Delete message from queue on success
                        sqs.delete_message(
                            QueueUrl=args.queue_url,
                            ReceiptHandle=message["ReceiptHandle"]
                        )
                        print(f"Job completed successfully. Total: {jobs_processed}")
                    else:
                        jobs_failed += 1
                        # Message will return to queue after visibility timeout
                        print(f"Job failed. Will retry. Total failed: {jobs_failed}")

                except json.JSONDecodeError as e:
                    print(f"Invalid message format: {e}")
                    # Delete malformed messages
                    sqs.delete_message(
                        QueueUrl=args.queue_url,
                        ReceiptHandle=message["ReceiptHandle"]
                    )

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Worker error: {e}")
            traceback.print_exc()
            time.sleep(5)  # Back off on error

    print(f"\nWorker shutdown complete")
    print(f"Jobs processed: {jobs_processed}")
    print(f"Jobs failed: {jobs_failed}")


def main():
    parser = argparse.ArgumentParser(description="SQS Worker for LatentSync")
    parser.add_argument("--queue-url", type=str, required=True, help="SQS queue URL")
    parser.add_argument("--work-dir", type=str, default="/tmp/latentsync", help="Working directory")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latentsync_unet.pt")
    parser.add_argument("--region", type=str, default="ap-northeast-1")
    parser.add_argument("--poll-interval", type=int, default=20, help="SQS long-polling interval")
    parser.add_argument("--visibility-timeout", type=int, default=3600, help="Message visibility timeout (seconds)")
    parser.add_argument("--no-cleanup", dest="cleanup", action="store_false", help="Don't cleanup work directory")
    args = parser.parse_args()

    worker_loop(args)


if __name__ == "__main__":
    main()
