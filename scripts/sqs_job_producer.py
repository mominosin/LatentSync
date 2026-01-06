# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""
SQS Job Producer - Submit lipsync jobs to the queue.

Usage:
    # Submit a single job
    python scripts/sqs_job_producer.py \
        --queue-url https://sqs.ap-northeast-1.amazonaws.com/123456789/latentsync-jobs \
        --video-s3 s3://my-bucket/input/videos/video1.mp4 \
        --audio-s3 s3://my-bucket/input/audios/audio1.wav \
        --output-s3 s3://my-bucket/output/result1.mp4

    # Submit batch jobs from JSON file
    python scripts/sqs_job_producer.py \
        --queue-url https://sqs.ap-northeast-1.amazonaws.com/123456789/latentsync-jobs \
        --batch-file jobs.json
"""

import argparse
import json
import uuid
from datetime import datetime
from typing import Optional

import boto3


def create_job_message(
    video_s3_path: str,
    audio_s3_path: str,
    output_s3_path: str,
    cache_s3_prefix: str = "s3://my-bucket/cache/video",
    job_id: Optional[str] = None,
    enable_deepcache: bool = True,
    inference_steps: int = 20,
) -> dict:
    """Create a job message for SQS."""
    if job_id is None:
        job_id = str(uuid.uuid4())[:8]

    return {
        "job_id": job_id,
        "created_at": datetime.utcnow().isoformat(),
        "job_type": "lipsync_inference",
        "params": {
            "video_s3_path": video_s3_path,
            "audio_s3_path": audio_s3_path,
            "output_s3_path": output_s3_path,
            "cache_s3_prefix": cache_s3_prefix,
            "enable_deepcache": enable_deepcache,
            "inference_steps": inference_steps,
        }
    }


def submit_job(sqs_client, queue_url: str, job: dict) -> str:
    """Submit a job to SQS queue."""
    response = sqs_client.send_message(
        QueueUrl=queue_url,
        MessageBody=json.dumps(job),
        MessageGroupId="lipsync-jobs",  # For FIFO queues
        MessageDeduplicationId=job["job_id"],  # For FIFO queues
    )
    return response["MessageId"]


def submit_jobs_batch(sqs_client, queue_url: str, jobs: list) -> list:
    """Submit multiple jobs in batch (up to 10 at a time)."""
    results = []
    for i in range(0, len(jobs), 10):
        batch = jobs[i:i+10]
        entries = [
            {
                "Id": job["job_id"],
                "MessageBody": json.dumps(job),
                "MessageGroupId": "lipsync-jobs",
                "MessageDeduplicationId": job["job_id"],
            }
            for job in batch
        ]
        response = sqs_client.send_message_batch(
            QueueUrl=queue_url,
            Entries=entries
        )
        results.extend(response.get("Successful", []))
    return results


def main():
    parser = argparse.ArgumentParser(description="Submit lipsync jobs to SQS")
    parser.add_argument("--queue-url", type=str, required=True, help="SQS queue URL")
    parser.add_argument("--video-s3", type=str, help="S3 path to video file")
    parser.add_argument("--audio-s3", type=str, help="S3 path to audio file")
    parser.add_argument("--output-s3", type=str, help="S3 path for output")
    parser.add_argument("--cache-s3-prefix", type=str, default="s3://my-bucket/cache/video")
    parser.add_argument("--batch-file", type=str, help="JSON file with batch jobs")
    parser.add_argument("--enable-deepcache", action="store_true", default=True)
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--region", type=str, default="ap-northeast-1")
    args = parser.parse_args()

    sqs = boto3.client("sqs", region_name=args.region)

    if args.batch_file:
        # Batch submission
        with open(args.batch_file) as f:
            batch_data = json.load(f)

        jobs = []
        for item in batch_data:
            job = create_job_message(
                video_s3_path=item["video_s3"],
                audio_s3_path=item["audio_s3"],
                output_s3_path=item["output_s3"],
                cache_s3_prefix=args.cache_s3_prefix,
                enable_deepcache=args.enable_deepcache,
                inference_steps=args.inference_steps,
            )
            jobs.append(job)

        results = submit_jobs_batch(sqs, args.queue_url, jobs)
        print(f"Submitted {len(results)} jobs")

    else:
        # Single job submission
        if not all([args.video_s3, args.audio_s3, args.output_s3]):
            parser.error("--video-s3, --audio-s3, and --output-s3 are required for single job")

        job = create_job_message(
            video_s3_path=args.video_s3,
            audio_s3_path=args.audio_s3,
            output_s3_path=args.output_s3,
            cache_s3_prefix=args.cache_s3_prefix,
            enable_deepcache=args.enable_deepcache,
            inference_steps=args.inference_steps,
        )

        message_id = submit_job(sqs, args.queue_url, job)
        print(f"Submitted job {job['job_id']} (MessageId: {message_id})")
        print(json.dumps(job, indent=2))


if __name__ == "__main__":
    main()
