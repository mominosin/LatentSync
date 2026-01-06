# LatentSync 分散実行ガイド

このガイドでは、LatentSyncを複数のEC2インスタンスで分散実行するための手順を説明します。

## 目次

1. [概要](#概要)
2. [アーキテクチャ](#アーキテクチャ)
3. [セットアップ](#セットアップ)
4. [キャッシュ生成](#キャッシュ生成)
5. [推論実行](#推論実行)
6. [AWS構成例](#aws構成例)
7. [パフォーマンス最適化](#パフォーマンス最適化)

---

## 概要

### 課題
- 1分の動画に対して約50分の処理時間
- U-Net推論が全体の80%（約40分）を占めるボトルネック
- 同じ動画に異なる音声を合成する場合、前処理が毎回実行される

### 解決策
- 動画の前処理結果（顔検出、アフィン変換行列）をキャッシュ
- キャッシュをS3/NFSで共有し、推論インスタンスで再利用
- 同じ動画への複数音声合成時に大幅な時間短縮

### 期待効果
| シナリオ | 処理時間 | 改善率 |
|---------|---------|--------|
| 初回実行（キャッシュなし） | ~50分 | - |
| 2回目以降（キャッシュあり） | ~42分 | 16%削減 |
| 前処理のみ | ~45秒→2秒 | 95%削減 |

---

## アーキテクチャ

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client / API                            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Job Queue (SQS)                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Preprocess  │  │ Inference   │  │ Inference   │             │
│  │ Job         │  │ Job (v1,a1) │  │ Job (v1,a2) │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ Preprocess    │   │ Inference     │   │ Inference     │
│ Worker (GPU)  │   │ Worker (GPU)  │   │ Worker (GPU)  │
│               │   │               │   │               │
│ - Face Detect │   │ - Load Cache  │   │ - Load Cache  │
│ - Affine Tx   │   │ - U-Net       │   │ - U-Net       │
│ - Save Cache  │   │ - VAE         │   │ - VAE         │
└───────────────┘   └───────────────┘   └───────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
              ┌─────────────────────────┐
              │   Shared Storage        │
              │   (S3 / EFS / FSx)      │
              │                         │
              │  /cache/video/          │
              │    ├── abc123_512.pt    │
              │    └── def456_512.pt    │
              │  /cache/audio/          │
              │    └── audio1_embeds.pt │
              │  /output/               │
              │    └── result.mp4       │
              └─────────────────────────┘
```

---

## セットアップ

### 1. 必要なファイル構成

```
LatentSync/
├── scripts/
│   ├── generate_video_cache.py    # キャッシュ生成
│   ├── inference_with_cache.py    # キャッシュ利用推論
│   └── inference.py               # 通常推論
├── cache/
│   ├── video/                     # 動画キャッシュ
│   └── audio/                     # 音声キャッシュ
└── checkpoints/                   # モデルファイル
```

### 2. 依存関係のインストール

```bash
# 基本環境
pip install torch torchvision torchaudio
pip install diffusers accelerate omegaconf
pip install opencv-python insightface kornia
pip install soundfile decord imageio

# オプション: DeepCache（U-Net高速化）
pip install DeepCache
```

### 3. 共有ストレージのマウント

```bash
# S3 (s3fs)
s3fs your-bucket /mnt/shared -o iam_role=auto

# EFS
sudo mount -t efs fs-xxxxx:/ /mnt/shared

# FSx for Lustre
sudo mount -t lustre fs-xxxxx.fsx.region.amazonaws.com@tcp:/fsx /mnt/shared
```

---

## キャッシュ生成

### 単一動画のキャッシュ生成

```bash
python scripts/generate_video_cache.py \
    --video_path /path/to/video.mp4 \
    --output_dir /mnt/shared/cache/video \
    --resolution 512 \
    --device cuda
```

### バッチキャッシュ生成

```bash
# ディレクトリ内の全動画を処理
python scripts/generate_video_cache.py \
    --video_dir /path/to/videos \
    --output_dir /mnt/shared/cache/video \
    --resolution 512 \
    --num_workers 1
```

### キャッシュファイルの構造

```python
# キャッシュファイル名: {video_hash}_{resolution}.pt
# 例: abc123def456_512.pt

{
    'video_hash': 'abc123def456',      # 動画のSHA256ハッシュ（先頭16文字）
    'video_path': '/path/to/video.mp4', # 元の動画パス
    'num_frames': 1500,                 # フレーム数
    'resolution': 512,                  # 解像度
    'faces': torch.Tensor,              # shape=(N, 3, 512, 512), dtype=uint8
    'boxes': [(x1,y1,x2,y2), ...],      # N個のバウンディングボックス
    'affine_matrices': [tensor, ...],   # N個のアフィン変換行列
}
```

### 推定キャッシュサイズ

| 動画長 | フレーム数 | キャッシュサイズ |
|--------|-----------|----------------|
| 30秒   | 750       | ~600MB         |
| 1分    | 1500      | ~1.2GB         |
| 2分    | 3000      | ~2.4GB         |

---

## 推論実行

### キャッシュを使用した推論

```bash
# キャッシュパスを直接指定
python scripts/inference_with_cache.py \
    --video_path /path/to/video.mp4 \
    --audio_path /path/to/audio.wav \
    --video_out_path /path/to/output.mp4 \
    --inference_ckpt_path checkpoints/latentsync_unet.pt \
    --cache_path /mnt/shared/cache/video/abc123_512.pt
```

```bash
# キャッシュディレクトリから自動検索
python scripts/inference_with_cache.py \
    --video_path /path/to/video.mp4 \
    --audio_path /path/to/audio.wav \
    --video_out_path /path/to/output.mp4 \
    --inference_ckpt_path checkpoints/latentsync_unet.pt \
    --cache_dir /mnt/shared/cache/video
```

### DeepCacheを有効化（U-Net 20-30%高速化）

```bash
python scripts/inference_with_cache.py \
    --video_path /path/to/video.mp4 \
    --audio_path /path/to/audio.wav \
    --video_out_path /path/to/output.mp4 \
    --inference_ckpt_path checkpoints/latentsync_unet.pt \
    --cache_path /mnt/shared/cache/video/abc123_512.pt \
    --enable_deepcache
```

### 音声キャッシュの有効化

```bash
# 音声のWhisper埋め込みもキャッシュ
python scripts/inference_with_cache.py \
    --video_path /path/to/video.mp4 \
    --audio_path /path/to/audio.wav \
    --video_out_path /path/to/output.mp4 \
    --inference_ckpt_path checkpoints/latentsync_unet.pt \
    --cache_dir /mnt/shared/cache/video \
    --audio_cache_dir /mnt/shared/cache/audio
```

---

## AWS構成例

### EC2インスタンス推奨構成

| 用途 | インスタンスタイプ | GPU | VRAM | 備考 |
|------|-------------------|-----|------|------|
| 前処理 | g4dn.xlarge | T4 | 16GB | コスト効率重視 |
| 推論 | g5.xlarge | A10G | 24GB | バランス型 |
| 推論（高速） | g5.2xlarge | A10G | 24GB | より高速 |
| 推論（大規模） | p4d.24xlarge | A100 | 40GB | 最高性能 |

### Terraform構成例

```hcl
# main.tf
resource "aws_efs_file_system" "cache" {
  creation_token = "latentsync-cache"
  performance_mode = "generalPurpose"
  throughput_mode  = "bursting"
}

resource "aws_launch_template" "preprocess_worker" {
  name_prefix   = "latentsync-preprocess-"
  image_id      = "ami-xxxxx"  # Deep Learning AMI
  instance_type = "g4dn.xlarge"

  user_data = base64encode(<<-EOF
    #!/bin/bash
    mount -t efs ${aws_efs_file_system.cache.id}:/ /mnt/shared
    cd /opt/LatentSync
    python scripts/generate_video_cache.py \
      --video_dir /mnt/shared/input \
      --output_dir /mnt/shared/cache/video
  EOF
  )
}

resource "aws_launch_template" "inference_worker" {
  name_prefix   = "latentsync-inference-"
  image_id      = "ami-xxxxx"
  instance_type = "g5.xlarge"

  user_data = base64encode(<<-EOF
    #!/bin/bash
    mount -t efs ${aws_efs_file_system.cache.id}:/ /mnt/shared
    # SQSからジョブを取得して処理
  EOF
  )
}
```

### SQSジョブキュー設計

```python
# job_producer.py
import boto3
import json

sqs = boto3.client('sqs')
queue_url = 'https://sqs.region.amazonaws.com/account/latentsync-jobs'

def submit_inference_job(video_path, audio_path, output_path, cache_path=None):
    job = {
        'type': 'inference',
        'video_path': video_path,
        'audio_path': audio_path,
        'output_path': output_path,
        'cache_path': cache_path,
    }
    sqs.send_message(
        QueueUrl=queue_url,
        MessageBody=json.dumps(job)
    )
```

```python
# job_consumer.py (ワーカー側)
import boto3
import json
import subprocess

sqs = boto3.client('sqs')
queue_url = 'https://sqs.region.amazonaws.com/account/latentsync-jobs'

while True:
    response = sqs.receive_message(QueueUrl=queue_url, WaitTimeSeconds=20)
    for message in response.get('Messages', []):
        job = json.loads(message['Body'])

        if job['type'] == 'inference':
            cmd = [
                'python', 'scripts/inference_with_cache.py',
                '--video_path', job['video_path'],
                '--audio_path', job['audio_path'],
                '--video_out_path', job['output_path'],
                '--cache_dir', '/mnt/shared/cache/video',
                '--inference_ckpt_path', 'checkpoints/latentsync_unet.pt',
            ]
            subprocess.run(cmd)

        sqs.delete_message(
            QueueUrl=queue_url,
            ReceiptHandle=message['ReceiptHandle']
        )
```

---

## パフォーマンス最適化

### 1. DeepCacheの活用

DeepCacheはU-Net推論を20-30%高速化します。

```bash
python scripts/inference_with_cache.py \
    ... \
    --enable_deepcache
```

### 2. 推論ステップ数の調整

デフォルトは20ステップですが、品質とのトレードオフで調整可能。

```bash
# 高速（品質低下あり）
--inference_steps 10

# 高品質（処理時間増加）
--inference_steps 30
```

### 3. バッチ処理の最適化

同じ動画に対する複数音声の処理は、キャッシュを一度読み込んでバッチ処理すると効率的。

```python
# batch_inference.py
from scripts.inference_with_cache import CachedLipsyncPipeline, load_cache

# キャッシュを一度だけ読み込み
cache_data = load_cache('/mnt/shared/cache/video/abc123_512.pt')

# 複数の音声に対して推論
for audio_path in audio_paths:
    pipeline(
        video_path=video_path,
        audio_path=audio_path,
        video_out_path=f'output_{audio_path}.mp4',
        cache_data=cache_data,  # キャッシュデータを直接渡す
    )
```

### 4. ストレージ選択

| ストレージ | スループット | レイテンシ | コスト | 推奨用途 |
|-----------|-------------|-----------|-------|---------|
| S3 | 高 | 中 | 低 | 大量の動画アーカイブ |
| EFS | 中 | 低 | 中 | 一般的なワークロード |
| FSx Lustre | 非常に高 | 非常に低 | 高 | 高頻度アクセス |
| Instance Store | 最高 | 最低 | 含む | 一時キャッシュ |

### 5. GPU メモリ最適化

```bash
# float16 を使用（デフォルト）
# VRAM 使用量: ~12GB

# float32 を強制（デバッグ用）
# VRAM 使用量: ~24GB
```

---

## トラブルシューティング

### キャッシュが見つからない

```bash
# キャッシュファイル名はビデオのハッシュ値に基づく
# ファイル名を確認
ls -la /mnt/shared/cache/video/

# ハッシュ値を確認
python -c "
from scripts.generate_video_cache import compute_video_hash
print(compute_video_hash('/path/to/video.mp4'))
"
```

### GPU メモリ不足

```bash
# 1. DeepCache を無効化
# 2. バッチサイズ（num_frames）を削減
python scripts/inference_with_cache.py \
    --unet_config_path configs/unet_small.yaml \
    ...
```

### 顔検出失敗

```bash
# ログを確認
# 連続して失敗する場合は動画品質を確認
# 解像度が低すぎる、または顔が小さすぎる可能性
```

---

## SQS分散実行の詳細

### ファイル構成

```
scripts/
├── sqs_job_producer.py    # ジョブ投入スクリプト
├── sqs_worker.py          # ワーカースクリプト（各EC2で実行）
└── sqs_worker.service     # systemdサービスファイル
```

### Step 1: S3バケット準備

```bash
# バケット作成
aws s3 mb s3://my-latentsync-bucket

# ディレクトリ構造
s3://my-latentsync-bucket/
├── input/
│   ├── videos/      # 入力動画
│   └── audios/      # 入力音声
├── cache/
│   └── video/       # 前処理キャッシュ
└── output/          # 出力動画
```

### Step 2: SQSキュー作成

```bash
# FIFOキュー作成（順序保証・重複排除あり）
aws sqs create-queue \
    --queue-name latentsync-jobs.fifo \
    --attributes '{
        "FifoQueue": "true",
        "ContentBasedDeduplication": "false",
        "VisibilityTimeout": "3600",
        "MessageRetentionPeriod": "86400"
    }'
```

### Step 3: ジョブ投入

```bash
# 単一ジョブ投入
python scripts/sqs_job_producer.py \
    --queue-url https://sqs.ap-northeast-1.amazonaws.com/123456789/latentsync-jobs.fifo \
    --video-s3 s3://my-latentsync-bucket/input/videos/video1.mp4 \
    --audio-s3 s3://my-latentsync-bucket/input/audios/audio1.wav \
    --output-s3 s3://my-latentsync-bucket/output/result1.mp4 \
    --cache-s3-prefix s3://my-latentsync-bucket/cache/video
```

```bash
# バッチ投入（JSONファイルから）
cat > jobs.json << 'EOF'
[
  {
    "video_s3": "s3://my-bucket/input/videos/video1.mp4",
    "audio_s3": "s3://my-bucket/input/audios/audio1.wav",
    "output_s3": "s3://my-bucket/output/result1.mp4"
  },
  {
    "video_s3": "s3://my-bucket/input/videos/video1.mp4",
    "audio_s3": "s3://my-bucket/input/audios/audio2.wav",
    "output_s3": "s3://my-bucket/output/result2.mp4"
  }
]
EOF

python scripts/sqs_job_producer.py \
    --queue-url https://sqs.ap-northeast-1.amazonaws.com/123456789/latentsync-jobs.fifo \
    --batch-file jobs.json
```

### Step 4: EC2ワーカー起動

```bash
# 手動実行（デバッグ用）
python scripts/sqs_worker.py \
    --queue-url https://sqs.ap-northeast-1.amazonaws.com/123456789/latentsync-jobs.fifo \
    --work-dir /tmp/latentsync \
    --checkpoint checkpoints/latentsync_unet.pt \
    --region ap-northeast-1
```

```bash
# systemdサービスとして実行（本番推奨）
# 1. サービスファイルをコピー
sudo cp scripts/sqs_worker.service /etc/systemd/system/

# 2. 環境変数を設定（/etc/systemd/system/latentsync-worker.service を編集）
sudo vim /etc/systemd/system/latentsync-worker.service
# Environment="SQS_QUEUE_URL=..." を実際の値に変更

# 3. サービス起動
sudo systemctl daemon-reload
sudo systemctl enable latentsync-worker
sudo systemctl start latentsync-worker

# 4. ログ確認
sudo journalctl -u latentsync-worker -f
```

### Step 5: Auto Scaling設定（オプション）

```bash
# キューの長さに応じてEC2を自動スケール
aws autoscaling create-auto-scaling-group \
    --auto-scaling-group-name latentsync-workers \
    --launch-template LaunchTemplateId=lt-xxxxx \
    --min-size 0 \
    --max-size 10 \
    --desired-capacity 1

# CloudWatchアラームでスケーリング
aws cloudwatch put-metric-alarm \
    --alarm-name sqs-queue-depth \
    --metric-name ApproximateNumberOfMessagesVisible \
    --namespace AWS/SQS \
    --dimensions Name=QueueName,Value=latentsync-jobs.fifo \
    --comparison-operator GreaterThanThreshold \
    --threshold 5 \
    --evaluation-periods 2 \
    --alarm-actions arn:aws:autoscaling:...:policy/scale-up
```

### ジョブの流れ

```
1. Producer が S3 に動画/音声をアップロード
2. Producer が SQS にジョブメッセージを投入
3. Worker が SQS からメッセージを取得（Long Polling）
4. Worker が S3 から動画/音声をダウンロード
5. Worker がキャッシュの有無を確認、あればダウンロード
6. Worker が推論実行
7. Worker が結果を S3 にアップロード
8. Worker がキャッシュを S3 にアップロード（新規生成時）
9. Worker が SQS メッセージを削除（成功時）
```

### IAMポリシー例

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "sqs:ReceiveMessage",
                "sqs:DeleteMessage",
                "sqs:GetQueueAttributes"
            ],
            "Resource": "arn:aws:sqs:ap-northeast-1:123456789:latentsync-jobs.fifo"
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:HeadObject"
            ],
            "Resource": "arn:aws:s3:::my-latentsync-bucket/*"
        }
    ]
}
```

---

## 参考リンク

- [LatentSync GitHub](https://github.com/bytedance/LatentSync)
- [DeepCache](https://github.com/horseee/DeepCache)
- [InsightFace](https://github.com/deepinsight/insightface)
