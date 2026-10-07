# videogen-server

Self-hosted video generation service. Chạy model **LTX-2.5** (`Lightricks/LTX-2.5-Diffusers`, qua `diffusers`) trên 1 GPU, nhận job qua HTTP API, trả video lên S3 và báo kết quả qua webhook.

## Kiến trúc

- **FastAPI**, 1 worker, 1 GPU, hàng đợi tuần tự (`asyncio.Queue`, tối đa `max_queue_depth` job).
- **Redis** lưu trạng thái job (TTL 7 ngày). Khi server restart, job đang `running` bị đánh dấu `failed` (`server restarted`).
- Video dài hơn `max_native_duration_s` (10s) được **tách thành nhiều đoạn**. Khung cuối của đoạn trước làm khung đầu của đoạn sau, rồi ghép lại bằng `ffmpeg`.
- Kết quả upload lên **S3** (`boto3`), job trả về `s3://bucket/key`.
- Xong job thì POST webhook tới `callback_url`, ký HMAC-SHA256 trong header `X-Signature`.
- Config chung nằm ở `vendor/vidgen-core` (package `vidgen_core`, gồm `VideoRequest`, `VideoJobStatus`, `SelfHostConfig`).

```
src/videogen_server/
  service.py      FastAPI app, JobStore (Redis), worker, webhook, upload S3
  engine_ltx2.py  load pipeline + generate 1 đoạn
  segments.py     chia đoạn, tính số frame, lấy khung cuối, ghép bằng ffmpeg
  download.py     CLI `videogen-download`: tải model về HF cache
configs/ltx25.json  config mẫu
vendor/vidgen-core  models + config schema dùng chung
tests/              test không cần GPU
```

## Yêu cầu

- GPU NVIDIA, driver hỗ trợ CUDA 13 (đã test: RTX A6000 48GB, driver 595, CUDA 13.2). Cần trống ít nhất `min_free_vram_gb` = 40GB.
- Python 3.12, [`uv`](https://docs.astral.sh/uv/), `ffmpeg`.
- Redis.
- Tài khoản Hugging Face đã được cấp quyền vào repo gated `Lightricks/LTX-2.5-Diffusers`, cùng token đọc (`HF_TOKEN`).
- Ít nhất 40GB ổ đĩa trống (`storage.min_free_gb`) cho model.
- AWS credentials ghi được vào bucket cấu hình trong `storage.s3_bucket`.

## Cài đặt và chạy (không Docker, không sudo)

### 1. Lấy code và cài dependency

```bash
git clone https://github.com/DungNA-9720/videogen-server.git
cd videogen-server
curl -LsSf https://astral.sh/uv/install.sh | sh      # nếu chưa có uv
uv sync --frozen --no-dev --extra gpu
```

`ffmpeg`: nếu máy chưa có và không có sudo, tải bản static vào `~/bin`:

```bash
mkdir -p ~/bin && curl -L https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz \
  | tar -xJ --strip-components=1 -C ~/bin --wildcards '*/ffmpeg' '*/ffprobe'
export PATH=~/bin:$PATH
```

### 2. Chạy Redis

Port 6379 thường đã bị chiếm bởi service khác, nên chạy Redis riêng ở port 6380:

```bash
docker run -d --name redis-videogen -p 6380:6379 --restart unless-stopped redis:7
```

### 3. Tạo config local

`configs/ltx25.json` dùng đường dẫn `/srv/...` cần quyền root. Tạo bản sao trỏ vào thư mục của bạn:

```bash
mkdir -p ~/models/hf ~/videogen/scratch
sed -e "s#/srv/models/hf#$HOME/models/hf#" \
    -e "s#/srv/videogen/scratch#$HOME/videogen/scratch#" \
    configs/ltx25.json > configs/local.json
```

Cần chỉnh thêm trong `configs/local.json` cho đúng môi trường:

| Trường | Ý nghĩa |
|---|---|
| `storage.s3_bucket`, `s3_prefix` | nơi upload video kết quả |
| `service.callback_url` | URL nhận webhook khi job xong (bỏ trống thì không gửi) |
| `service.base_url` | địa chỉ server này |
| `runtime.min_free_vram_gb` | VRAM trống tối thiểu khi khởi động |

`configs/local.json` chứa đường dẫn riêng của máy, nên nằm ngoài những gì cần commit.

### 4. Tải model

Repo model là gated. Bấm **Agree/Request access** ở https://huggingface.co/Lightricks/LTX-2.5-Diffusers bằng đúng tài khoản tạo token, rồi:

```bash
export HF_TOKEN=hf_xxx
uv run videogen-download --config configs/local.json            # tải
uv run videogen-download --config configs/local.json --dry-run  # chỉ in dung lượng cần và đĩa trống
```

Lệnh bỏ qua `transformer_full/*` và file LoRA distilled (`ignore_patterns`), vì pipeline không dùng chúng.

### 5. Chạy server

```bash
export VIDGEN_SELFHOST_CONFIG=$PWD/configs/local.json
export HF_HOME=~/models/hf HF_HUB_OFFLINE=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       REDIS_URL=redis://localhost:6380/0 \
       SELFHOST_WEBHOOK_SECRET=<chuỗi-bí-mật> \
       AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_DEFAULT_REGION=...
uv run uvicorn --factory videogen_server.service:build_app --host 0.0.0.0 --port 8800 --workers 1
```

Chạy nền: dùng `tmux`, hoặc `nohup ... > server.log 2>&1 &`. Giữ `--workers 1`: chỉ có 1 GPU và hàng đợi nằm trong process.

Kiểm tra: `curl localhost:8800/healthz`, hoặc mở `http://localhost:8800/docs`.

Biến môi trường:

| Biến | Bắt buộc | Mô tả |
|---|---|---|
| `VIDGEN_SELFHOST_CONFIG` | có | đường dẫn file config JSON |
| `HF_HOME` | có | thư mục cache model, trùng `storage.hf_home` |
| `HF_HUB_OFFLINE` | khuyên đặt `1` | bắt buộc dùng model đã tải sẵn. Nếu không đặt, server tự tải model khi khởi động |
| `HF_TOKEN` | khi tải model | token Hugging Face |
| `REDIS_URL` | có | mặc định `redis://localhost:6379/0` |
| `SELFHOST_WEBHOOK_SECRET` | khi dùng webhook | khóa ký HMAC (tên biến lấy từ `callback_hmac_secret_env`) |
| `AWS_*` | có | credentials cho `boto3` |

## Chạy bằng Docker (cần GPU runtime)

`Dockerfile` dựa trên `nvidia/cuda:13.0.2-runtime-ubuntu24.04`. Cần `nvidia-container-toolkit` trên host (cần sudo). Nếu `docker run --gpus all ...` báo `could not select device driver "" with capabilities: [[gpu]]`, host chưa có toolkit, và nếu không có sudo thì dùng cách chạy không Docker ở trên.

```bash
docker build -t videogen-server .
docker run -d --name videogen --gpus all --network host \
  -e VIDGEN_SELFHOST_CONFIG=/app/configs/local.json \
  -e REDIS_URL=redis://localhost:6380/0 \
  -e SELFHOST_WEBHOOK_SECRET=... \
  -v ~/models/hf:/models/hf videogen-server
```

Trong image `HF_HOME=/models/hf` và `HF_HUB_OFFLINE=1`, nên `storage.hf_home` trong config phải là `/models/hf`.

## API

| Method | Path | Mô tả |
|---|---|---|
| `POST` | `/v1/jobs` | tạo job, trả `202 {"job_id": "..."}` |
| `GET` | `/v1/jobs/{job_id}` | trạng thái job |
| `GET` | `/v1/capabilities` | khả năng của provider |
| `GET` | `/healthz` | độ sâu hàng đợi, VRAM trống, model |

Tạo job:

```bash
curl -X POST localhost:8800/v1/jobs -H 'Content-Type: application/json' -d '{
  "prompt": "a cat walking on a beach at sunset",
  "duration_s": 8,
  "spec": {"width": 1280, "height": 704, "fps": 24},
  "first_frame_uri": null,
  "last_frame_uri": null,
  "seed": 42
}'
```

- `first_frame_uri` và `last_frame_uri` phải là URL `http(s)` (presigned). `last_frame_uri` cần có `first_frame_uri`.
- `duration_s` tối đa `constraints.max_duration_s` (30s). Vượt thì `422`.
- `reference_uris` hiện không hỗ trợ (`max_reference_images = 0`).
- Hàng đợi đầy thì `503` kèm `Retry-After: 60`.

Trạng thái job (`VideoJobStatus`): `state` là `running`, `succeeded` hoặc `failed`. Khi thành công có `result_url` (`s3://...`). Khi lỗi có `error` với tiền tố `OOM:`, `invalid_input:` hoặc `internal:`.

Webhook: POST JSON `VideoJobStatus` tới `service.callback_url`, header `X-Signature` = HMAC-SHA256 hex của body bằng `SELFHOST_WEBHOOK_SECRET`. Lỗi gửi webhook bị bỏ qua, nên client vẫn cần poll `GET /v1/jobs/{id}`.

## Test

Không cần GPU:

```bash
uv sync --group dev
uv run pytest
```

Test đánh dấu `gpu` bị bỏ qua mặc định (`-m 'not gpu'`).

## Xử lý sự cố

| Lỗi | Nguyên nhân và cách xử lý |
|---|---|
| `could not select device driver "" with capabilities: [[gpu]]` | Docker thiếu `nvidia-container-toolkit`. Nhờ admin cài, hoặc chạy không Docker |
| `failed to bind port 0.0.0.0:6379 ... address already in use` | Port Redis đã bị chiếm. Dùng port khác (6380), xem ai giữ port bằng `ss -ltnp \| grep 6379` |
| `GatedRepoError: 403` khi tải model | Tài khoản chưa được cấp quyền repo, hoặc `HF_TOKEN` chưa export / sai. Kiểm tra `curl -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/api/whoami-v2` |
| `KeyError: 'VIDGEN_SELFHOST_CONFIG'` | Chưa đặt biến môi trường trỏ tới file config |
| `RuntimeError: free VRAM ... < min_free_vram_gb` | GPU đang bị process khác chiếm. Giải phóng VRAM hoặc hạ `runtime.min_free_vram_gb` |
| `Thiếu đĩa: cần ...GB` | Ổ đĩa không đủ chỗ cho model |
| `OSError: ... is incomplete: 5 file(s) are missing (transformer_full/...)` | `HF_HUB_OFFLINE=1` khiến diffusers đòi cache đầy đủ. `engine_ltx2.py` đã xử lý bằng cách load từ đường dẫn snapshot local. Nếu gặp lại, kiểm tra code đã cập nhật chưa |
| Job `failed` với `OOM:` | Hết VRAM khi sinh. Giảm `render_width/height` hoặc đặt `runtime.offload_mode` thành `sequential` |
| Upload S3 lỗi (`internal: ...NoCredentialsError`) | Thiếu `AWS_*` hoặc không có quyền ghi bucket |

## Cập nhật

```bash
git pull
uv sync --frozen --no-dev --extra gpu
# restart uvicorn
```
