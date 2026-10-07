# videogen-server

Self-hosted video generation service. Chạy model **LTX-2.5** (`Lightricks/LTX-2.5-Diffusers`, qua `diffusers`) trên 1 GPU, nhận job qua HTTP API, trả video lên S3 và báo kết quả qua webhook.

## Kiến trúc

- **FastAPI**, 1 worker, 1 GPU, hàng đợi tuần tự (`asyncio.Queue`, tối đa `max_queue_depth` job).
- **SQLite** (`jobs.db`, nằm cạnh video trong `storage.output_dir`, tức trên volume) lưu trạng thái job. Khi server restart, job đang `running` bị đánh dấu `failed` (`server restarted`).
- Video dài hơn `max_native_duration_s` (10s) được **tách thành nhiều đoạn**. Khung cuối của đoạn trước làm khung đầu của đoạn sau, rồi ghép lại bằng `ffmpeg`.
- Kết quả lưu trên server tại `storage.output_dir/<job_id>.mp4` và tải qua API. Nếu không đặt `output_dir` thì upload **S3** (`boto3`) như cũ.
- Xong job thì POST webhook tới `callback_url`, ký HMAC-SHA256 trong header `X-Signature`.
- Config chung nằm ở `vendor/vidgen-core` (package `vidgen_core`, gồm `VideoRequest`, `VideoJobStatus`, `SelfHostConfig`).

```
src/videogen_server/
  service.py      FastAPI app, JobStore (SQLite), worker, webhook, lưu file/S3
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

### 2. Config

Mặc định server đọc `configs/ltx25.json` (đường dẫn tương đối theo thư mục chạy). Dữ liệu nằm trong `./data/`: `data/output` (video và `jobs.db`), `data/scratch`, `data/hf` (model). Nhớ chạy lệnh ở thư mục gốc của repo. Muốn dùng file khác thì đặt `VIDGEN_SELFHOST_CONFIG=<đường dẫn>`.

Chỉnh trong `configs/ltx25.json` nếu cần:

| Trường | Ý nghĩa |
|---|---|
| `storage.hf_home` | nơi chứa model (`HF_HOME` đặt sẵn thì ưu tiên biến đó) |
| `storage.output_dir` | thư mục lưu video và `jobs.db`. Bỏ trống thì upload S3 (`s3_bucket`, `s3_prefix`) |
| `service.base_url` | địa chỉ server này, dùng cho `result_url` |
| `service.callback_url` | URL nhận webhook khi job xong (`null` thì không gửi) |
| `runtime.min_free_vram_gb` | VRAM trống tối thiểu khi khởi động |

Đã có model ở chỗ khác (ví dụ `~/models/hf`)? Đặt `export HF_HOME=~/models/hf` hoặc `ln -s ~/models/hf data/hf`.

### 3. Tải model

Repo model là gated. Bấm **Agree/Request access** ở https://huggingface.co/Lightricks/LTX-2.5-Diffusers bằng đúng tài khoản tạo token, rồi:

```bash
export HF_TOKEN=hf_xxx
uv run videogen-download --config configs/ltx25.json            # tải
uv run videogen-download --config configs/ltx25.json --dry-run  # chỉ in dung lượng cần và đĩa trống
```

Lệnh bỏ qua `transformer_full/*` và file LoRA distilled (`ignore_patterns`), vì pipeline không dùng chúng.

### 4. Chạy server

```bash
uv run uvicorn --factory videogen_server.service:build_app --host 0.0.0.0 --port 8800 --workers 1
```

Mặc định server đặt `HF_HUB_OFFLINE=1` (dùng model đã tải ở bước 3) và `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Đặt biến môi trường trước để ghi đè.

Chạy nền: dùng `tmux`, hoặc `nohup ... > server.log 2>&1 &`. Giữ `--workers 1`: chỉ có 1 GPU và hàng đợi nằm trong process.

Kiểm tra: `curl localhost:8800/healthz`, hoặc mở `http://localhost:8800/docs`.

Biến môi trường (đều tùy chọn):

| Biến | Mô tả |
|---|---|
| `VIDGEN_SELFHOST_CONFIG` | đường dẫn file config, mặc định `configs/ltx25.json` |
| `HF_HOME` | thư mục cache model, mặc định `storage.hf_home` |
| `HF_HUB_OFFLINE` | mặc định `1`. Đặt `0` để server tự tải model khi khởi động |
| `HF_TOKEN` | token Hugging Face, khi tải model |
| `SELFHOST_WEBHOOK_SECRET` | khóa ký HMAC khi dùng webhook |
| `AWS_*` | chỉ khi dùng S3 (không đặt `output_dir`) |

## Chạy bằng Docker (cần GPU runtime)

Cần `nvidia-container-toolkit` trên host (cần sudo). Nếu `docker run --gpus all ...` báo `could not select device driver "" with capabilities: [[gpu]]`, host chưa có toolkit, và nếu không có sudo thì dùng cách chạy không Docker ở trên.

```bash
export HF_CACHE=$HOME/models/hf     # thư mục đã tải model (bước 3)
docker compose up -d --build
curl localhost:8800/healthz
```

- Image chạy cùng lệnh `uvicorn` và cùng `configs/ltx25.json`: model ở `/models/hf` (mount từ `HF_CACHE`, `HF_HOME` đặt sẵn trong image), dữ liệu ở volume `videogen-data` mount tại `/app/data`.
- `/app/data` chứa `jobs.db` (thông tin job), `output/` (video) và `scratch/`. Volume giữ nguyên khi `docker compose down` hoặc rebuild. Chỉ `docker compose down -v` mới xóa.
- Sửa `service.base_url` trong `configs/ltx25.json` thành `http://<IP-server>:8800` để `result_url` đúng host.
- Xem log: `docker compose logs -f`. Dừng: `docker compose down`.

## API

| Method | Path | Mô tả |
|---|---|---|
| `POST` | `/v1/jobs` | tạo job, trả `202 {"job_id": "..."}` |
| `GET` | `/v1/jobs/{job_id}` | trạng thái job |
| `GET` | `/v1/jobs/{job_id}/video` | tải video của job đã xong (`404` nếu chưa có) |
| `DELETE` | `/v1/jobs/{job_id}` | xóa video và thông tin job đã xong, trả `204`. Job đang chạy trả `409` |
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
