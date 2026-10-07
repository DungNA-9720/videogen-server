"""LTX-2.5 engine (diffusers). Needs the `gpu` extra; imported lazily by the service."""

import gc

import torch
from diffusers import LTX2ConditionPipeline
from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT, DISTILLED_SIGMA_VALUES
from diffusers.utils import encode_video
from huggingface_hub import snapshot_download
from vidgen_core.selfhost import SelfHostConfig

from videogen_server.segments import GenerationOOM, SegmentJob


class LTX2Engine:
    def __init__(self, cfg: SelfHostConfig):
        self.cfg = cfg
        self.pipe: LTX2ConditionPipeline | None = None

    def load(self) -> None:
        c = self.cfg
        dtype = getattr(torch, c.model.torch_dtype)
        # Local snapshot path skips diffusers' "cache complete?" check, which fails offline
        # because ignore_patterns (transformer_full/*) were never downloaded.
        path = snapshot_download(
            c.model.model_id,
            revision=c.model.revision,
            ignore_patterns=c.model.ignore_patterns,
            local_files_only=True,
        )
        pipe = LTX2ConditionPipeline.from_pretrained(
            path,
            torch_dtype=dtype,
            variant=c.model.variant,
        )  # model_id LẤY TỪ JSON
        if c.runtime.layerwise_fp8_storage:  # lưu FP8, tính BF16 → hợp lệ trên sm_86
            pipe.transformer.enable_layerwise_casting(
                storage_dtype=torch.float8_e4m3fn, compute_dtype=dtype
            )
        if c.runtime.attention_backend != "native":
            pipe.transformer.set_attention_backend(c.runtime.attention_backend)
        if c.runtime.vae_tiling:
            pipe.vae.enable_tiling()
        match c.runtime.offload_mode:
            case "model":
                pipe.enable_model_cpu_offload()
            case "sequential":
                pipe.enable_sequential_cpu_offload()
            case "none":
                pipe.to(c.runtime.device)
            case "group":
                pipe.enable_model_cpu_offload()
        self.pipe = pipe

    @torch.inference_mode()
    def generate(self, j: SegmentJob) -> str:
        assert self.pipe is not None
        g = self.cfg.generation
        conds = []
        if j.first is not None:
            conds.append(LTX2VideoCondition(frames=j.first, index=0, strength=g.condition_strength))
        if j.last is not None:
            conds.append(LTX2VideoCondition(frames=j.last, index=-1, strength=g.condition_strength))
        kwargs = dict(g.extra_call_kwargs)
        if g.sigmas_preset == "distilled":
            kwargs["sigmas"] = DISTILLED_SIGMA_VALUES
        else:
            kwargs["num_inference_steps"] = g.num_inference_steps
        try:
            video, audio = self.pipe(
                conditions=conds or None,
                prompt=j.prompt,
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                width=j.width,
                height=j.height,
                num_frames=j.num_frames,
                frame_rate=float(g.fps),
                guidance_scale=g.guidance_scale,
                generator=torch.Generator("cuda").manual_seed(j.seed),
                output_type="np",
                return_dict=False,
                **kwargs,
            )
            encode_video(
                video[0],
                fps=g.fps,
                output_path=j.out_path,
                audio=audio[0].float().cpu() if g.keep_audio else None,
                audio_sample_rate=self.pipe.vocoder.config.output_sampling_rate,
            )
            return j.out_path
        except torch.cuda.OutOfMemoryError as e:
            raise GenerationOOM(f"CUDA OOM @ {j.width}x{j.height}x{j.num_frames}: {e}") from e
        finally:
            gc.collect()
            torch.cuda.empty_cache()
