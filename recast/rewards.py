"""Reward implementations used by the public 8xH200 recipe."""

from __future__ import annotations

import importlib
import io
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class Reward(Protocol):
    def __call__(
        self,
        images: Sequence[Image.Image],
        prompts: Sequence[str],
        metadata: Sequence[Mapping[str, object]],
    ) -> torch.Tensor:
        """Return one scalar reward per image/prompt pair."""


def _feature_tensor(output: object, attribute: str) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    value = getattr(output, attribute, None)
    if isinstance(value, torch.Tensor):
        return value
    pooler_output = getattr(output, "pooler_output", None)
    if isinstance(pooler_output, torch.Tensor):
        return pooler_output
    raise TypeError(f"model feature output has no tensor field {attribute!r}")


class CLIPScoreReward:
    """CLIP ViT-L/14 prompt-image alignment, scaled as cosine similarity."""

    def __init__(self, device: torch.device):
        from transformers import CLIPModel, CLIPProcessor

        model_name = "openai/clip-vit-large-patch14"
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).eval().to(device)

    @torch.inference_mode()
    def __call__(
        self,
        images: Sequence[Image.Image],
        prompts: Sequence[str],
        metadata: Sequence[Mapping[str, object]],
    ) -> torch.Tensor:
        del metadata
        inputs = self.processor(
            text=list(prompts),
            images=list(images),
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**inputs)
        return outputs.logits_per_image.diagonal() / 100.0


class PickScoreReward:
    """Prompt-image alignment from the public PickScore checkpoint."""

    def __init__(self, device: torch.device):
        from transformers import AutoModel, AutoProcessor

        self.device = device
        self.processor = AutoProcessor.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        )
        self.model = (
            AutoModel.from_pretrained("yuvalkirstain/PickScore_v1")
            .eval()
            .to(device=device, dtype=torch.float32)
        )

    @torch.inference_mode()
    def __call__(
        self,
        images: Sequence[Image.Image],
        prompts: Sequence[str],
        metadata: Sequence[Mapping[str, object]],
    ) -> torch.Tensor:
        del metadata
        image_inputs = self.processor(images=list(images), return_tensors="pt")
        text_inputs = self.processor(
            text=list(prompts),
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        image_inputs = {
            key: value.to(self.device) for key, value in image_inputs.items()
        }
        text_inputs = {key: value.to(self.device) for key, value in text_inputs.items()}
        image_features = _feature_tensor(
            self.model.get_image_features(**image_inputs), "image_embeds"
        )
        text_features = _feature_tensor(
            self.model.get_text_features(**text_inputs), "text_embeds"
        )
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        scores = self.model.logit_scale.exp() * (text_features * image_features).sum(
            dim=-1
        )
        return scores / 26.0


class HPSv21Reward:
    """Human Preference Score v2.1 with cached Hugging Face checkpoints."""

    _MEAN = (0.48145466, 0.4578275, 0.40821073)
    _STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, device: torch.device):
        try:
            from hpsv2.src.open_clip import create_model, get_tokenizer
        except ImportError as error:
            raise ImportError(
                "hpsv2 requires the H200 recipe dependencies; install with "
                "pip install -e '.[h200]'"
            ) from error
        from huggingface_hub import hf_hub_download

        base_checkpoint = hf_hub_download(
            repo_id="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            filename="open_clip_pytorch_model.bin",
        )
        hps_checkpoint = hf_hub_download(
            repo_id="xswu/HPSv2",
            filename="HPS_v2.1_compressed.pt",
        )
        model = create_model(
            "ViT-H-14",
            base_checkpoint,
            precision="fp32",
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            output_dict=True,
        )
        checkpoint = torch.load(hps_checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        self.device = device
        self.model = model.eval().to(device)
        self.tokenizer = get_tokenizer("ViT-H-14")
        image_size = model.visual.image_size
        self.image_size = int(
            image_size[0] if isinstance(image_size, tuple) else image_size
        )
        image_mean = getattr(model.visual, "image_mean", None)
        image_std = getattr(model.visual, "image_std", None)
        self.mean = torch.tensor(
            self._MEAN if image_mean is None else image_mean,
            device=device,
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            self._STD if image_std is None else image_std,
            device=device,
        ).view(1, 3, 1, 1)

    def _prepare_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        prepared: list[torch.Tensor] = []
        for image in images:
            array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
            height, width = tensor.shape[-2:]
            scale = self.image_size / max(height, width)
            new_height = round(height * scale)
            new_width = round(width * scale)
            tensor = F.interpolate(
                tensor,
                size=(new_height, new_width),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            pad_height = self.image_size - new_height
            pad_width = self.image_size - new_width
            tensor = F.pad(
                tensor,
                (
                    pad_width // 2,
                    pad_width - pad_width // 2,
                    pad_height // 2,
                    pad_height - pad_height // 2,
                ),
            )
            prepared.append(tensor)
        batch = torch.cat(prepared).to(self.device)
        return (batch - self.mean) / self.std

    @torch.inference_mode()
    def __call__(
        self,
        images: Sequence[Image.Image],
        prompts: Sequence[str],
        metadata: Sequence[Mapping[str, object]],
    ) -> torch.Tensor:
        del metadata
        image_tensor = self._prepare_images(images)
        tokens = self.tokenizer(list(prompts)).to(self.device)
        outputs = self.model(image_tensor, tokens)
        return (outputs["image_features"] @ outputs["text_features"].T).diagonal()


class JpegIncompressibilityReward:
    """JPEG byte size in kilobytes, a dependency-free complexity reward."""

    def __call__(
        self,
        images: Sequence[Image.Image],
        prompts: Sequence[str],
        metadata: Sequence[Mapping[str, object]],
    ) -> torch.Tensor:
        del prompts, metadata
        sizes: list[float] = []
        for image in images:
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=95)
            sizes.append(buffer.tell() / 1_000.0)
        return torch.tensor(sizes, dtype=torch.float32)


def load_reward(name: str, device: torch.device) -> Reward:
    """Build a built-in reward or a ``package.module:factory`` plugin."""

    if name == "geneval":
        from .geneval import GenEvalReward

        return GenEvalReward(device)
    builtins: dict[str, Callable[[torch.device], Reward]] = {
        "clipscore": CLIPScoreReward,
        "hpsv2": HPSv21Reward,
        "pickscore": PickScoreReward,
        "jpeg_incompressibility": lambda unused: JpegIncompressibilityReward(),
    }
    if name in builtins:
        return builtins[name](device)
    if ":" not in name:
        available = ", ".join((*builtins, "geneval"))
        raise ValueError(
            f"unknown reward {name!r}; use one of {available}, or "
            "package.module:factory"
        )
    module_name, factory_name = name.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    factory: Callable[[torch.device], Reward] = getattr(module, factory_name)
    reward = factory(device)
    if not callable(reward):
        raise TypeError(f"custom reward factory {name!r} did not return a callable")
    return reward


def score_reward(
    reward: Reward,
    images: Sequence[Image.Image],
    prompts: Sequence[str],
    metadata: Sequence[Mapping[str, object]],
) -> torch.Tensor:
    if len(images) != len(prompts) or len(images) != len(metadata):
        raise ValueError("reward inputs must contain one prompt and metadata per image")
    scores = torch.as_tensor(reward(images, prompts, metadata), dtype=torch.float32)
    if scores.shape != (len(images),):
        raise ValueError(
            f"reward returned shape {tuple(scores.shape)}; expected {(len(images),)}"
        )
    if not torch.isfinite(scores).all():
        raise ValueError("reward returned a non-finite score")
    return scores
