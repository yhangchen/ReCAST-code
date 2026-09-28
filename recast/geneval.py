"""Self-contained GenEval reward for structured Stage-2 prompts.

The detector is a public COCO Mask2Former checkpoint from Transformers. Color
attributes are classified with CLIP. The evaluator returns a continuous,
non-negative score over requested counts, colors, and spatial relations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass(frozen=True)
class DetectedObject:
    box: tuple[int, int, int, int]
    mask: np.ndarray
    score: float


def relative_position(
    first: DetectedObject,
    second: DetectedObject,
    threshold: float = 0.1,
) -> set[str]:
    """Return cardinal relations of the first object relative to the second."""

    boxes = np.asarray([first.box, second.box], dtype=np.float32).reshape(2, 2, 2)
    centers = boxes.mean(axis=1)
    dimensions = np.abs(np.diff(boxes, axis=1))[:, 0, :]
    offset = centers[0] - centers[1]
    revised = np.maximum(
        np.abs(offset) - threshold * dimensions.sum(axis=0), 0.0
    ) * np.sign(offset)
    norm = float(np.linalg.norm(offset))
    if norm == 0.0 or np.all(np.abs(revised) < 1e-3):
        return set()
    dx, dy = revised / norm
    relations: set[str] = set()
    if dx < -0.5:
        relations.add("left of")
    if dx > 0.5:
        relations.add("right of")
    if dy < -0.5:
        relations.add("above")
    if dy > 0.5:
        relations.add("below")
    return relations


def _feature_tensor(output: object, attribute: str) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    value = getattr(output, attribute, None)
    if isinstance(value, torch.Tensor):
        return value
    pooler_output = getattr(output, "pooler_output", None)
    if isinstance(pooler_output, torch.Tensor):
        return pooler_output
    raise TypeError(f"feature output has no tensor field {attribute!r}")


class GenEvalReward:
    """Continuous GenEval score for JSONL records with include/exclude clauses."""

    colors = (
        "red",
        "orange",
        "yellow",
        "green",
        "blue",
        "purple",
        "pink",
        "brown",
        "black",
        "white",
    )

    def __init__(self, device: torch.device, detector_batch_size: int = 8):
        from transformers import (
            AutoImageProcessor,
            CLIPModel,
            CLIPProcessor,
            Mask2FormerForUniversalSegmentation,
        )

        detector_name = "facebook/mask2former-swin-small-coco-instance"
        clip_name = "openai/clip-vit-large-patch14"
        self.device = device
        self.detector_batch_size = detector_batch_size
        self.detector_processor = AutoImageProcessor.from_pretrained(detector_name)
        self.detector = (
            Mask2FormerForUniversalSegmentation.from_pretrained(detector_name)
            .eval()
            .to(device)
        )
        self.clip_processor = CLIPProcessor.from_pretrained(clip_name)
        self.clip = CLIPModel.from_pretrained(clip_name).eval().to(device)
        self._color_features: dict[str, torch.Tensor] = {}

    @torch.inference_mode()
    def _detect(
        self, images: Sequence[Image.Image], metadata: Sequence[Mapping[str, object]]
    ) -> list[dict[str, list[DetectedObject]]]:
        all_objects: list[dict[str, list[DetectedObject]]] = []
        for start in range(0, len(images), self.detector_batch_size):
            image_batch = list(images[start : start + self.detector_batch_size])
            metadata_batch = metadata[start : start + self.detector_batch_size]
            inputs = self.detector_processor(
                images=image_batch, return_tensors="pt"
            ).to(self.device)
            outputs = self.detector(**inputs)
            target_sizes = [(image.height, image.width) for image in image_batch]
            results = self.detector_processor.post_process_instance_segmentation(
                outputs, threshold=0.0, target_sizes=target_sizes
            )
            for image, record, result in zip(
                image_batch, metadata_batch, results, strict=True
            ):
                tag = record.get("tag")
                threshold = 0.9 if tag == "counting" else 0.3
                detected: dict[str, list[DetectedObject]] = {}
                if result["segmentation"] is None:
                    all_objects.append(detected)
                    continue
                segmentation = result["segmentation"].detach().cpu().numpy()
                for segment in result["segments_info"]:
                    score = float(segment["score"])
                    if score <= threshold:
                        continue
                    label_id = int(segment["label_id"])
                    labels = self.detector.config.id2label
                    class_name = str(
                        labels.get(label_id, labels.get(str(label_id), label_id))
                    ).lower()
                    mask = segmentation == int(segment["id"])
                    rows, columns = np.nonzero(mask)
                    if not rows.size or not columns.size:
                        continue
                    box = (
                        int(columns.min()),
                        int(rows.min()),
                        int(columns.max()) + 1,
                        int(rows.max()) + 1,
                    )
                    detected.setdefault(class_name, []).append(
                        DetectedObject(box=box, mask=mask, score=score)
                    )
                for values in detected.values():
                    values.sort(key=lambda item: item.score, reverse=True)
                    del values[16:]
                all_objects.append(detected)
        return all_objects

    @torch.inference_mode()
    def _classify_colors(
        self,
        image: Image.Image,
        objects: Sequence[DetectedObject],
        class_name: str,
    ) -> list[str]:
        if not objects:
            return []
        image = image.convert("RGB")
        background = Image.new("RGB", image.size, color="#999")
        crops: list[Image.Image] = []
        for detected in objects:
            mask = Image.fromarray((detected.mask * 255).astype(np.uint8), mode="L")
            crops.append(Image.composite(image, background, mask).crop(detected.box))

        image_inputs = self.clip_processor(images=crops, return_tensors="pt").to(
            self.device
        )
        image_features = _feature_tensor(
            self.clip.get_image_features(**image_inputs), "image_embeds"
        )
        image_features = F.normalize(image_features, dim=-1)
        if class_name not in self._color_features:
            prompts = [
                template.format(color=color, class_name=class_name)
                for color in self.colors
                for template in (
                    "a photo of a {color} {class_name}",
                    "a photo of a {color}-colored {class_name}",
                    "a photo of a {color} object",
                )
            ]
            text_inputs = self.clip_processor(
                text=prompts,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(self.device)
            text_features = _feature_tensor(
                self.clip.get_text_features(**text_inputs), "text_embeds"
            )
            text_features = F.normalize(text_features, dim=-1)
            text_features = text_features.view(len(self.colors), 3, -1).mean(dim=1)
            self._color_features[class_name] = F.normalize(text_features, dim=-1)
        indices = (
            image_features @ self._color_features[class_name].transpose(0, 1)
        ).argmax(dim=1)
        return [self.colors[int(index)] for index in indices]

    def _score_one(
        self,
        image: Image.Image,
        objects: Mapping[str, list[DetectedObject]],
        metadata: Mapping[str, object],
    ) -> float:
        clauses = metadata.get("include")
        if not isinstance(clauses, list) or not clauses:
            raise ValueError("GenEval metadata requires a non-empty include list")
        components: list[float] = []
        matched_groups: list[list[DetectedObject] | None] = []
        for raw_clause in clauses:
            if not isinstance(raw_clause, dict):
                raise TypeError("GenEval include clauses must be objects")
            class_name = str(raw_clause.get("class", "")).lower()
            expected_count = int(raw_clause.get("count", 0))
            if not class_name or expected_count < 1:
                raise ValueError("GenEval clauses require class and positive count")
            found = objects.get(class_name, [])
            exact_count = len(found) == expected_count
            count_score = max(
                0.0, 1.0 - abs(expected_count - len(found)) / expected_count
            )
            components.append(count_score)
            matched = exact_count

            if "color" in raw_clause:
                if exact_count:
                    expected_color = str(raw_clause["color"]).lower()
                    colors = self._classify_colors(image, found, class_name)
                    color_score = colors.count(expected_color) / expected_count
                    components.append(color_score)
                    matched = matched and color_score == 1.0
                else:
                    components.append(0.0)
                    matched = False

            if "position" in raw_clause:
                position = raw_clause["position"]
                if (
                    not isinstance(position, (list, tuple))
                    or len(position) != 2
                    or not isinstance(position[1], int)
                ):
                    raise ValueError("GenEval position must be [relation, group_index]")
                relation = str(position[0])
                target_index = position[1]
                target = (
                    matched_groups[target_index]
                    if 0 <= target_index < len(matched_groups)
                    else None
                )
                relation_matches = bool(
                    exact_count
                    and target
                    and all(
                        relation in relative_position(item, target_item)
                        for item in found
                        for target_item in target
                    )
                )
                components.append(float(relation_matches))
                matched = matched and relation_matches

            matched_groups.append(found if matched else None)
        return float(sum(components) / len(components))

    @torch.inference_mode()
    def __call__(
        self,
        images: Sequence[Image.Image],
        prompts: Sequence[str],
        metadata: Sequence[Mapping[str, object]],
    ) -> torch.Tensor:
        del prompts
        detections = self._detect(images, metadata)
        scores = [
            self._score_one(image, objects, record)
            for image, objects, record in zip(images, detections, metadata, strict=True)
        ]
        return torch.tensor(scores, dtype=torch.float32)
