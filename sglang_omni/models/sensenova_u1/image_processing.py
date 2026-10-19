# SPDX-License-Identifier: Apache-2.0
"""Decode, resize, and encode SenseNova image requests and results."""

from __future__ import annotations

import base64
import binascii
import io

import torch
from PIL import Image

from sglang_omni.models.sensenova_u1.neo_unify.utils import smart_resize
from sglang_omni.models.sensenova_u1.sampling import SenseNovaU1ImageEditSampling
from sglang_omni.proto.request import StagePayload

DEFAULT_INPUT_MAX_PIXELS = 2048 * 2048
MIN_INPUT_MAX_PIXELS = 512 * 512
IMAGE_RESIZE_FACTOR = 32


def decode_reference_images(value: str | list[str] | None) -> list[Image.Image]:
    values = value if isinstance(value, list) else [value]
    if not values or any(
        not isinstance(encoded_image, str) or not encoded_image
        for encoded_image in values
    ):
        raise ValueError("SenseNova-U1 requires at least one reference image")
    else:
        pass
    images: list[Image.Image] = []
    for encoded_image in values:
        try:
            image_bytes = base64.b64decode(encoded_image, validate=True)
            image = Image.open(io.BytesIO(image_bytes))
            image.load()
        except (binascii.Error, OSError, ValueError) as exc:
            raise ValueError(
                "SenseNova-U1 received an invalid reference image"
            ) from exc
        images.append(image)
    return images


def prepare_reference_images(
    value: str | list[str] | None, options: SenseNovaU1ImageEditSampling
) -> list[Image.Image]:
    images = decode_reference_images(value)
    if options.input_max_pixels:
        input_max_pixels = options.input_max_pixels
    elif len(images) <= 2:
        input_max_pixels = DEFAULT_INPUT_MAX_PIXELS
    else:
        input_max_pixels = max(
            MIN_INPUT_MAX_PIXELS, 2 * DEFAULT_INPUT_MAX_PIXELS // len(images)
        )
    prepared = []
    for image in images:
        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        else:
            image = image.convert("RGB")
        if options.do_resize:
            height, width = smart_resize(
                height=image.height,
                width=image.width,
                factor=IMAGE_RESIZE_FACTOR,
                min_pixels=input_max_pixels,
                max_pixels=input_max_pixels,
            )
            if image.size != (width, height):
                image = image.resize((width, height), Image.Resampling.LANCZOS)
            else:
                pass
        else:
            pass
        prepared.append(image)
    return prepared


def resolve_edit_output_size(
    references: list[Image.Image], options: SenseNovaU1ImageEditSampling
) -> tuple[int, int]:
    if options.size_explicit:
        return options.width, options.height
    else:
        pass
    target_pixels = options.width * options.height
    height, width = smart_resize(
        height=references[0].height,
        width=references[0].width,
        factor=IMAGE_RESIZE_FACTOR,
        min_pixels=target_pixels,
        max_pixels=target_pixels,
    )
    return width, height


def encode_image(
    payload: StagePayload, images: torch.Tensor, width: int, height: int
) -> StagePayload:
    if not isinstance(images, torch.Tensor) or images.shape != (1, 3, height, width):
        raise ValueError("SenseNova-U1 returned an invalid image tensor")
    else:
        pass
    image = ((images[0].float() + 1.0) * 127.5).clamp(0, 255)
    image = image.permute(1, 2, 0).byte().cpu().numpy()
    buffer = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="PNG")
    payload.data = {
        "image_b64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "modality": "image",
        "finish_reason": "stop",
    }
    return payload
