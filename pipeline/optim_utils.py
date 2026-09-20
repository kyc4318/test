"""Utilities derived from Tree-Ring (YuxinWenRick/tree-ring-watermark) and
METR (deepvk/metr), both MIT licensed.  Contains seed handling, image
transforms, distortions and CLIP similarity used by the first-phase
experiments.
"""

from __future__ import annotations

import io
import json
import random
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision import transforms


def read_json(filename: str) -> Mapping[str, Any]:
    with open(filename) as fp:
        return json.load(fp)


def set_random_seed(seed: int = 0) -> None:
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    random.seed(seed + 5)


def transform_img(image, target_size: int = 512):
    tform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ]
    )
    image = tform(image)
    return 2.0 * image - 1.0


def latents_to_imgs(pipe, latents):
    x = pipe.decode_image(latents)
    x = pipe.torch_to_numpy(x)
    x = pipe.numpy_to_pil(x)
    return x


def image_distortion(
    img1,
    img2,
    seed: int = 0,
    r_degree: float | None = None,
    jpeg_ratio: int | None = None,
    crop_scale: float | None = None,
    crop_ratio: float | None = None,
    gaussian_blur_r: int | None = None,
    gaussian_std: float | None = None,
    brightness_factor: float | None = None,
):
    """Apply the same distortion to img1/img2 (either may be None).
    Adapted from Tree-Ring / METR; the Gaussian-blur assignment bug of the
    upstream implementation is fixed here."""
    if r_degree is not None:
        rot = transforms.RandomRotation((r_degree, r_degree))
        img1 = rot(img1) if img1 is not None else None
        img2 = rot(img2)

    if jpeg_ratio is not None:
        if img1 is not None:
            buf = io.BytesIO()
            img1.save(buf, format="JPEG", quality=jpeg_ratio)
            img1 = Image.open(buf)
        buf2 = io.BytesIO()
        img2.save(buf2, format="JPEG", quality=jpeg_ratio)
        img2 = Image.open(buf2)

    if crop_scale is not None and crop_ratio is not None:
        if img1 is not None:
            set_random_seed(seed)
            img1 = transforms.RandomResizedCrop(
                img1.size, scale=(crop_scale, crop_scale), ratio=(crop_ratio, crop_ratio)
            )(img1)
        set_random_seed(seed)
        img2 = transforms.RandomResizedCrop(
            img2.size, scale=(crop_scale, crop_scale), ratio=(crop_ratio, crop_ratio)
        )(img2)

    if gaussian_blur_r is not None:
        if img1 is not None:
            img1 = img1.filter(ImageFilter.GaussianBlur(radius=gaussian_blur_r))
        img2 = img2.filter(ImageFilter.GaussianBlur(radius=gaussian_blur_r))

    if gaussian_std is not None:
        img_shape = np.array(img2).shape
        g_noise = np.random.normal(0, gaussian_std, img_shape) * 255
        g_noise = g_noise.astype(np.uint8)
        if img1 is not None:
            img1 = Image.fromarray(np.clip(np.array(img1) + g_noise, 0, 255))
        img2 = Image.fromarray(np.clip(np.array(img2) + g_noise, 0, 255))

    if brightness_factor is not None:
        jitter = transforms.ColorJitter(brightness=brightness_factor)
        if img1 is not None:
            img1 = jitter(img1)
        img2 = jitter(img2)

    return img1, img2


def apply_attack(img, seed: int, attack: str):
    """Map an attack string to image_distortion kwargs."""
    if attack == "clean":
        kwargs = {}
    elif attack.startswith("rot") and attack.endswith("_fix"):
        # P0 oracle-correction attack: rotate by angle, then rotate back with
        # the *known* angle using the same interpolation before inversion.
        angle = float(attack[3:-4])
        aug = image_distortion(None, img, seed=seed, r_degree=angle)[1]
        from torchvision.transforms import functional as F

        return F.rotate(aug, -angle, fill=0)
    elif attack.startswith("jpeg"):
        kwargs = {"jpeg_ratio": int(attack[4:])}
    elif attack.startswith("noise"):
        kwargs = {"gaussian_std": float(attack[5:])}
    elif attack.startswith("blur"):
        kwargs = {"gaussian_blur_r": int(attack[4:])}
    elif attack.startswith("bright"):
        kwargs = {"brightness_factor": float(attack[6:])}
    elif attack.startswith("rot"):
        kwargs = {"r_degree": float(attack[3:])}
    elif attack.startswith("crop"):
        v = float(attack[4:])
        kwargs = {"crop_scale": v, "crop_ratio": v}
    else:
        raise ValueError(f"unknown attack: {attack}")
    return image_distortion(None, img, seed=seed, **kwargs)[1]


def measure_similarity(images, prompt, model, clip_preprocess, tokenizer, device):
    with torch.no_grad():
        img_batch = torch.concatenate([clip_preprocess(i).unsqueeze(0) for i in images]).to(device)
        image_features = model.encode_image(img_batch)
        text = tokenizer([prompt]).to(device)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return (image_features @ text_features.T).mean(-1)


def get_dataset(dataset_id: str):
    """Support HF dataset ids and local COCO annotation jsons (coco:/path)."""
    from datasets import load_dataset

    if dataset_id.startswith("coco:"):
        with open(dataset_id.split(":", 1)[1]) as f:
            dataset = json.load(f)["annotations"]
        return dataset, "caption"
    if "laion" in dataset_id:
        dataset = load_dataset(dataset_id)["train"]
        return dataset, "TEXT"
    dataset = load_dataset(dataset_id)["test"]
    return dataset, "Prompt"
