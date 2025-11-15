import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

NOISE_CHOICES = ['gaussian', 'salt_pepper', 'poisson']
DEFAULT_NOISE_SEED = 42


def get_noise_generator(seed: Optional[int] = None) -> torch.Generator:
    gen = torch.Generator()
    gen.manual_seed(DEFAULT_NOISE_SEED if seed is None else seed)
    return gen


def _circular_mask(shape, radius, device):
    h, w = shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing='ij',
    )
    cy, cx = h // 2, w // 2
    mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius ** 2
    return mask.float()


def _rand_like(tensor, generator):
    return torch.rand(
        tensor.shape,
        dtype=tensor.dtype,
        device=tensor.device,
        layout=tensor.layout,
        generator=generator,
    )


def _randn_like(tensor, generator):
    return torch.randn(
        tensor.shape,
        dtype=tensor.dtype,
        device=tensor.device,
        layout=tensor.layout,
        generator=generator,
    )


def _apply_freq_noise(magnitude, mode, sigma, amount, ratio, generator):
    if mode == 'gaussian':
        noise = _randn_like(magnitude, generator) * sigma
        return (magnitude + noise).clamp_min(0.0)

    if mode == 'salt_pepper':
        if not 0 <= amount <= 1:
            raise ValueError('amount must be within [0, 1]')
        if not 0 <= ratio <= 1:
            raise ValueError('ratio must be within [0, 1]')
        rand = _rand_like(magnitude, generator)
        salt = rand < amount * ratio
        pepper = (rand >= amount * ratio) & (rand < amount)
        out = magnitude.clone()
        max_val = magnitude.max()
        if max_val <= 0:
            max_val = torch.tensor(1.0, device=magnitude.device)
        out[salt] = max_val
        out[pepper] = 0.0
        return out

    if mode == 'poisson':
        scaled = (magnitude * 255.0).clamp_min(0.0)
        noisy = torch.poisson(scaled)
        return noisy / 255.0

    raise ValueError(f'unsupported noise mode: {mode}')


def lowpass(image: np.ndarray, *, radius, mode, sigma, amount, ratio, generator):
    tensor = torch.from_numpy(image).float()
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(f'expected image shape (C,H,W); got {image.shape}')

    if tensor.ndim == 4:
        _, _, h, w = tensor.shape
    else:
        _, h, w = tensor.shape
    mask = _circular_mask((h, w), radius, tensor.device).view(1, 1, h, w)

    fft_img = torch.fft.fftshift(torch.fft.fft2(tensor, dim=(-2, -1)))
    magnitude = fft_img.abs()
    phase = torch.angle(fft_img)
    filtered_mag = magnitude * mask
    noisy_mag = _apply_freq_noise(filtered_mag, mode, sigma, amount, ratio, generator) * mask

    complex_spec = torch.polar(noisy_mag, phase)
    protected = torch.fft.ifft2(
        torch.fft.ifftshift(complex_spec, dim=(-2, -1)),
        dim=(-2, -1),
    ).real

    protected = protected.clamp(0.0, 1.0)
    return protected.squeeze(0).numpy()


def process_one(
    src_path: Path,
    dst_path: Path,
    *,
    mode: str,
    radius: float,
    sigma: float,
    amount: float,
    ratio: float,
    generator,
):
    img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"failed to read {src_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    protected = lowpass(
        img.transpose(2, 0, 1),
        radius=radius,
        mode=mode,
        sigma=sigma,
        amount=amount,
        ratio=ratio,
        generator=generator,
    ).transpose(1, 2, 0)
    protected = (protected * 255).clip(0, 255).astype('uint8')
    protected = cv2.cvtColor(protected, cv2.COLOR_RGB2BGR)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_path), protected)


def parse_args():
    parser = argparse.ArgumentParser(description='Apply frequency-domain protection to dataset.')
    parser.add_argument(
        '--src-root',
        type=Path,
        default=Path('D:/Tencent_facial/data/utkface_aligned_cropped/UTKFace'),
        help='source dataset root',
    )
    parser.add_argument(
        '--data-root',
        type=Path,
        default=Path('D:/Tencent_facial/data'),
        help='directory under which noise-specific folders are created',
    )
    parser.add_argument(
        '--modes',
        nargs='+',
        choices=NOISE_CHOICES,
        default=NOISE_CHOICES,
        help='noise types to generate',
    )
    parser.add_argument('--radius', type=float, default=45.0, help='low-pass radius (larger keeps more structure)')
    parser.add_argument('--sigma', type=float, default=0.005, help='gaussian noise std in freq domain')
    parser.add_argument('--amount', type=float, default=0.005, help='salt/pepper amount in freq domain')
    parser.add_argument('--ratio', type=float, default=0.5, help='salt ratio within salt-pepper noise')
    parser.add_argument('--seed', type=int, default=None, help='seed for shared noise generator')
    parser.add_argument('--ext', type=str, default='*.jpg', help='glob pattern for image files')
    return parser.parse_args()


def main():
    args = parse_args()
    src_root = args.src_root
    if not src_root.exists():
        raise FileNotFoundError(f"source root not found: {src_root}")

    generator = get_noise_generator(args.seed)
    image_paths = sorted(src_root.rglob(args.ext))
    if not image_paths:
        print(f'no files matching {args.ext} found under {src_root}')
        return

    for mode in args.modes:
        dst_root = (args.data_root / mode).resolve()
        dst_root.mkdir(parents=True, exist_ok=True)
        for src_path in tqdm(image_paths, desc=f"protecting faces ({mode})"):
            rel = src_path.relative_to(src_root)
            dst_path = dst_root / rel
            process_one(
                src_path,
                dst_path,
                mode=mode,
                radius=args.radius,
                sigma=args.sigma,
                amount=args.amount,
                ratio=args.ratio,
                generator=generator,
            )


if __name__ == '__main__':
    main()
