import argparse
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

NoiseMode = Literal['gaussian', 'salt_pepper', 'poisson']


def _rng(seed: Optional[int]) -> torch.Generator:
    gen = torch.Generator()
    gen.manual_seed(42 if seed is None else seed)
    return gen


def _prepare(image) -> tuple[torch.Tensor, bool, bool]:
    if isinstance(image, torch.Tensor):
        tensor = image.float()
        from_numpy = False
    else:
        tensor = torch.from_numpy(image).float()
        from_numpy = True
    was_uint8 = tensor.max().item() > 1.0
    if was_uint8:
        tensor = tensor / 255.0
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim != 4:
        raise ValueError('expected (H,W), (C,H,W) or (N,C,H,W)')
    return tensor, from_numpy, was_uint8


def lowpass(
    image,
    *,
    radius: int = 40,
    mode: NoiseMode = 'gaussian',
    sigma: float = 0.2,
    photon_lambda: float = 5.0,
    salt_amount: float = 0.02,
    salt_ratio: float = 0.5,
    phase_sigma: float = 0.05,
    clip: bool = True,
    generator: Optional[torch.Generator] = None,
):
    gen = generator or _rng(None)
    tensor, from_numpy, was_uint8 = _prepare(image)
    b, c, h, w = tensor.shape

    yy, xx = torch.meshgrid(
        torch.arange(h, device=tensor.device),
        torch.arange(w, device=tensor.device),
        indexing='ij',
    )
    mask = (((yy - h // 2) ** 2 + (xx - w // 2) ** 2) <= radius ** 2).float().view(1, 1, h, w)

    R = tensor[:, 0:1]
    G = tensor[:, 1:2]
    B = tensor[:, 2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 0.5
    Cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 0.5

    tensor = Y
    fft_img = torch.fft.fftshift(torch.fft.fft2(tensor, dim=(-2, -1)))
    amplitude = fft_img.abs() * mask
    phase = torch.angle(fft_img)
    if phase_sigma > 0:
        # noise_phase = torch.randn(phase.shape, dtype=phase.dtype, device=phase.device, generator=gen) # RGB三通道不同噪声
        noise_phase = torch.randn((1, 1, h, w), dtype=phase.dtype, device=phase.device, generator=gen)# 三通道相同噪声
        phase = phase + phase_sigma * noise_phase

    if mode == 'gaussian':
        noise = torch.randn((1, 1, h, w), dtype=amplitude.dtype, device=amplitude.device, generator=gen)
        # noise = torch.randn(amplitude.shape, dtype=amplitude.dtype, device=amplitude.device, generator=gen)
        amplitude = (amplitude * (1.0 + sigma * noise)).clamp_min(0.0)
    elif mode == 'poisson':
        scale = amplitude.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        normalized = amplitude / scale
        photons = torch.poisson((normalized * photon_lambda).clamp_min(0.0))
        amplitude = photons / photon_lambda * scale
    amplitude = amplitude * mask

    restored = torch.fft.ifft2(
        torch.fft.ifftshift(torch.polar(amplitude, phase), dim=(-2, -1)),
        dim=(-2, -1),
    ).real

    Y_restored = restored[:, 0:1]
    if mode == 'salt_pepper':
        if not 0 <= salt_amount <= 1:
            raise ValueError('salt_amount must be in [0,1]')
        if not 0 <= salt_ratio <= 1:
            raise ValueError('salt_ratio must be in [0,1]')
        mask_sp = torch.rand(1, 1, h, w, generator=gen, device=restored.device)
        salt = mask_sp < salt_amount * salt_ratio
        pepper = (mask_sp >= salt_amount * salt_ratio) & (mask_sp < salt_amount)
        Y_restored = Y_restored.clone()
        salt = salt.expand_as(Y_restored)
        pepper = pepper.expand_as(Y_restored)
        Y_restored[salt] = torch.clamp(Y_restored[salt] * 1.5, 0.0, 1.0)
        Y_restored[pepper] = torch.clamp(Y_restored[pepper] * 0.2, 0.0, 1.0)

    R_rec = Y_restored + 1.402 * (Cr - 0.5)
    G_rec = Y_restored - 0.344136 * (Cb - 0.5) - 0.714136 * (Cr - 0.5)
    B_rec = Y_restored + 1.772 * (Cb - 0.5)
    restored = torch.cat([R_rec, G_rec, B_rec], dim=1)

    if clip:
        restored = restored.clamp(0.0, 1.0)
    if was_uint8:
        restored = (restored * 255.0).round().clamp(0, 255)

    out = restored.squeeze(0)
    if from_numpy:
        return out.cpu().numpy()
    return out


def _process_single(path: Path, dst_path: Path, **kwargs):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f'failed to read {path}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    protected = lowpass(img.transpose(2, 0, 1), **kwargs)
    if isinstance(protected, torch.Tensor):
        protected = protected.cpu().numpy()
    protected = np.clip(protected.transpose(1, 2, 0), 0, 1)
    protected = (protected * 255).astype(np.uint8)
    protected = cv2.cvtColor(protected, cv2.COLOR_RGB2BGR)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_path), protected)


def protect_dataset(args):
    gen = _rng(args.seed)
    image_paths = sorted(args.src_root.rglob(args.ext))
    if not image_paths:
        print('no images found')
        return
    for mode in args.modes:
        dst_root = (args.data_root / mode).resolve()
        for src_path in tqdm(image_paths, desc=f'protecting {mode}'):
            rel = src_path.relative_to(args.src_root)
            dst_path = dst_root / rel
            _process_single(
                src_path,
                dst_path,
                mode=mode,
                radius=args.radius,
                sigma=args.sigma,
                photon_lambda=args.photon_lambda,
                salt_amount=args.salt_amount,
                salt_ratio=args.salt_ratio,
                phase_sigma=args.phase_sigma,
                generator=gen,
            )


def build_parser():
    p = argparse.ArgumentParser(description='Frequency-domain noise injector')
    p.add_argument('--src-root', type=Path, required=True)
    p.add_argument('--data-root', type=Path, default=Path('D:/Tencent_facial/data'))
    p.add_argument('--modes', nargs='+', default=['gaussian', 'salt_pepper', 'poisson'])
    p.add_argument('--radius', type=int, default=20)
    p.add_argument('--sigma', type=float, default=0.45)
    p.add_argument('--photon-lambda', type=float, default=55.0)
    p.add_argument('--salt-amount', type=float, default=0.1)
    p.add_argument('--salt-ratio', type=float, default=0.5)
    p.add_argument('--phase-sigma', type=float, default=0.02)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--ext', type=str, default='*.jpg')
    p.add_argument('--test', action='store_true', help='preview a random sample instead of batch processing')
    return p

def test_preview(args):
    import random
    import matplotlib.pyplot as plt

    paths = sorted(args.src_root.rglob(args.ext))
    if not paths:
        print('No images found for preview.')
        return
    sample = random.choice(paths)
    img = cv2.imread(str(sample), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    cols = len(args.modes) + 2
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    axes[0].imshow(img)
    axes[0].set_title('original')
    axes[0].axis('off')
    lpf = lowpass(
        img.transpose(2, 0, 1),
        mode='gaussian',
        radius=args.radius,
        sigma=0.0,
        photon_lambda=args.photon_lambda,
        salt_amount=0.0,
        salt_ratio=args.salt_ratio,
        phase_sigma=0.0,
    ).transpose(1, 2, 0)
    axes[1].imshow(np.clip(lpf, 0, 1))
    axes[1].set_title('low-pass only')
    axes[1].axis('off')
    for ax, mode in zip(axes[2:], args.modes):
        noisy = lowpass(
            img.transpose(2, 0, 1),
            mode=mode,
            radius=args.radius,
            sigma=args.sigma,
            photon_lambda=args.photon_lambda,
            salt_amount=args.salt_amount,
            salt_ratio=args.salt_ratio,
            phase_sigma=args.phase_sigma,
        ).transpose(1, 2, 0)
        ax.imshow(np.clip(noisy, 0, 1))
        ax.set_title(mode)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    if args.test:
        test_preview(args)
    else:
        protect_dataset(args)
