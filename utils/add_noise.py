import numpy as np
import numpy as np
import torch
from typing import Literal, Optional

NoiseMode = Literal['gaussian', 'salt_pepper', 'poisson']

DEFAULT_NOISE_SEED = 42
_GLOBAL_NOISE_GENERATOR = torch.Generator()
_GLOBAL_NOISE_GENERATOR.manual_seed(DEFAULT_NOISE_SEED)


def get_noise_generator(seed: Optional[int] = None) -> torch.Generator:
    """
    Return the shared noise generator or create a seeded copy.
    """
    if seed is None:
        return _GLOBAL_NOISE_GENERATOR
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def _to_tensor(image):
    if isinstance(image, torch.Tensor):
        return image.float(), False
    if isinstance(image, np.ndarray):
        return torch.from_numpy(image).float(), True
    raise TypeError(f'Unsupported image type: {type(image)}')


def _circular_mask(hw, radius, device):
    h, w = hw
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing='ij',
    )
    cy, cx = h // 2, w // 2
    mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius ** 2
    return mask.float()


def _apply_freq_noise(
    magnitude,
    *,
    mode: NoiseMode,
    sigma: float,
    amount: float,
    ratio: float,
    generator: Optional[torch.Generator],
):
    if mode == 'gaussian':
        noise = _randn_like(magnitude, generator=generator) * sigma
        return (magnitude + noise).clamp_min(0.0)

    if mode == 'salt_pepper':
        if not 0 <= amount <= 1:
            raise ValueError('amount must be within [0, 1].')
        if not 0 <= ratio <= 1:
            raise ValueError('ratio must be within [0, 1].')
        rand = _rand_like(magnitude, generator=generator)
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


def lowpass(
    image,
    *,
    radius: int = 40,
    mode: NoiseMode = 'gaussian',
    sigma: float = 0.002,
    amount: float = 0.002,
    ratio: float = 0.5,
    clip: bool = True,
    generator: Optional[torch.Generator] = None,
):
    """
    FFT -> shift -> circular low-pass -> selectable frequency noise -> IFFT.
    """
    generator = generator or get_noise_generator()
    tensor, from_numpy = _to_tensor(image)
    was_uint8 = tensor.max().item() > 1.0
    if was_uint8:
        tensor = tensor / 255.0

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError('expected (N,C,H,W), (C,H,W), or (H,W)')

    _, _, h, w = tensor.shape
    mask = _circular_mask((h, w), radius, tensor.device).view(1, 1, h, w)
    fft_img = torch.fft.fftshift(torch.fft.fft2(tensor, dim=(-2, -1)))
    magnitude = fft_img.abs()
    phase = torch.angle(fft_img)
    filtered_mag = magnitude * mask

    noisy_mag = _apply_freq_noise(
        filtered_mag,
        mode=mode,
        sigma=sigma,
        amount=amount,
        ratio=ratio,
        generator=generator,
    )
    noisy_mag = noisy_mag * mask

    complex_spec = torch.polar(noisy_mag, phase)
    restored = torch.fft.ifft2(
        torch.fft.ifftshift(complex_spec, dim=(-2, -1)),
        dim=(-2, -1),
    ).real

    if clip:
        restored = restored.clamp(0.0, 1.0)
    if was_uint8:
        restored = (restored * 255.0).round().clamp(0, 255)

    out = restored.squeeze(0)
    if from_numpy:
        return out.numpy()
    return out


def add_noise(
    image,
    mode: NoiseMode = 'gaussian',
    r_ratio: float = 0.5,
    sigma: float = 0.05,
    *,
    clip: bool = True,
    amount: float = 0.02,
    generator: Optional[torch.Generator] = None,
):
    """
    Add spatial noise for quick experiments (gaussian / salt_pepper / poisson).
    """
    generator = generator or get_noise_generator()
    tensor, from_numpy = _to_tensor(image)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError('expected tensor shape: (N,C,H,W) or (C,H,W)')

    if mode == 'gaussian':
        noisy = _gaussian_noise(tensor, sigma, clip, generator)
    elif mode == 'salt_pepper':
        noisy = _sp_noise(tensor, amount, r_ratio, clip, generator)
    elif mode == 'poisson':
        noisy = _poisson_noise(tensor, clip, generator)
    else:
        raise ValueError(f'unsupported noise mode: {mode}')

    if from_numpy:
        return noisy.squeeze(0).numpy()
    return noisy.squeeze(0)


def _gaussian_noise(tensor, sigma, clip, generator: Optional[torch.Generator] = None):
    noise = _randn_like(tensor, generator=generator) * sigma
    out = tensor + noise
    return out.clamp(0, 1) if clip else out


def _sp_noise(tensor, amount, r_ratio, clip, generator: Optional[torch.Generator] = None):
    if not 0 <= amount <= 1:
        raise ValueError('amount must be within [0, 1].')
    if not 0 <= r_ratio <= 1:
        raise ValueError('ratio must be within [0, 1].')
    rand = _rand_like(tensor, generator=generator)
    salt = rand < amount * r_ratio
    pepper = (rand >= amount * r_ratio) & (rand < amount)
    out = tensor.clone()
    out[salt] = 1
    out[pepper] = 0
    return out.clamp(0, 1) if clip else out


def _poisson_noise(tensor, clip, generator: Optional[torch.Generator] = None):
    scaled = (tensor * 255).clamp(0, 255)
    noisy = torch.poisson(scaled)
    out = noisy / 255.0
    return out.clamp(0, 1) if clip else out


def _randn_like(tensor, generator: Optional[torch.Generator] = None):
    if generator is None:
        return torch.randn_like(tensor)
    return torch.randn(
        tensor.shape,
        dtype=tensor.dtype,
        layout=tensor.layout,
        device=tensor.device,
        generator=generator,
    )


def _rand_like(tensor, generator: Optional[torch.Generator] = None):
    if generator is None:
        return torch.rand_like(tensor)
    return torch.rand(
        tensor.shape,
        dtype=tensor.dtype,
        layout=tensor.layout,
        device=tensor.device,
        generator=generator,
    )
