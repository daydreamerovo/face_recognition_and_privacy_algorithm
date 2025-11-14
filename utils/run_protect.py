import argparse
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.add_noise import get_noise_generator, lowpass  # noqa: E402

NOISE_CHOICES = ['gaussian', 'salt_pepper', 'poisson']


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
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
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
