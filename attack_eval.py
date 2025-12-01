import argparse
import random
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from attack_model import Unet
from dataset import UTKFaceDataset
from eval_noise import cal_nme, build_model

# ================= Configuration =================
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# 默认路径配置
DEFAULT_PATHS = {
    'clean_csv': 'data/landmarks_dataset.csv',
    'vit_ckpt': 'checkpoints/vit/best_model.pth',
    'resnet_ckpt': 'checkpoints/resnet18/best_model.pth',
}
# =================================================


class PairedDataset(torch.utils.data.Dataset):
    """Dataset delivering triplets: Noisy (Input), Clean (Target), Landmarks (GT)"""
    def __init__(self, clean_csv, noisy_csv, imgsize=224, do_warp=True):
        self.clean = UTKFaceDataset(clean_csv, imgsize=imgsize, do_warp=do_warp)
        self.noisy = UTKFaceDataset(noisy_csv, imgsize=imgsize, do_warp=do_warp)
        self.length = min(len(self.clean), len(self.noisy))
        if len(self.clean) != len(self.noisy):
            print(f"[Warn] Dataset length mismatch. Truncating to {self.length}")
            
        self.names = self.clean.df['filepath'].astype(str).apply(lambda x: Path(x).name).tolist()[:self.length]
        self.name_to_idx = {name: idx for idx, name in enumerate(self.names)}

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        noisy_img, _ = self.noisy[idx]
        clean_img, clean_lm = self.clean[idx]
        return noisy_img, clean_img, clean_lm, self.names[idx]

    def resolve_name(self, name: str):
        key = Path(name).name
        if key not in self.name_to_idx:
            print(f"[Warn] Image '{name}' not found. Using random sample.")
            return random.choice(self.names)
        return key


def load_attack_model(ckpt_path, device):
    print(f"Loading Attack U-Net: {ckpt_path}")
    model = Unet().to(device)
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Attack checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def load_landmark_model(ckpt_path, backbone, device, lora_adapter=''):
    print(f"Loading Landmark Model ({backbone}): {ckpt_path}")
    if not Path(ckpt_path).exists():
        print(f"[Skip] Checkpoint not found: {ckpt_path}")
        return None
        
    args = SimpleNamespace(
        checkpoint=ckpt_path,
        backbone=backbone, 
        use_lora=False, 
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        lora_adapter=lora_adapter,
    )
    return build_model(args, device)


def denorm(tensor):
    """Normalized [-2, 2] -> [0, 1]"""
    return torch.clamp(tensor * IMAGENET_STD.to(tensor.device) + IMAGENET_MEAN.to(tensor.device), 0.0, 1.0)

def norm(tensor):
    """[0, 1] -> Normalized [-2, 2]"""
    return (tensor - IMAGENET_MEAN.to(tensor.device)) / IMAGENET_STD.to(tensor.device)


def tensor_to_image(tensor):
    return tensor.cpu().detach()


def to_pixel_coords(norm_lms, width, height):
    coords = norm_lms.view(-1, 2).detach().cpu().numpy()
    coords[:, 0] = np.clip(coords[:, 0] * width, 0, width)
    coords[:, 1] = np.clip(coords[:, 1] * height, 0, height)
    return coords


def build_sample_record(label, clean_norm, noisy_norm, recon_01, gt_t,
                        preds_main_clean, preds_main_noisy, preds_main_recon,
                        preds_extra_clean=None, preds_extra_noisy=None, preds_extra_recon=None):
    
    clean_01 = denorm(clean_norm.unsqueeze(0)).squeeze(0)
    noisy_01 = denorm(noisy_norm.unsqueeze(0)).squeeze(0)
    
    record = {
        'label': label,
        'images': {
            'clean': clean_01.cpu().detach(),
            'noisy': noisy_01.cpu().detach(),
            'recon': recon_01.cpu().detach(),
        },
        'gt': gt_t.detach().cpu(),
        'preds_main': {
            'clean': preds_main_clean.detach().cpu(),
            'noisy': preds_main_noisy.detach().cpu(),
            'recon': preds_main_recon.detach().cpu(),
        },
        'preds_extra': None,
    }
    if preds_extra_clean is not None:
        record['preds_extra'] = {
            'clean': preds_extra_clean.detach().cpu(),
            'noisy': preds_extra_noisy.detach().cpu(),
            'recon': preds_extra_recon.detach().cpu(),
        }
    return record


def plot_landmark_comparison(sample, main_name, extra_name, noise_mode, save_dir):
    models_to_plot = [(main_name, sample['preds_main'])]
    if sample['preds_extra'] is not None:
        models_to_plot.append((extra_name, sample['preds_extra']))
        
    cols = ['clean', 'noisy', 'recon']
    col_titles = ['Original (Clean)', f'Protected ({noise_mode})', 'Attacked (Recon)']
    
    fig, axes = plt.subplots(len(models_to_plot), 3, figsize=(12, 4.5 * len(models_to_plot)))
    axes = np.atleast_2d(axes)
    stem = Path(sample['label']).stem

    for r, (model_label, preds_dict) in enumerate(models_to_plot):
        for c, key in enumerate(cols):
            ax = axes[r, c]
            
            img_tensor = sample['images'][key].permute(1, 2, 0).numpy()
            h, w = img_tensor.shape[:2]
            ax.imshow(img_tensor)
            
            gt = to_pixel_coords(sample['gt'], w, h)
            ax.scatter(gt[:, 0], gt[:, 1], c='#00FF00', s=40, label='Ground Truth')
            
            pred = to_pixel_coords(preds_dict[key], w, h)
            ax.scatter(pred[:, 0], pred[:, 1], c='cyan', marker='x', s=60, linewidth=2, label=f'{model_label} Pred')
            
            ax.set_xlim([0, w])
            ax.set_ylim([h, 0])
            ax.axis('off')
            
            if c == 0:
                ax.text(-0.1, 0.5, f"{model_label}\nModel", transform=ax.transAxes, 
                        va='center', ha='right', fontsize=14, fontweight='bold', rotation=90)
            
            if r == 0:
                ax.set_title(col_titles[c], fontsize=14, pad=10)
                
            if r == 0 and c == 2:
                ax.legend(loc='upper right', fontsize=8, framealpha=0.8)

    plt.tight_layout()
    out_path = save_dir / f'Vis_{noise_mode}_{stem}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'✅ Visualization saved to: {out_path}')


def collect_sample(dataset, idx, attack, landmark, extra_landmark, device):
    noisy_norm, clean_norm, lms, label = dataset[idx]
    
    noisy_norm = noisy_norm.unsqueeze(0).to(device)
    clean_norm = clean_norm.unsqueeze(0).to(device)
    lms = lms.unsqueeze(0).to(device)

    with torch.no_grad():
        preds_main_clean = landmark(clean_norm)
        
        noisy_01 = denorm(noisy_norm)
        recon_01 = attack(noisy_01).clamp(0, 1)
        recon_norm = norm(recon_01)
        
        preds_main_noisy = landmark(noisy_norm)
        preds_main_recon = landmark(recon_norm)
        
        preds_extra_clean = preds_extra_noisy = preds_extra_recon = None
        if extra_landmark:
            preds_extra_clean = extra_landmark(clean_norm)
            preds_extra_noisy = extra_landmark(noisy_norm)
            preds_extra_recon = extra_landmark(recon_norm)

    return build_sample_record(
        label,
        clean_norm.squeeze(0).cpu(),
        noisy_norm.squeeze(0).cpu(),
        recon_01.squeeze(0).cpu(),
        lms.squeeze(0).cpu(),
        preds_main_clean.squeeze(0),
        preds_main_noisy.squeeze(0),
        preds_main_recon.squeeze(0),
        preds_extra_clean.squeeze(0) if preds_extra_clean is not None else None,
        preds_extra_noisy.squeeze(0) if preds_extra_noisy is not None else None,
        preds_extra_recon.squeeze(0) if preds_extra_recon is not None else None,
    )


def evaluate(args):
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # 1. Path Inference
    noisy_csv = args.noisy_csv if args.noisy_csv else f'data/landmarks_dataset_{args.noise_mode}.csv'
    attack_ckpt = args.attack_ckpt if args.attack_ckpt else f'attack_checkpoints/{args.noise_mode}/Unet_epoch30.pth'
    
    print("\n=== Evaluation Config ===")
    print(f"Noise Mode:   {args.noise_mode}")
    print(f"Clean CSV:    {args.clean_csv}")
    print(f"Noisy CSV:    {noisy_csv}")
    print(f"Attack Ckpt:  {attack_ckpt}")
    print(f"Output Dir:   {args.save_dir}")
    
    dataset = PairedDataset(args.clean_csv, noisy_csv, imgsize=args.imgsize, do_warp=not args.no_warp)
    attack = load_attack_model(attack_ckpt, device)
    
    landmark = load_landmark_model(args.landmark_ckpt, args.backbone, device, args.lora_adapter)
    if landmark is None: return 

    extra_landmark = None
    extra_name = ''
    if args.extra_landmark_ckpt:
        extra_name = args.extra_landmark_name
        print(f"Extra Model: {args.extra_backbone} ({args.extra_landmark_ckpt})")
        extra_landmark = load_landmark_model(
            args.extra_landmark_ckpt,
            args.extra_backbone,
            device,
            args.extra_lora_adapter,
        )

    main_name = args.landmark_name
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Plot Only Mode
    if args.plot_only:
        target_label = args.sample_image
        if not target_label:
            target_label = random.choice(dataset.names)
        
        print(f"\nVisualizing Sample: {target_label}...")
        idx = dataset.name_to_idx.get(target_label)
        if idx is None: idx = 0
            
        sample = collect_sample(dataset, idx, attack, landmark, extra_landmark, device)
        plot_landmark_comparison(sample, main_name, extra_name, args.noise_mode, save_dir)
        return

    # === Full Evaluation Loop (Metrics) ===
    print(f"\nStarting full evaluation on {len(dataset)} images...")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    pix_mse = 0.0
    
    # Metrics accumulators for Main Model
    main_metrics = {'mse_clean': 0.0, 'mse_recon': 0.0, 'nme_clean': 0.0, 'nme_recon': 0.0}
    # Metrics accumulators for Extra Model
    extra_metrics = {'mse_clean': 0.0, 'mse_recon': 0.0, 'nme_clean': 0.0, 'nme_recon': 0.0}
    
    total = 0

    with torch.no_grad():
        for noisy_norm, clean_norm, lms, labels in tqdm(loader, desc='Evaluating'):
            noisy_norm = noisy_norm.to(device, non_blocking=True)
            clean_norm = clean_norm.to(device, non_blocking=True)
            lms = lms.to(device, non_blocking=True)
            bs = noisy_norm.size(0)
            total += bs

            # Attack Pipeline
            noisy_01 = denorm(noisy_norm)
            recon_01 = attack(noisy_01).clamp(0, 1)
            recon_norm = norm(recon_01)
            
            clean_01 = denorm(clean_norm)
            batch_pix = F.mse_loss(recon_01, clean_01, reduction='mean').item()
            pix_mse += batch_pix * bs

            # --- Main Model Eval ---
            preds_clean = landmark(clean_norm)
            preds_recon = landmark(recon_norm)
            
            main_metrics['mse_clean'] += F.mse_loss(preds_clean, lms, reduction='mean').item() * bs
            main_metrics['mse_recon'] += F.mse_loss(preds_recon, lms, reduction='mean').item() * bs
            main_metrics['nme_clean'] += cal_nme(preds_clean.cpu(), lms.cpu()) * bs
            main_metrics['nme_recon'] += cal_nme(preds_recon.cpu(), lms.cpu()) * bs

            # --- Extra Model Eval ---
            if extra_landmark:
                preds_ex_clean = extra_landmark(clean_norm)
                preds_ex_recon = extra_landmark(recon_norm)
                
                extra_metrics['mse_clean'] += F.mse_loss(preds_ex_clean, lms, reduction='mean').item() * bs
                extra_metrics['mse_recon'] += F.mse_loss(preds_ex_recon, lms, reduction='mean').item() * bs
                extra_metrics['nme_clean'] += cal_nme(preds_ex_clean.cpu(), lms.cpu()) * bs
                extra_metrics['nme_recon'] += cal_nme(preds_ex_recon.cpu(), lms.cpu()) * bs

            # Save one triplet visual from the first batch
            if total == bs:
                sample = collect_sample(dataset, 0, attack, landmark, extra_landmark, device)
                plot_landmark_comparison(sample, main_name, extra_name, args.noise_mode, save_dir)

    print('\n' + '='*40)
    print(f'ATTACK REPORT: {args.noise_mode.upper()}')
    print('='*40)
    print(f'Reconstruction Pixel MSE: {pix_mse / total:.6f}')
    print('-'*40)
    
    print(f'Model: {main_name}')
    print(f'  Clean NME: {main_metrics["nme_clean"] / total:.4f} | MSE: {main_metrics["mse_clean"] / total:.6f}')
    print(f'  Recon NME: {main_metrics["nme_recon"] / total:.4f} | MSE: {main_metrics["mse_recon"] / total:.6f}')
    
    if extra_landmark:
        print('-'*40)
        print(f'Model: {extra_name}')
        print(f'  Clean NME: {extra_metrics["nme_clean"] / total:.4f} | MSE: {extra_metrics["mse_clean"] / total:.6f}')
        print(f'  Recon NME: {extra_metrics["nme_recon"] / total:.4f} | MSE: {extra_metrics["mse_recon"] / total:.6f}')
    print('='*40)


def parse_args():
    parser = argparse.ArgumentParser(description='Simpler Attack Evaluation')
    
    # Required
    parser.add_argument('--noise-mode', required=True, choices=['gaussian', 'salt_pepper', 'poisson'])
    
    # Defaults
    parser.add_argument('--clean-csv', default=DEFAULT_PATHS['clean_csv'])
    parser.add_argument('--noisy-csv', default='') 
    parser.add_argument('--attack-ckpt', default='') 
    
    # ViT Defaults
    parser.add_argument('--landmark-ckpt', default=DEFAULT_PATHS['vit_ckpt'])
    parser.add_argument('--backbone', default='vit')
    parser.add_argument('--landmark-name', default='ViT (Main)')
    parser.add_argument('--lora-adapter', default='')

    # ResNet Defaults
    parser.add_argument('--extra-landmark-ckpt', default=DEFAULT_PATHS['resnet_ckpt'])
    parser.add_argument('--extra-backbone', default='resnet18')
    parser.add_argument('--extra-landmark-name', default='ResNet (Compare)')
    parser.add_argument('--extra-lora-adapter', default='')

    # Misc
    parser.add_argument('--imgsize', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', default='')
    parser.add_argument('--no-warp', action='store_true')
    # 修改了默认 save-dir
    parser.add_argument('--save-dir', default='attack_eval')
    parser.add_argument('--sample-image', default='')
    parser.add_argument('--sample-seed', type=int, default=1234)
    # 默认关闭 plot-only，以便输出 Metrics
    parser.add_argument('--plot-only', action='store_true', help='Set to skip metrics and only plot')
    
    return parser.parse_args()


if __name__ == '__main__':
    evaluate(parse_args())