import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from baseline_models_prep import LandmarkModel
from dataset import UTKFaceDataset


def cal_nme(preds, gts):
    preds = preds.view(-1, 5, 2)
    gts = gts.view(-1, 5, 2)
    diff = torch.norm(preds - gts, dim=2)
    dist = torch.norm(gts[:, 0, :] - gts[:, 1, :], dim=1)
    return (diff.mean(dim=1) / dist).mean().item()


def build_model(args, device):
    model = LandmarkModel(backbone=args.backbone, pretrained=False).to(device)

    if args.use_lora:
        from peft import LoraConfig, TaskType
        from peft.tuners.lora import LoraModel

        if args.backbone == 'resnet18':
            conv_names = [
                name for name, module in model.backbone.named_modules()
                if isinstance(module, nn.Conv2d)
            ]
            target_modules = conv_names
            modules_to_save = ['fc'] if conv_names else []
        elif args.backbone == 'vit':
            linear_names = []
            for name, module in model.backbone.named_modules():
                if isinstance(module, nn.Linear) and not name.startswith('heads.head'):
                    linear_names.append(name)
            if not linear_names:
                linear_names = ['out_proj', 'fc', 'linear']
            target_modules = linear_names
            modules_to_save = ['heads.head']
        else:
            raise ValueError(f"Unsupported backbone for LoRA: {args.backbone}")

        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias='none',
            task_type=TaskType.FEATURE_EXTRACTION,
            target_modules=target_modules,
            modules_to_save=modules_to_save,
        )
        model.backbone = LoraModel(model.backbone, {"default": lora_cfg}, adapter_name="default")

        if args.lora_adapter and os.path.isdir(args.lora_adapter):
            try:
                model.backbone.load_adapter(args.lora_adapter, "default")
            except Exception as exc:
                raise RuntimeError(f"Failed to load LoRA adapter from {args.lora_adapter}: {exc}") from exc

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on noisy images.")
    parser.add_argument('--meta-path', type=str, required=True, help='CSV/Parquet with noisy image paths')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'vit'])
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--imgsize', type=int, default=224)
    parser.add_argument('--no-warp', action='store_true', help='Disable affine alignment in dataset')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    # LoRA options
    parser.add_argument('--use-lora', action='store_true', help='Expect checkpoint with LoRA layers')
    parser.add_argument('--lora-r', type=int, default=8)
    parser.add_argument('--lora-alpha', type=int, default=16)
    parser.add_argument('--lora-dropout', type=float, default=0.05)
    parser.add_argument('--lora-adapter', type=str, default='', help='Optional directory with saved LoRA adapter')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    dataset = UTKFaceDataset(args.meta_path, imgsize=args.imgsize, do_warp=not args.no_warp)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    criterion = nn.MSELoss()
    model = build_model(args, device)

    mse_total = 0.0
    nme_total = 0.0

    with torch.no_grad():
        for imgs, lms in tqdm(loader, desc='evaluating'):
            imgs = imgs.to(device)
            lms = lms.to(device)
            preds = model(imgs)
            mse_total += criterion(preds, lms).item()
            nme_total += cal_nme(preds, lms)

    mse = mse_total / len(loader)
    nme = nme_total / len(loader)

    print(f"Results on {args.meta_path}:")
    print(f"  MSE: {mse:.4f}")
    print(f"  NME: {nme:.4f}")


if __name__ == '__main__':
    main()
