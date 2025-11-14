import os
import json
from typing import Optional

import torch
import torch.nn as nn

from baseline_models_prep import LandmarkModel


def _load_adapter_config(lora_dir: str) -> dict:
    cfg_path = os.path.join(lora_dir, 'adapter_config.json')
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"adapter_config.json not found in: {lora_dir}")
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _load_adapter_state(lora_dir: str) -> dict:
    # Prefer safetensors if present
    st_path = os.path.join(lora_dir, 'adapter_model.safetensors')
    if os.path.isfile(st_path):
        try:
            from safetensors.torch import load_file as safe_load
            return safe_load(st_path)
        except Exception as e:
            raise RuntimeError(
                f"Found safetensors file but failed to load: {st_path}.\n{e}"
            )
    # Fallback to torch serialized bin
    st_path = os.path.join(lora_dir, 'adapter_model.bin')
    if os.path.isfile(st_path):
        return torch.load(st_path, map_location='cpu')
    raise FileNotFoundError(
        f"No adapter weight file found in: {lora_dir}. Expected adapter_model.safetensors or adapter_model.bin"
    )


def load_lora_into_backbone(backbone_module: nn.Module, lora_dir: str) -> nn.Module:
    """
    Wrap a torchvision backbone with LoRA using the saved adapter under `lora_dir`.

    - Expects `adapter_config.json` and `adapter_model.(safetensors|bin)` in `lora_dir`.
    - Returns a new backbone module wrapped with PEFT LoRA and loaded with adapter weights.
    """
    cfg = _load_adapter_config(lora_dir)
    state = _load_adapter_state(lora_dir)

    try:
        from peft import LoraConfig, TaskType
        from peft.tuners.lora import LoraModel
    except Exception as e:
        raise RuntimeError("PEFT is required to load LoRA adapters. Please `pip install peft`.\n" + str(e))

    task_type_str = cfg.get('task_type', 'FEATURE_EXTRACTION')
    task_type = getattr(TaskType, task_type_str, TaskType.FEATURE_EXTRACTION)

    lora_cfg = LoraConfig(
        r=cfg.get('r', 8),
        lora_alpha=cfg.get('lora_alpha', 16),
        lora_dropout=cfg.get('lora_dropout', 0.05),
        bias=cfg.get('bias', 'none'),
        task_type=task_type,
        target_modules=cfg.get('target_modules', []),
        modules_to_save=cfg.get('modules_to_save', []),
    )

    wrapped = LoraModel(backbone_module, {"default": lora_cfg}, adapter_name="default")

    # Best-effort load: only LoRA tensors were saved
    missing, unexpected = wrapped.load_state_dict(state, strict=False)
    if len(missing) > 0:
        # Warn but continue; base weights are expected to be missing
        print(f"[LoRA] Missing keys when loading adapter: {len(missing)} (expected for base weights)")
    if len(unexpected) > 0:
        print(f"[LoRA] Unexpected keys when loading adapter: {len(unexpected)}")

    return wrapped


def build_model_with_lora(
    backbone: str,
    lora_dir: str,
    pretrained: bool = True,
    num_landmarks: int = 5,
    device: Optional[str] = None,
) -> LandmarkModel:
    """
    Construct a `LandmarkModel`, wrap its backbone with the LoRA adapter from `lora_dir`,
    and return the ready-to-eval model.

    Example:
        model = build_model_with_lora(
            backbone='vit',
            lora_dir='checkpoints/vit/lora',
            pretrained=False,
        )
        model.eval()
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = LandmarkModel(backbone=backbone, pretrained=pretrained, num_landmarks=num_landmarks)
    model.backbone = load_lora_into_backbone(model.backbone, lora_dir)
    return model.to(device)

