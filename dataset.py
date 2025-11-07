import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import os


# define useful columns
LM_COLS = [
    "right_eye_x","right_eye_y",
    "left_eye_x","left_eye_y",
    "nose_tip_x","nose_tip_y",
    "right_mouth_x","right_mouth_y",
    "left_mouth_x","left_mouth_y",
]

# dataset class
class UTKFaceDataset(Dataset):
    def __init__(self, meta_path, imgsize=224, do_warp=True):
        """
        meta_path: file path: csv or parquet;
        imgsize: size of output;
        do_wrap: do wrapAffine (or not).
        """
        # Anchor project root to this file's directory
        self.root = Path(__file__).resolve().parent
        if meta_path.endswith('csv'):
            self.df = pd.read_csv(meta_path)
        else:
            self.df = pd.read_parquet(meta_path)
        
        # extract successfully detected imgs
        self.df = self.df[self.df['detection_successful'] == True].copy()
        
        self.transform = transforms.Compose([
            transforms.Resize((imgsize, imgsize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.imgsize = imgsize
        self.do_warp = do_warp
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path_str = str(row['filepath'])

        # Resolve path robustly to handle historical relative entries like '../data/...'
        p = Path(path_str)
        if not p.is_absolute():
            normalized = path_str.replace('\\', '/')
            if normalized.startswith('../'):
                normalized = normalized[3:]
            p = (self.root / normalized).resolve()

        # Fallback: if still missing, try within project data dir using just filename
        if not p.exists():
            alt = (self.root / 'data' / Path(path_str).name).resolve()
            if alt.exists():
                p = alt
            else:
                raise FileNotFoundError(f"Image not found for '{path_str}'. Tried: '{p}' and '{alt}'")

        # read images
        img = np.array(Image.open(str(p)).convert('RGB'))
        h ,w = img.shape[:2]
        
        # landmarks coordinates
        pts = np.array(
            [row[c] for c in LM_COLS], dtype=np.float32
        ).reshape(-1, 2)

        # wrap affine
        if self.do_warp:
            right_eye, left_eye = pts[0], pts[1]
            mouth_center = (pts[3] + pts[4]) / 2

            # transfrom standard
            src = np.stack([right_eye, left_eye, mouth_center])
            dst = np.array([
                [0.35 * w, 0.35 * h],
                [0.65 * w, 0.35 * h],
                [0.45 * w, 0.65 * h]
            ], dtype=np.float32)

            M = cv2.getAffineTransform(src, dst)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
            ones = np.ones((pts.shape[0], 1))
            pts_h = np.hstack([pts, ones])
            pts = (M @ pts_h.T).T # apply trandsform to landmarks

            # normalise landmarks
            pts_norm = pts / np.array([w, h], dtype=np.float32)

            img_t = self.transform(Image.fromarray(img))
            lm_t = torch.tensor(pts_norm.reshape(-1), dtype=torch.float32)

        return img_t, lm_t
