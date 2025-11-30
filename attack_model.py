import torch.nn as nn
import torch
import cv2
import argparse 
from pathlib import Path
import os 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import numpy as np
from tqdm import tqdm
from dataset import UTKFaceDataset
import torch.nn.functional as F
# model
class UnetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down=True, act='relu', use_dropout=False):
        super().__init__()
        layers = []
        if down:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False), # kernel size=4, stride=2, padding=1
                nn.BatchNorm2d(out_ch)
            ])
        else:
            layers.extend([
                nn.ConvTranspose2d(in_ch, out_ch, 4 ,2, 1, bias=False),
                nn.BatchNorm2d(out_ch)
            ])

        if act == 'relu':
            layers.append(nn.ReLU(inplace=True))
        else:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)



class Unet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3):
        super().__init__()
        self.down1 = UnetBlock(in_ch, 64, down=True, act='lrelu', use_dropout=False)
        self.down2 = UnetBlock(64, 128, down=True, act='lrelu')
        self.down3 = UnetBlock(128, 256, down=True, act='lrelu')
        self.down4 = UnetBlock(256, 512, down=True, act='lrelu')
        self.down5 = UnetBlock(512, 512, down=True, act='lrelu')
        self.bridge = UnetBlock(512, 512, down=True, act='lrelu')

        # skip connection
        self.up5 = UnetBlock(512, 512, down=False, use_dropout=True) 
        self.up4 = UnetBlock(1024, 512, down=False, use_dropout=True) # u5 + d5
        self.up3 = UnetBlock(1024, 256, down=False) # u4 + d4
        self.up2 = UnetBlock(512, 128, down=False) # u3 + d3
        self.up1 = UnetBlock(256, 64, down=False)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_ch, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        bridge = self.bridge(d5)

        u5 = self.up5(bridge)
        u5 = F.interpolate(u5, size=d5.shape[2:], mode='bilinear', align_corners=False)
        u4 = self.up4(torch.cat([u5, d5], dim=1))
        u4 = F.interpolate(u4, size=d4.shape[2:], mode='bilinear', align_corners=False)
        u3 = self.up3(torch.cat([u4, d4], dim=1))
        u4 = F.interpolate(u3, size=d3.shape[2:], mode='bilinear', align_corners=False)
        u2 = self.up2(torch.cat([u3, d3], dim=1))
        u4 = F.interpolate(u2, size=d2.shape[2:], mode='bilinear', align_corners=False)
        u1 = self.up1(torch.cat([u2, d2], dim=1))
        u4 = F.interpolate(u1, size=d1.shape[2:], mode='bilinear', align_corners=False)

        out = self.final(torch.cat([u1, d1], dim=1))
        u4 = F.interpolate(out, size=out.shape[2:], mode='bilinear', align_corners=False)

        return (out + 1) / 2 # map back to [0, 1]
    

# dataset

class PairedNoiseDataset(Dataset):
    def __init__(self, clean_csv, noisy_csv, imgsize=224):
        self.clean = UTKFaceDataset(clean_csv, imgsize=imgsize, do_warp=True)
        self.noisy = UTKFaceDataset(noisy_csv, imgsize=imgsize, do_warp=True)
        assert len(self.clean) == len(self.noisy)

    def __len__(self):
        return len(self.clean)
    
    def __getitem__(self, idx):
        clean_img, _ = self.clean[idx]
        noisy_img, _ = self.noisy[idx]
        return noisy_img, clean_img
    



# ----------------------------Training---------------------------------
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = PairedNoiseDataset(args.clean_csv, args.noisy_csv, imgsize=args.imgsize)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model =Unet().to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        epoch_loss = 0
        for noisy, clean in pbar:
            noisy, clean = noisy.to(device), clean.to(device)
            preds = model(noisy)
            loss = criterion(preds, clean)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss':f'{loss.item():.4f}'})
        print(f'Epoch{epoch+1}/{args.epochs}: Loss={epoch_loss/len(loader):.4f}')
        torch.save(model.state_dict(), os.path.join(args.save_dir, f'Unet_epoch{epoch+1}.pth'))

    # save sample comparsions
    model.eval()
    noisy, clean = next(iter(loader))
    noisy = noisy.to(device)
    with torch.no_grad():
        recon = model(noisy).cpu()
    
    save_image(clean, os.path.join(args.save_dir, 'sample_clean.png'))
    save_image(noisy.cpu(), os.path.join(args.save_dir, 'sample_noisy.png'))
    save_image(recon, os.path.join(args.save_dir, 'sample_recon.png'))



#--------------------------------CLI-------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean-csv', type=str, default='data/landmarks_dataset.csv')
    parser.add_argument('--noisy-csv', type=str, required=True, help='paired noisy csv (e.g., landmarks_dataset_gaussian.csv)')
    parser.add_argument('--imgsize', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--save-dir', type=str, default='attack_checkpoints')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
