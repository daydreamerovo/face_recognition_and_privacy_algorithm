import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from baseline_models_prep import LandmarkModel
from dataset import UTKFaceDataset
import os
import argparse
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument('--backbone', type=str, default='resnet18',
                    choices=['resnet18', 'vit'], help='Model backbone')
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-4)
args = parser.parse_args()

batch_size = args.batch_size
epochs = args.epochs
lr = args.lr
backbone = args.backbone
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Fix random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
try:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except Exception:
    pass

model = LandmarkModel(backbone=backbone, pretrained=True).to(device)
save_dir = f"checkpoints/{backbone}"
os.makedirs(save_dir, exist_ok=True)
best_path = os.path.join(save_dir, "best_model.pth")

# TensorBoard writer
writer = SummaryWriter(log_dir=os.path.join('runs', backbone))
global_step = 0

dataset = UTKFaceDataset('data/landmarks_dataset.csv', imgsize = 224, do_warp=True)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
# deterministic split
g = torch.Generator().manual_seed(42)
train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=g)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds,batch_size=batch_size, shuffle=False)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6
)
def cal_nme(preds, gts):
    preds = preds.view(-1, 5, 2)
    gts = gts.view(-1, 5, 2)
    diff = torch.norm(preds - gts, dim=2)
    dist = torch.norm(gts[:, 0, :] - gts[:, 1, :], dim=1)
    return (diff.mean(dim=1) / dist).mean().item()


best_nme = float('inf')
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for imgs, lms in tqdm(train_loader, desc=f"{epoch+1}/epochs"):
        imgs , lms = imgs.to(device), lms.to(device)
        preds = model(imgs)
        loss = criterion(preds, lms)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()
        # log batch loss
        try:
            writer.add_scalar('loss/train_batch', loss.item(), global_step)
            global_step += 1
        except Exception:
            pass

    model.eval()
    val_loss = 0
    nme_total = 0
    with torch.no_grad():
        for imgs, lms in val_loader:
            imgs , lms = imgs.to(device), lms.to(device)
            preds = model(imgs)
            val_loss += criterion(preds, lms).item()
            nme_total += cal_nme(preds, lms)

    val_loss /= len(val_loader)
    scheduler.step(val_loss)
    mean_nme = nme_total / len(val_loader)

    # log epoch metrics
    try:
        writer.add_scalar('loss/train_epoch', train_loss/len(train_loader), epoch)
        writer.add_scalar('loss/val_epoch', val_loss, epoch)
        writer.add_scalar('nme/val_epoch', mean_nme, epoch)
    except Exception:
        pass

    print(f"[Epoch {epoch+1}] Train Loss: {train_loss/len(train_loader):.4f} | "
          f"Val Loss: {val_loss:.4f} | NME: {mean_nme:.4f}")
    
    if mean_nme < best_nme:
        best_nme = mean_nme
        torch.save(model.state_dict(), best_path)
        print(f"saved new model to: {best_path}")
    
print('training complete, best nme:', best_nme)
try:
    writer.close()
except Exception:
    pass
