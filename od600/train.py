from transforms import get_transform

transform = get_transform()
import os, math
import pandas as pd
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class ODImageDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, log_transform=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.log = log_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img_path = os.path.join(self.img_dir, row['image'])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        od = float(row['od600'])
        if self.log:
            od = math.log(od + 1e-6)
        return img, torch.tensor([od], dtype=torch.float32)

class RegressionModel(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True, freeze_backbone=False, dropout_p=0.3):
        super().__init__()
        if backbone == 'resnet18':
            self.net = models.resnet18(pretrained=pretrained)
            nfeat = self.net.fc.in_features
            self.net.fc = nn.Identity()
        else:
            self.net = models.resnet50(pretrained=pretrained)
            nfeat = self.net.fc.in_features
            self.net.fc = nn.Identity()
        if freeze_backbone:
            for p in self.net.parameters():
                p.requires_grad = False
        self.reg = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(nfeat, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        feat = self.net(x)
        out = self.reg(feat)
        return out

def train_epoch(model, loader, optimizer, device, criterion):
    model.train()
    total_loss = 0.0
    for imgs, targets in loader:
        imgs = imgs.to(device); targets = targets.to(device)
        preds = model(imgs)
        loss = criterion(preds, targets)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

def eval_model(model, loader, device, criterion, log_transform=False):
    model.eval()
    ys, yps = [], []
    total_loss = 0.0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device); targets = targets.to(device)
            preds = model(imgs)
            total_loss += criterion(preds, targets).item() * imgs.size(0)
            ys.append(targets.cpu().numpy())
            yps.append(preds.cpu().numpy())
    y = np.vstack(ys).reshape(-1)
    yp = np.vstack(yps).reshape(-1)
    if log_transform:
        y = np.exp(y)
        yp = np.exp(yp)
    rmse = mean_squared_error(y, yp) ** 0.5
    r2 = r2_score(y, yp)
    return total_loss/len(loader.dataset), rmse, r2, y, yp

if __name__ == '__main__':
    df = pd.read_csv('labels.csv')
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    train_ds = ODImageDataset(train_df, './imgs', transform=transform, log_transform=True)
    val_ds = ODImageDataset(val_df, './imgs', transform=transform, log_transform=True)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RegressionModel('resnet18', pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    best_rmse = 1e9

    for epoch in range(1, 31):
        tr_loss = train_epoch(model, train_loader, optimizer, device, criterion)
        val_loss, rmse, r2, y_true, y_pred = eval_model(model, val_loader, device, criterion, log_transform=True)
        print(f"Epoch {epoch} train_loss={tr_loss:.4f} val_loss={val_loss:.4f} RMSE={rmse:.4f} R2={r2:.4f}")
        if rmse < best_rmse:
            best_rmse = rmse
            torch.save(model.state_dict(), 'best_model.pt')