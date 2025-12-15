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
    df = pd.read_csv('labels.csv')  # labels.csv 파일에 저장된 이미지-OD600 쌍을 불러온다.
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    # 전체 데이터셋중 20%가 검증 과정에 이용되고, 80%가 학습에 사용되도록 한다.

    transform = T.Compose([  # Resnet에 사용할 수 있도록 이미지를 전처리하는 과정을 정의한다.
        T.Resize((224, 224)),                 # ResNet 입력 크기
        T.ToTensor(),                         # PIL → Tensor (0~1)
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    train_ds = ODImageDataset(
        train_df,
        'imgs',
        transform=transform,  # 주어진 이미지를 위에서 정의한 방법을 통해 전처리한다.
        log_transform=True   # OD600의 분포 안정화를 위해 log 변환한다.
    )

    val_ds = ODImageDataset(  # 위와 같은 과정이다.
        val_df,
        'imgs',
        transform=transform,
        log_transform=True
    )

    train_loader = DataLoader(  # Data를 불러오는 과정이다.
        train_ds,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=32,
        shuffle=False,
        num_workers=2
    )
    device = torch.device('cpu')  # 학습은 cpu로 진행하였다.(데이터셋이 작고, 모델이 가볍다.)

    model = RegressionModel(
        backbone='resnet18',
        pretrained=True
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    target_r2 = 0.90        # 학습이 종료되는 최소 조건 R²값
    patience = 30           # 해당 변수값만큼의 에포크동안 최대 R²값이 갱신되지 않으면, 가장 높은 R²값을 보인 모델을 반환한다.
    best_r2 = -1e9          # 현재까지의 최대 R²값을 저장할 변수이다.
    patience_counter = 0   # patience보다 커지는지 확인하기 위한 변수이다.

    epoch = 0
    while True:
        epoch += 1
        train_loss = train_epoch(  # 학습 과정이다.
            model,
            train_loader,
            optimizer,
            device,
            criterion
        )

        val_loss, rmse, r2, y_true, y_pred = eval_model(
            model,
            val_loader,
            device,
            criterion,
            log_transform=True
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"RMSE={rmse:.4f} | "
            f"R2={r2:.4f}"
        )

        if r2 > best_r2:  # 위에서 설정한 하이퍼파라미터 값에 따라 정지하고, 모델을 반환하도록 한다.
            best_r2 = r2
            patience_counter = 0

            torch.save(model.state_dict(), 'model.pt')  # else문의 조건에 따라 프로그램이 조기 종료되었을 때 반환할 모델을 저장한다.
        else:
            patience_counter += 1
        if best_r2 >= target_r2:
            print(f"Target R2({target_r2}) reached. Training stopped.")
            break
        if patience_counter >= patience:
            print("Early stopping triggered (no R2 improvement).")
            print(f"largest R2: {best_r2:.4f}")
            break