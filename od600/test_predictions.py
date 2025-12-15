import os
import math
import pandas as pd
import torch
import numpy as np
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from model import RegressionModel
from transforms import get_transform
import matplotlib.pyplot as plt

TEST_IMG_DIR = "tests"
TEST_LABEL_CSV = "test_labels.csv"
MODEL_PATH = "best_model.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RegressionModel(pretrained=False).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

transform = get_transform()

df = pd.read_csv(TEST_LABEL_CSV)

y_true = []
y_pred = []

results = []

with torch.no_grad():
    for _, row in df.iterrows():
        img_name = row["image"]
        true_od = float(row["od600"])

        img_path = os.path.join(TEST_IMG_DIR, img_name)
        img = Image.open(img_path).convert("RGB")

        x = transform(img).unsqueeze(0).to(DEVICE)
        pred_log = model(x).item()
        pred_od = math.exp(pred_log)  # log 학습 가정

        y_true.append(true_od)
        y_pred.append(pred_od)

        results.append({
            "image": img_name,
            "true_od": true_od,
            "pred_od": pred_od,
            "abs_error": abs(pred_od - true_od),
            "rel_error_%": abs(pred_od - true_od) / true_od * 100 if true_od > 0 else abs(pred_od - true_od)
        })

y_true = np.array(y_true)
y_pred = np.array(y_pred)

mae = mean_absolute_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred) ** 0.5
r2 = r2_score(y_true, y_pred)

print("===== Test Set Performance =====")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R^2  : {r2:.4f}")

result_df = pd.DataFrame(results)
result_df.to_csv("test_predictions.csv", index=False)
print("\nSaved detailed results to test_predictions.csv")

plt.figure()
plt.scatter(y_true, y_pred)
plt.plot(
    [y_true.min(), y_true.max()],
    [y_true.min(), y_true.max()]
)

plt.xlabel("True OD600")
plt.ylabel("Predicted OD600")
plt.title("True vs Predicted OD600")

plt.show()