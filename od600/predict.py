from model import RegressionModel
import math
from PIL import Image
from transforms import get_transform

transform = get_transform()
import torch

# load model and infer
model = RegressionModel(pretrained=False)
model.load_state_dict(torch.load('best_model.pt', map_location='cpu'))
model.eval()
# 이미지 전처리 동일하게 적용
img = Image.open('tests/test_3.jpg').convert('RGB')
inp = transform(img).unsqueeze(0)
with torch.no_grad():
    pred_log = model(inp).item()
pred_od = math.exp(pred_log)  # 로그 변환 사용 시
print("Predicted OD600:", pred_od)