from train import model, data_transform,cur_path,device,model_path
import torch
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# 加载模型
model.load_state_dict(torch.load(model_path))
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
image_dir = os.path.join(cur_path, 'owndata')
fig = plt.figure(figsize=(20, 20))
# 遍历目录中的所有图片
for filename in os.listdir(image_dir):
    if filename.endswith('.jpg'):  # 只处理jpg图片，你可以根据需要修改
        # 加载图片
        image = Image.open(os.path.join(image_dir, filename))
        # 对图片进行预处理
        transformed_image = data_transform['test'](image)
        
        # mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
        # std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

        # # 将图像移动到 CPU，并转换为 numpy 数组
        # transformed_image_np = transformed_image.cpu().numpy()

        # # 对图像进行反归一化
        # transformed_image_np = std * transformed_image_np + mean

        # # PyTorch 的图像是 (C, H, W)，而 matplotlib 需要 (H, W, C)，所以需要进行转置
        # transformed_image_np = np.transpose(transformed_image_np, (1, 2, 0))

        # # 确保图像的值在 [0, 1] 范围内
        # transformed_image_np = np.clip(transformed_image_np, 0, 1)

        # # 使用 matplotlib 显示图像
        # plt.imshow(transformed_image_np)
        # plt.show()
        
        # 添加一个额外的维度来表示批次大小，并将图片移动到设备上
        transformed_image = transformed_image.unsqueeze(0).to(device)

        # 使用模型进行预测
        with torch.no_grad():
            prediction = model(transformed_image)

        # 获取预测结果
        predicted_class = prediction.argmax(dim=1).item()
        # 根据预测结果重命名图片
        new_filename = f'class_{predicted_class}_{filename}'
        os.rename(os.path.join(image_dir, filename), os.path.join(image_dir, new_filename))
