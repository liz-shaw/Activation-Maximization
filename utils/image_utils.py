import torch
import torchvision.transforms as transforms
from PIL import Image
import os

def save_tensor_as_image(tensor, path, filename):
    """
    将 ActMaxAnalyzer 生成的 Tensor 转换为图片并保存
    :param tensor: 形状为 (1, 3, 224, 224) 的张量
    :param path: 存储文件夹路径 (比如 'results/')
    :param filename: 文件名 (比如 'neuron_511.jpg')
    """
    # 1. 确保目录存在
    if not os.path.exists(path):
        os.makedirs(path)

    # 2. 去掉 Batch 维度：从 (1, 3, 224, 224) 变成 (3, 224, 224)
    img_tensor = tensor.squeeze(0)

    # 3. 将 Tensor 转换为 PIL Image 对象
    # transforms.ToPILImage() 会自动处理 0-1 到 0-255 的映射
    to_pil = transforms.ToPILImage()
    image = to_pil(img_tensor)

    # 4. 保存
    full_path = os.path.join(path, filename)
    image.save(full_path)
    print(f"图像已保存至: {full_path}")

