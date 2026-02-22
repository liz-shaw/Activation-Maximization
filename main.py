import argparse
import torch
from torchvision import models
from core.act_max import ActMaxAnalyzer
from utils.image_utils import save_tensor_as_image 

def main():
    parser = argparse.ArgumentParser(description="Activation Maximization Analyzer")

    # 定义命令行参数
    parser.add_argument("--layer", type=int, default=17, help="Target layer index")
    parser.add_argument("--unit", type=int, default=511, help="Target neuron unit index")
    parser.add_argument("--steps", type=int, default=100, help="Number of optimization steps")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for image optimization")
    parser.add_argument("--model", type=str, default="vgg16", help="Pretrained model name")

    args = parser.parse_args()

    # 初始化预训练模型
    # 以后可以在此处扩展对 ResNet 或 Inception 的支持
    if args.model == "vgg16":
        weights = models.VGG16_Weights.IMAGENET1K_V1
        model = models.vgg16(weights=weights)
    else:
        # 默认回退到 VGG16
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    # 启动分析器并执行图像生成流程
    # 内部通过 Hook 捕捉 args.layer 的激活值
    analyzer = ActMaxAnalyzer(model, layer_index=args.layer)
    
    print(f"Status: Optimizing {args.model} layer {args.layer} unit {args.unit}...")
    result_img = analyzer.generate_image(
        unit_index=args.unit, 
        steps=args.steps, 
        lr=args.lr
    )

    # 基于实验参数自动构建输出路径
    # 确保实验结果的可追溯性，避免文件名冲突
    file_name = f"{args.model}_L{args.layer}_U{args.unit}_S{args.steps}.jpg"
    
    save_tensor_as_image(result_img, path="results", filename=file_name)
    print(f"Success: Visualization saved to results/{file_name}")

if __name__ == "__main__":
    main()