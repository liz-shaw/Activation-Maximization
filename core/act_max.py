# Activation Maximization
# 是激活值最大化部分反向传播的实现， 过反向传播来最大化特定神经元的激活值，从而生成输入图像
# 详情看DLFBP:21.9.2节
import torch
import torch.nn as nn


# 依旧建立类  
class ActMaxAnalyzer:
    """
    这个类的功能
        1. 接收一个训练好的模型
        2. 接收层级、神经元、卷积核等的索引信息，明确具体要最大化哪个神经元的激活值
        3. 通过反向传播来最大化这个神经元的激活值，从而生成输入图像
    
    这个类需要保存的信息
        1. 模型本身
        2. 目标层级、神经元、卷积核等的索引信息
        3. 激活值

    这个类中重要的核心功能：
        1. hook函数：用来捕捉目标神经元的激活值.相当于流体力学中的皮托管，捕捉流体的速度和压力
        2. 反向传播函数：用来最大化目标神经元的激活值.
        3. 图像生成函数：用来生成输入图像，通常是从随机噪声开始，然后通过反向传播来优化这个图像，使得目标神经元的激活值最大化.

    具体的过程看笔记，好了现在开始手搓代码
    """
    def __init__(self, model, layer_index):
        self.model = model
        self.model.eval()  # 设置模型为评估模式,不再更新权重,只关注激活值的变化
        
        self.target_layer = self.model.features[layer_index]  #直接取到目标层级

        self.captured_activation = None  # 用来保存捕捉到的激活值,初始值啥也没有

        self._setup_hook()  # 设置hook函数来捕捉激活值


    def _setup_hook(self):
        # 内部私有函数
        """
        input 
            就是输入图像和标签等信息，self就行了
        output
            就是目标层级的输出，也就是我们要捕捉的激活值
        """
        def hook_fn(module, input, output):
            self.captured_activation = output  # 捕捉到的激活值保存在self.captured_activation中
        self.target_layer.register_forward_hook(hook_fn)  # 注册前向传播的hook函数
       #这个register_forward_hook函数
       # 1. 接收一个函数作为参数，这个函数就是我们定义的hook_fn
       # 2. 在hook_fn函数中，我们将捕捉到的激活值保存在self.captured_activation中，以便后续使用。
        

    
    def generate_image(self, unit_index, steps=100, lr=0.1):
        """
        :param unit_index: 这是神经元的索引
        :param steps: 这是更新迭代次数，通常需要多次迭代来优化输入图像
        :param lr: 学习率，控制每次迭代更新输入图像的步长
        """
        input_img = (torch.randn(1, 3, 224, 224)*0.05+0.5).requires_grad_(True)
        # 从随机噪声开始生成输入图像,1是batch size,3是通道数,224x224是图像尺寸
        # requires_grad=True表示这个张量需要计算梯度
        # +0.5是为了让初始图像的像素值在0.5附近，避免ReLU导致神经元死亡

        optimizer = torch.optim.Adam([input_img], lr=lr)  # 使用Adam优化器来更新输入图像

        for step in range(steps):
            optimizer.zero_grad()  # 清除之前的梯度
            self.model(input_img-0.5)  
            # 前向传播，触发hook函数来捕捉激活值
            # 注意这里我们输入的图像是input_img-0.5，因为我们在初始化时加了0.5，所以这里要减去0.5来还原到原始范围
            loss = -self.captured_activation[:, unit_index].mean()  # 计算损失，目标是最大化激活值，所以取负号

            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新输入图像

            input_img.data.clamp_(0, 1)  # 将输入图像的像素值限制在0到1之间
            if (step + 1) % 10 == 0:
                print(f"Step [{step+1}/{steps}], 激活值强度: {-loss.item():.4f}")

        return input_img.detach()  # 返回生成的输入图像，detach()表示不再计算梯度
        