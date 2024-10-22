# GoogLeNet

Link: [Going Deeper with Convolutions](http://arxiv.org/abs/1409.4842).

> We propose a deep convolutional neural network architecture codenamed **Inception**, which was responsible for setting the new **state of the art** for classification and detection in the ImageNet Large-Scale Visual Recognition Challenge 2014 (ILSVRC14). The main hallmark of this architecture is the improved utilization of the computing resources inside the network. This was achieved by a carefully crafted design that allows for increasing the depth and width of the network while keeping the computational budget constant. To optimize quality, the architectural decisions were based on the Hebbian principle and the intuition of multi-scale processing. One particular incarnation used in our submission for ILSVRC14 is called **GoogLeNet**, a 22 layers deep network, the quality of which is assessed in the context of classification and detection.



## Background

提升模型性能的**最直接**方法就是增加模型的参数量，即增加模型的深度和宽度。本文提出了一个 **Inception** 的结构，基于 Inception 的具有 22 层的 **GoogLeNet** 取得了 ILSVRC 2014 分类和检测比赛的第一名，参数量仅仅为 [**AlexNet**](https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) 的 1/12，但性能却得到的显著的提升。本文旨在构建一个**高效**的卷积神经网络，就是通过**更少的参数量**达到更好的结果，同时，更少的参数量也意味着网络的推理速度更快。

下图为 Inception 模块的结构图，图 (a) 是一个比较 naive 的版本，Inception 模块共有 4 条**并行**的路径，分别执行 1x1 卷积、3x3 卷积、5x5 卷积、3x3 最大池化，每个路径得到的特征图的空间大小相同，直接**按通道维度连接**起来得到最终的输出。图 (b) 和图 (a) 的区别在于执行 3x3 卷积、5x5 卷积之前进行了一次 1x1 卷积，目的是使得通道**维度降低**（channels reduction），从而**降低计算量**的目的，此外，3x3 最大池化层后也添加了一个 1x1 卷积，目的也是降低计算量。

![Inception Module](./assets/Inception-Module.png)



## Model Architecture

GoogLeNet 模型的详细结构如下表所示：

![Model Architecture](./assets/GoogLeNet-Architecture.png)

所有的卷积层后均跟着一个 ReLU 激活函数。#3x3 reduce 和 #5x5 reduce 表示 Inception 模块中 3×3 和 5×5 卷积之前使用的**降维层**中 1×1 卷积核的数量，pool proj 表示 3x3 最大池化层后的 1x1 卷积的卷积核数量。

完整的 GoogLeNet 模型结构如下图所示，具有两个**辅助分类器**（auxiliary classifer），分别连接到网络中间层输出的特征图，作用是**增加传播回来的梯度信号**，避免模型训练时执行反向传播时，浅层的梯度较小，导致浅层无法学习，训练时两个辅助分类器的**损失权重为 0.3**，**在测试时不考虑这两个辅助分类器的结果**。

辅助分类器的结构为：

- 步长为 3 的 5x5 **平均池化层**。
- 128 个卷积核的 **1x1 卷积**，降低通道维度，后面跟一个 ReLU 激活函数。
- 具有 1024 单元的**全连接层**，后跟一个 ReLU 激活函数。
- **Dropout 层**，dropout 比例为 70%
- 具有 `num_classes` 个单元的**全连接层**。

![Model Architecture](./assets/GoogLeNet-Architecture-picture.png)



所有的分类器（包括两个辅助分类器）都使用**交叉熵函数**（Cross Entropy）作为损失函数。



## Implementation

Inception 模块的 PyTorch 实现代码如下，这里参考了[《动手学深度学习》GoogLeNet 部分](https://zh.d2l.ai/chapter_convolutional-modern/googlenet.html) 的 Inception 模块实现。

```python
class Inception(nn.Module):
    """Inception Module in GoogLeNet."""
    def __init__(self, in_channels: int, c1: int, c2: tuple[int, int], c3: tuple[int, int], c4: int) -> None:
        """Initialize an Inception Module."""
        super(Inception, self).__init__()
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1, stride=1)
        
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1, stride=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, stride=1, padding=1)
        
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1, stride=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, stride=1, padding=2)
        
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1, stride=1)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass in Inception Module."""
        out1 = F.relu(self.p1_1(x))
        out2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        out3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        out4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat([out4, out3, out2, out1], dim=1)
```

基于 Inception 模块，GoogLeNet 的 PyTorch 实现代码如下：

```python

class GoogLeNet(nn.Module):
    """GoogLeNet."""
    def __init__(self, num_channels: int, num_classes: int) -> None:
        """Initialize GoogLeNet.
        
        Args:
            num_channels(int): number of channels of input images.
            num_classes(int): number of classes.
        """
        super(GoogLeNet, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inception3_1 = Inception(192, 64, (96, 128), (16, 32), 32)
        self.inception3_2 = Inception(256, 128, (128, 192), (32, 96), 64)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inception4_1 = Inception(480, 192, (96, 208), (16, 48), 64)
        self.inception4_2 = Inception(512, 160, (112, 224), (24, 64), 64)
        self.inception4_3 = Inception(512, 128, (128, 256), (24, 64), 64)
        self.inception4_4 = Inception(512, 112, (144, 288), (32, 64), 64)
        self.inception4_5 = Inception(528, 256, (160, 320), (32, 128), 128)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inception5_1 = Inception(832, 256, (160, 320), (32, 128), 128)
        self.inception5_2 = Inception(832, 384, (192, 384), (48, 128), 128)
        self.pool5 = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Linear(1024, num_classes)
    
    def forward(self, images: Tensor):
        """Forward pass in GoogLeNet.
        
        Args:
            images(Tensor): input images of shape (N, C, H, W)
        """
        out = self.conv1(images)
        out = self.relu1(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool2(out)

        out = self.inception3_1(out)
        out = self.inception3_2(out)
        out = self.pool3(out)

        out = self.inception4_1(out)
        out = self.inception4_2(out)
        out = self.inception4_3(out)
        out = self.inception4_4(out)
        out = self.inception4_5(out)
        out = self.pool4(out)

        out = self.inception5_1(out)
        out = self.inception5_2(out)
        out = self.pool5(out)
        
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
```





