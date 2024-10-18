# Generative Adversarial Networks

Link: [Generative Adversarial Networks](https://papers.nips.cc/paper_files/paper/2014/hash/5ca3e9b122f61f8f06494c97b1afccf3-Abstract.html).

> We propose a new framework for estimating generative models via an adversarial process, in which we **simultaneously train two models**: **a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G**. The training procedure for G is to maximize the probability of D making a mistake. This framework corresponds to a ***minimax two-player game***. In the space of arbitrary functions G and D, a unique solution exists, with G recovering the training data distribution and D equal to 1/2 everywhere. In the case where G and D are defined  by multilayer perceptrons, the entire system can be trained with backpropagation. There is no need for any Markov chains or unrolled approximate inference networks during either training or generation of samples. Experiments demonstrate the potential of the framework through qualitative and quantitative evaluation of the generated samples.





## Adversarial Networks

**生成对抗网络**（Generative Adversarial Networks）包含两个部分：

- **生成器**（Generator），用于**捕获真实数据的分布**，以生成尽可能像真实数据的虚假数据。
- **辨别器**（Discriminator），用于**区分**真实数据和由生成器生成的假数据。

生成对抗网络的结构如下图所示：

![GAN Overview](./assets/GAN-overview.png)

生成对抗网络的训练过程如下图所示：

![Illustration of GAN](./assets/illustration-of-GAN.png)

假设随机噪声 $z$ 服从**一维均匀分布**，真实数据 $x$ 服从**一维正态分布**，黑色点线表示 $x$ 的概率密度函数，绿色实线表示 $z$ 的概率密度函数，蓝色虚线表示辨别器的辨别密度函数，在 GAN 训练的过程中，生成器**逐渐**学习真实的数据分布，因此由生成器生成的虚假数据的概率密度函数逐渐向真实数据靠近，理论上，生成器可以完全学习到真实数据底层的分布，此时，辨别器就无法区分真实数据和虚假数据了。



### Discriminator

辨别器本质上就是一个**分类器**（Classifier），用于学习真实数据和生成器生成的虚假数据之间的**差异**，尽可能区分真实数据和虚假数据。辨别器的输入数据有**两个来源**：真实数据和生成器生成的虚假数据。

![GAN Discriminator](./assets/GAN-discriminator.png)

辨别器的**训练过程**：

1. 从一个随机分布中采样一组随机噪声，并从真实数据中采样一组**真实数据样本**。
2. 利用生成器将采样的随机噪声进行转换，获取**虚假数据样本**。
3. 让辨别器分别对真实数据和虚假数据**进行分类**。
4. 计算辨别器损失，并执行反向传播获得梯度，**更新辨别器的权重**。



### Generator

生成器用于学习真实数据的底层数据分布，以生成可信的数据，**尽可能骗过辨别器**。

![GAN Generator](./assets/GAN-generator.png)

生成器的训练过程如下：

1. 从一个随机分布中采样一组随机噪声。
2. 利用生成器将采样的随机噪声进行转换，获取**虚假数据样本**。
3. 让辨别器对这一组虚假数据进行分类，获取分类结果。
4. 计算辨别器损失，并执行反向传播获得梯度，**更新生成器的权重**。



### Training Algorithm

生成对抗网络的训练算法如下图所示：

![GAN training algorithm](./assets/GAN-training-algorithm.png)



