# HRNet

Link: [Deep High-Resolution Representation Learning for Visual Recognition](http://arxiv.org/abs/1908.07919).

> **High-resolution representations are essential for position-sensitive vision problems**, such as human pose estimation, semantic segmentation, and object detection. Existing state-of-the-art frameworks first encode the input image as a low-resolution representation through a subnetwork that is formed by connecting high-to-low resolution convolutions in series (e.g., ResNet, VGGNet), and then recover the high-resolution representation from the encoded low-resolution representation. Instead, our proposed network, named as **High-Resolution Network** (HRNet), maintains high-resolution representations through the whole process. There are two key characteristics: (i) Connect the high-to-low resolution convolution streams in **parallel**; (ii) Repeatedly exchange the information across resolutions. The benefit is that the resulting representation is semantically richer and spatially more precise. We show the superiority of the proposed HRNet in a wide range of applications, including human pose estimation, semantic segmentation, and object detection, suggesting that the HRNet is a stronger backbone for computer vision problems. All the codes are available at https://github.com/HRNet.



## Background

深度卷积神经网络（DCNNs）在许多计算机视觉任务中达到了最先进的性能，例如图像分类、目标检测、语义分割。

