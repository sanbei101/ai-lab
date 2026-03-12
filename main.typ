#import "cover.typ": *
#cover(
  title: "实验一 医学图像分类",
  course: "人工智能实验",
  class: "计算231",
  student-id: "2023308250117",
  student-name: "龚浩然",
)

#set par(
  first-line-indent: (
    all: true,
    amount: 2em,
  ),
)
#set text(font: "Noto Serif SC")
#show image: it => align(center, it)

= 方法原理

本实验旨在利用深度卷积神经网络对医学图像进行多分类预测。卷积神经网络通过卷积层、池化层和全连接层自动提取图像的局部特征与全局表示。

实验选用了两种经典的卷积神经网络架构：残差网络和移动端网络。残差网络通过引入残差连接，解决了深度神经网络训练过程中的梯度消失问题，使得网络能够学习到更深层次的特征。移动端网络则采用了倒残差结构和深度可分离卷积，在保证模型精度的同时大幅度降低了模型的参数量和计算复杂度。

为了提升模型的泛化能力并缓解过拟合现象，实验采用了数据增强技术，包括随机水平翻转和随机旋转等空间变换方法。

模型评估采用了多种评价指标，包括准确率、精确度、召回率、调和平均数以及混淆矩阵。准确率衡量整体预测正确的比例；精确度衡量预测为正样本中实际为正样本的比例；召回率衡量实际正样本中被正确预测的比例；调和平均数综合了精确度与召回率的表现；混淆矩阵则直观展示了各个类别之间的误判情况。

= 方法流程

== 数据预处理与增强

利用医学图像开源库获取结直肠病理学数据集。为了防止模型过拟合，在训练集上定义了组合变换，包含随机水平翻转、最大十五度的随机旋转、张量转换以及均值和标准差均为零点五的归一化处理。核心实现代码如下：

```python
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ]),
    'val/test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
}
```

== 模型构建与特征适配
为了使经典视觉模型适配本次病理图像的尺寸与九分类任务要求，分别对残差网络和移动端网络的初始卷积层与最终全连接层进行了修改。构建代码片段如下：

```python
from torchvision.models import resnet18
model1 = resnet18(weights=None)
model1.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
model1.fc = nn.Linear(model1.fc.in_features, num_classes)
model1 = model1.to(device)

from torchvision.models import mobilenet_v2
model2 = mobilenet_v2(weights=None)
model2.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
model2.classifier[1] = nn.Linear(model2.classifier[1].in_features, num_classes)
model2 = model2.to(device)
```
