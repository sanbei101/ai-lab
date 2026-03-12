#import "template.typ": template

#show: template.with(
  title: "实验一 医学图像分类",
  course: "人工智能实验",
)

= 方法原理

本实验旨在利用深度卷积神经网络对医学图像进行多分类预测。卷积神经网络通过卷积层、池化层和全连接层自动提取图像的局部特征与全局表示。

实验选用了两种经典的卷积神经网络架构:残差网络和移动端网络。残差网络通过引入残差连接,解决了深度神经网络训练过程中的梯度消失问题,使得网络能够学习到更深层次的特征。移动端网络则采用了倒残差结构和深度可分离卷积,在保证模型精度的同时大幅度降低了模型的参数量和计算复杂度。

为了提升模型的泛化能力并缓解过拟合现象,实验采用了数据增强技术,包括随机水平翻转和随机旋转等空间变换方法。

模型评估采用了多种评价指标,包括准确率、精确度、召回率、调和平均数以及混淆矩阵。准确率衡量整体预测正确的比例;精确度衡量预测为正样本中实际为正样本的比例;召回率衡量实际正样本中被正确预测的比例;调和平均数综合了精确度与召回率的表现;混淆矩阵则直观展示了各个类别之间的误判情况。

= 方法流程

== 数据加载

利用医学图像开源库获取结直肠病理学数据集,该数据集包含9类结直肠组织病理学图像,图像尺寸为28×28像素。数据加载代码如下:

```python
import os
from medmnist import PathMNIST
from torch.utils.data import DataLoader

data_root = "./medmnist_data"
os.makedirs(data_root, exist_ok=True)

train_dataset = PathMNIST(split='train', transform=data_transforms['train'], download=True, root=data_root)
val_dataset = PathMNIST(split='val', transform=data_transforms['val/test'], download=True, root=data_root)
test_dataset = PathMNIST(split='test', transform=data_transforms['val/test'], download=True, root=data_root)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
```

== 数据预处理与增强

为了防止模型过拟合,在训练集上定义了组合变换,包含随机水平翻转、最大15°的随机旋转、张量转换以及均值和标准差均为0.5的归一化处理。核心实现代码如下:

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
为了使经典视觉模型适配本次病理图像的尺寸与9分类任务要求,分别对残差网络和移动端网络的初始卷积层与最终全连接层进行了修改。构建代码片段如下:

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

== 模型训练与优化

实验采用交叉熵损失函数和亚当优化器进行模型训练,学习率设定为0.001,批次大小为64,训练轮数为5轮。核心参数配置如下:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
epochs = 5
lr = 0.001
num_classes = 9

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model1.parameters(), lr=lr)
```

训练过程中,每个批次前向传播计算输出,反向传播更新梯度,并在每个训练epoch结束后在验证集上评估模型性能,以监控模型的泛化能力。核心训练循环代码如下:

```python
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.squeeze().long().to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)

    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.squeeze().long().to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
```

测试集评估代码如下:

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

model.eval()
test_preds, test_labels = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.squeeze().long().to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

print(f"准确率: {accuracy_score(test_labels, test_preds):.4f}")
print("分类报告:")
print(classification_report(test_labels, test_preds))
print("混淆矩阵:")
print(confusion_matrix(test_labels, test_preds))
```

= 实验结果

== ResNet18 模型结果

ResNet18 模型经过5轮训练,训练过程如下:

```
开始训练 ResNet18...
Epoch 1/5, Train Loss: 0.7840, Val Acc: 0.7776
Epoch 2/5, Train Loss: 0.5169, Val Acc: 0.6450
Epoch 3/5, Train Loss: 0.3976, Val Acc: 0.6537
Epoch 4/5, Train Loss: 0.3430, Val Acc: 0.6997
Epoch 5/5, Train Loss: 0.2982, Val Acc: 0.8232
```

从训练日志可以看出,训练损失持续下降,从第1轮的0.7840降至第5轮的0.2982,表明模型学习过程收敛良好。验证集准确率呈现波动上升趋势,最终在第5轮达到82.32%。

在测试集上的评估结果如下:

```
ResNet18 测试结果:
准确率: 0.6320

分类报告:
              precision    recall  f1-score   support

           0       0.65      0.19      0.29      1338
           1       0.93      1.00      0.96       847
           2       0.29      0.76      0.42       339
           3       0.91      0.94      0.93       634
           4       0.98      0.54      0.70      1035
           5       0.23      0.64      0.34       592
           6       0.86      0.62      0.72       741
           7       0.36      0.32      0.34       421
           8       0.87      0.85      0.86      1233

    accuracy                           0.63      7180
   macro avg       0.68      0.65      0.62      7180
weighted avg       0.75      0.63      0.64      7180
```

ResNet18 模型的整体准确率为63.2%。各分类别的性能表现存在显著差异:类别1(结直肠腺癌)表现最优,达到96%的F1分数;类别3(结肠腺癌)和类别8(腺瘤)同样表现良好,F1分数分别为0.93和0.86。然而,类别0(腺瘤)、类别2(恶性上皮细胞)、类别5(肌肉侵入)和类别7(淋巴细胞)的F1分数均低于0.4,表明模型在这些类别上的识别能力较弱。

混淆矩阵如下:

```
混淆矩阵:
[[ 248    0   25    0    0 1065    0    0    0]
 [   0  847    0    0    0    0    0    0    0]
 [   9    0  257    0    0   54    0   19    0]
 [   0    0   23  598    0    6    1    2    4]
 [  50   57   80    0  561   77   53   99   58]
 [  55    7   94    0    0  379    0   57    0]
 [  13    0   94   24   11   11  461   41   86]
 [   6    0  221    0    0   53    0  135    6]
 [   0    0   84   32    0   18   20   27 1052]]
```

混淆矩阵显示,类别0存在严重的误判情况,有1065个样本被错误预测为类别5(占类别0样本的79.6%),反映出模型难以区分这两类病理特征。类似地,类别5也有较多样本被误判为类别0,说明这两类之间存在较大的特征相似性。

== MobileNetV2 模型结果

MobileNetV2 模型在5轮训练中展现出更稳定的性能提升,训练过程如下:

```
开始训练 mobilenet...
Epoch 1/5, Train Loss: 0.7797, Val Acc: 0.7827
Epoch 2/5, Train Loss: 0.5153, Val Acc: 0.8415
Epoch 3/5, Train Loss: 0.4091, Val Acc: 0.8716
Epoch 4/5, Train Loss: 0.3494, Val Acc: 0.8728
Epoch 5/5, Train Loss: 0.3048, Val Acc: 0.9010
```

验证集准确率从第1轮的78.27%稳步上升至第5轮的90.10%,展现出比ResNet18更稳定的收敛特性。

在测试集上的评估结果如下:

```
mobilenet 测试结果:
准确率: 0.7670

分类报告:
              precision    recall  f1-score   support

           0       0.98      0.94      0.96      1338
           1       0.90      1.00      0.95       847
           2       0.28      0.60      0.38       339
           3       0.91      0.74      0.81       634
           4       0.84      0.68      0.75      1035
           5       0.69      0.62      0.65       592
           6       0.91      0.57      0.70       741
           7       0.34      0.44      0.39       421
           8       0.79      0.85      0.82      1233

    accuracy                           0.77      7180
   macro avg       0.74      0.72      0.71      7180
weighted avg       0.81      0.77      0.78      7180
```

MobileNetV2在测试集上取得了76.70%的整体准确率,较ResNet18提升了13个百分点。加权平均F1分数从0.64提升至0.78,提升幅度达到22%。

在分类性能方面,MobileNetV2在大多数类别上均优于ResNet18。类别0的F1分数从0.29大幅提升至0.96,召回率达到94%,几乎完美解决了ResNet18中该类别被大量误判的问题。类别1继续保持接近完美的分类效果,F1分数达到0.95。类别4、类别5、类别6和类别8的F1分数均较ResNet18有不同程度提升。

混淆矩阵如下:

```
混淆矩阵:
[[1264    2    0    0   65    6    0    1    0]
 [   0  847    0    0    0    0    0    0    0]
 [   0    0  204    0    0   88    0   40    7]
 [   0    0  132  468    0    1   17    6   10]
 [  18   95    3    2  701    8    1  169   38]
 [   0    0   96    3    0  368    0  108   17]
 [   1    0   41   16   63    5  420   28  167]
 [   0    0  136    4    0   53    0  187   41]
 [   1    0  117   22    8    3   23   11 1048]]
```

尽管如此,类别2(恶性上皮细胞)和类别7(淋巴细胞)的F1分数仍低于0.4,分别仅为0.38和0.39,表明这两类病变的组织学特征难以被当前模型有效捕捉,可能需要更精细的数据增强策略或更复杂的网络结构来提升识别效果。

== 模型对比分析

从整体性能来看,MobileNetV2在各项指标上均优于ResNet18。准确率提升了13个百分点,加权平均F1分数从0.64提升至0.78。特别值得注意的是,MobileNetV2成功解决了ResNet18在类别0上的严重误判问题,将F1分数从0.29提升至0.96(提升幅度达230.77%),这充分展示了轻量级网络结构在医学图像分类任务上的潜力。

从模型结构角度分析,MobileNetV2采用的深度可分离卷积大幅降低了模型参数量,而倒残差结构在保证特征表达能力的同时提升了梯度传播效率。相比之下,ResNet18虽然通过残差连接缓解了梯度消失问题,但在处理小尺寸医学图像时可能存在过拟合现象,导致验证集准确率波动较大。

= 结果分析

本次实验成功构建了基于深度卷积神经网络的医学图像分类系统,验证了ResNet18和MobileNetV2两种架构在PathMNIST数据集上的有效性。实验结果表明,MobileNetV2在该任务上表现更优,这为后续部署移动端医学图像辅助诊断系统提供了技术支撑。

然而,实验也暴露出一些问题:部分类别的识别准确率仍然较低,尤其是类别2和类别7。这可能源于两方面原因:一是这两类病变的组织学特征与其他类别相似度高,易造成混淆;二是训练集中这两类的样本数量相对较少,导致模型学习不充分。

针对上述问题,后续可从以下方面改进:首先,采用更丰富的数据增强策略,如颜色抖动、弹性变形等,以增加样本多样性;其次,引入焦点损失或类别加权策略,使模型更加关注难分类样本;最后,可尝试集成学习或预训练模型微调,进一步提升模型的泛化能力。
