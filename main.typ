#import "template.typ": template

#show: template.with(
  title: "实验2 文本分类实验",
  course: "人工智能实验",
)

== 一、实验目的
1. 理解循环神经网络(RNN)的核心原理,掌握 RNN 处理序列文本数据的逻辑;
2. 掌握长短期记忆网络(LSTM)的门控机制(输入门、遗忘门、输出门),理解 LSTM 对长文本依赖的解决思路;
3. 实现 RNN、基础 LSTM 及改进型 LSTM(如 BiLSTM、LSTM+Attention)的文本分类模型;
4. 对比不同模型在同一文本数据集上的分类效果,分析改进模型的优势,培养模型优化思维;
5. 掌握文本分类任务的完整流程:数据预处理→模型构建→训练调优→效果评估→结果分析。


== 二、实验要求
1. 完成 RNN、LSTM、BiLSTM、LSTM+Attention 四种模型的实现,保证代码可运行;
2. 记录不同模型的训练过程(损失曲线、准确率曲线)及评估指标(准确率、精确率、召回率、F1 值);
3. 分析实验结果,明确不同模型的优缺点及适用场景,撰写规范的实验报告;

== 三、实验环境与数据集
1. 辅助库:Numpy、Pandas、Scikit-learn、NLTK
2. 数据集: IMDB 电影评论数据集(二分类,情感分析,英文,每条评论≤500 词);

== 四、实验任务与步骤
1. 数据预处理:完成文本清洗、分词、去停用词、向量化;
2. 模型训练与调优:设置合理的超参数(学习率、批次大小、迭代次数、隐藏层维度等),监控训练过程;
3. 模型评估:在测试集上计算各模型的评估指标,绘制对比图表;
4. 结果分析:对比不同模型的效果,分析改进模型的优势及原因。

== 五、实验考核与报告要求

=== 1. RNN 与 LSTM 的效果差异及 LSTM 解决的核心问题

==== 效果差异

从实验结果与训练曲线可以看出,基础 RNN 的损失下降速度较慢,且训练过程中波动较大,最终在测试集上的准确率、召回率和 F1 值通常都是四种模型中最低的。相比之下,LSTM 的收敛速度更快,损失曲线更加平滑,在验证集和测试集上的各项指标均明显优于基础 RNN。这说明 LSTM 在文本分类任务中具有更强的序列建模能力。

==== 核心问题解释

文本分类任务通常依赖较长的上下文信息,例如 IMDB 数据集中评论序列最长可达到 500 个词。基础 RNN 在反向传播过程中会经历长时间步的梯度连乘,这容易导致严重的梯度消失问题,使模型难以保留远距离词元的信息,从而无法有效捕捉长程依赖关系。

LSTM 通过引入 *细胞状态* 和 *门控机制*,包括输入门、遗忘门和输出门,使信息能够在序列中以更稳定的方式传播。特别是其通过加法路径缓解了传统 RNN 中梯度连乘带来的问题,使得关键信息能够跨越较长的序列传递。因此,LSTM 能够更有效地建模长文本中的上下文依赖关系,从而提升分类性能。

==== BiLSTM 相比基础 LSTM 的优势分析

基础 LSTM 按照从左到右的顺序单向处理文本,因此当前时刻的隐藏状态仅包含该词之前的上下文信息。然而在自然语言处理中,一个词语的真实语义往往不仅依赖前文,还与后文密切相关,因此单向 LSTM 在语义表达上存在一定局限。

BiLSTM 由两个平行的 LSTM 组成:一个从左到右处理序列,另一个从右到左处理序列。最终,模型将两个方向上的隐藏状态进行融合,从而同时利用句子的前向信息和后向信息。这样可以获得更全面的上下文表示,使模型对语义反转、情感转折以及复杂语境的识别更加敏感。

在文本分类任务中,BiLSTM 通常能够带来比单向 LSTM 更丰富的特征表达,因此在准确率和召回率等指标上往往会进一步提升,表现出更强的分类能力。

==== Attention 机制对分类效果提升的作用分析

在基础 LSTM 架构中,通常使用最后一个时间步的隐藏状态作为整个句子的语义表示,再将其输入全连接层进行分类。这种方法本质上是将整条长度可达 500 的文本序列压缩为一个固定维度向量,容易造成关键信息丢失,尤其是在长文本场景中更加明显。

Attention 机制突破了这一限制。它会对 LSTM 在每一个时间步输出的隐藏状态分配一个权重,即注意力分数,使模型能够自动学习哪些词语对于最终分类结果更重要。随后,模型对所有时间步的隐藏状态按照权重进行加权求和,得到更加有效的文本表示。

这种机制的优势主要体现在以下两个方面:

- *提高分类性能*:模型不再只依赖最后一个时间步的信息,而是能够综合利用整个序列中最关键的部分,因此在准确率、召回率和 F1 值上通常优于普通 LSTM。
- *增强可解释性*:通过观察注意力权重,可以分析模型在分类时重点关注了哪些词语,从而为情感分类结果提供一定的解释依据。

因此,Attention 机制在长文本情感分类任务中具有明显优势,能够有效提升模型对关键信息的捕捉能力。

=== 实验中遇到的问题及解决方法

==== 过拟合问题

在训练后期,复杂模型可能会出现训练损失持续下降,而验证损失开始回升的现象,这表明模型对训练数据学习过度,泛化能力下降,即发生了过拟合。

针对这一问题,可以采取以下方法进行缓解:

- 在网络结构中加入 `Dropout` 层,随机屏蔽部分神经元连接;
- 引入 *Early Stopping* 机制,在验证集性能不再提升时提前停止训练;
- 在优化器中设置 `weight_decay`,加入 L2 正则化约束模型参数规模。

这些方法能够有效抑制模型复杂度,提高模型在测试集上的泛化性能。

==== 训练速度慢与内存溢出问题

由于 IMDB 数据集中的文本序列较长,而 LSTM 类模型在处理长序列时计算开销较大,因此实验中容易出现训练速度过慢,甚至显存或内存溢出的情况。

为了解决这一问题,实验中采取了以下措施:

- 将最大序列长度截断为 500,即设置 `max_len = 500`,减少单条样本的计算量;
- 过滤低频词,例如仅保留出现次数不少于 5 次的词语,以减小词表规模;
- 使用 `DataLoader` 进行批量加载和训练,提高数据处理效率;
- 在硬件条件允许的情况下,使用 GPU 进行加速计算。

这些优化手段在保证模型性能的同时,有效提高了训练效率并降低了资源占用。

==== 梯度消失问题

对于基础 RNN 来说,由于其循环结构在时间维度上不断重复,梯度在反向传播时会随着时间步增加而迅速衰减,导致训练初期参数更新非常缓慢,甚至出现模型几乎无法学习有效特征的情况。

对此,可以采用以下解决方法:

- 直接使用 LSTM 或 BiLSTM 替代基础 RNN,从结构上缓解梯度消失问题;
- 在部分场景下,可尝试使用 `ReLU` 激活函数替代传统的 `Tanh`,以减轻梯度衰减现象;
- 结合梯度裁剪等技术,进一步稳定模型训练过程。

总体而言,LSTM 系列模型在处理长序列文本时明显优于基础 RNN,因此在实际实验中通常更具应用价值。

==== 小结

综合实验结果可知,基础 RNN 在长文本情感分类任务中的表现相对较弱,主要受限于梯度消失和长程依赖建模能力不足。LSTM 通过门控机制有效缓解了这一问题,显著提升了模型性能。BiLSTM 在此基础上进一步融合了双向上下文信息,使文本表示更加充分。加入 Attention 机制后,模型能够聚焦于更重要的词语,从而进一步提升分类效果并增强可解释性。

因此,从整体表现来看,带有 Attention 机制的 LSTM 模型在本次实验中取得了最优结果,说明其更适合用于长文本情感分类任务。
#image("assets/image.png")

=== 获取分词数据集
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())
```

=== 获取 IMDB 数据集
```python
from datasets import load_dataset, concatenate_datasets
import numpy as np
from collections import Counter

dataset = load_dataset("stanfordnlp/imdb")
full_dataset = concatenate_datasets([dataset["train"], dataset["test"]]).shuffle(seed=42)

neg_ds = full_dataset.filter(lambda x: x["label"] == 0)
pos_ds = full_dataset.filter(lambda x: x["label"] == 1)

sample_per_class = 3000
sampled_dataset = concatenate_datasets([
    neg_ds.select(range(sample_per_class)),
    pos_ds.select(range(sample_per_class))
]).shuffle(seed=42)


split_1 = sampled_dataset.train_test_split(test_size=0.2, seed=42)
temp_dataset = split_1["train"]
test_dataset = split_1["test"]

split_2 = temp_dataset.train_test_split(test_size=0.125, seed=42)  # 0.125 * 0.8 = 0.1
train_dataset = split_2["train"]
val_dataset = split_2["test"]


X_train = train_dataset["text"]
y_train = train_dataset["label"]

X_val = val_dataset["text"]
y_val = val_dataset["label"]

X_test = test_dataset["text"]
y_test = test_dataset["label"]

print("训练集标签分布:", np.bincount(y_train))
print("验证集标签分布:", np.bincount(y_val))
print("测试集标签分布:", np.bincount(y_test))
stop_words = set(stopwords.words('english'))
vocab = {'<pad>': 0, '<unk>': 1}
word_count = Counter()

X_train_tokens = []
for text in X_train:
    tokens = [w.lower() for w in word_tokenize(text) if w.lower() not in stop_words and w.isalpha()]
    tokens = tokens[:500]
    X_train_tokens.append(tokens)
    word_count.update(tokens)

idx = 2
for word, count in word_count.items():
    if count >= 2:
        vocab[word] = idx
        idx += 1

X_train_ids, X_val_ids, X_test_ids = [], [], []

for tokens in X_train_tokens:
    X_train_ids.append([vocab.get(w, 1) for w in tokens])

for text in X_val:
    tokens = [w.lower() for w in word_tokenize(text) if w.lower() not in stop_words and w.isalpha()][:500]
    X_val_ids.append([vocab.get(w, 1) for w in tokens])

for text in X_test:
    tokens = [w.lower() for w in word_tokenize(text) if w.lower() not in stop_words and w.isalpha()][:500]
    X_test_ids.append([vocab.get(w, 1) for w in tokens])

max_len = 500
for i in range(len(X_train_ids)):
    X_train_ids[i] = X_train_ids[i] + [0] * (max_len - len(X_train_ids[i]))
for i in range(len(X_val_ids)):
    X_val_ids[i] = X_val_ids[i] + [0] * (max_len - len(X_val_ids[i]))
for i in range(len(X_test_ids)):
    X_test_ids[i] = X_test_ids[i] + [0] * (max_len - len(X_test_ids[i]))

train_data = TensorDataset(torch.tensor(X_train_ids), torch.tensor(y_train, dtype=torch.float32))
val_data = TensorDataset(torch.tensor(X_val_ids), torch.tensor(y_val, dtype=torch.float32))
test_data = TensorDataset(torch.tensor(X_test_ids), torch.tensor(y_test, dtype=torch.float32))

batch_size = 128
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)
```
=== 创建模型
```python
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text):
        embedded = self.embedding(text)
        out, hidden = self.rnn(embedded)
        return self.sigmoid(self.fc(hidden.squeeze(0))).squeeze(1)

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text):
        embedded = self.embedding(text)
        out, (hidden, cell) = self.lstm(embedded)
        return self.sigmoid(self.fc(hidden.squeeze(0))).squeeze(1)

class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text):
        embedded = self.embedding(text)
        out, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.sigmoid(self.fc(hidden)).squeeze(1)

class LSTMAttentionModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text):
        embedded = self.embedding(text)
        lstm_out, _ = self.lstm(embedded)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return self.sigmoid(self.fc(context)).squeeze(1)vocab_size = len(vocab)
embed_dim = 100
hidden_dim = 128
epochs = 5
criterion = nn.BCELoss()
models = {
    'RNN': RNNModel(vocab_size, embed_dim, hidden_dim).to(device),
    'LSTM': LSTMModel(vocab_size, embed_dim, hidden_dim).to(device),
    'BiLSTM': BiLSTMModel(vocab_size, embed_dim, hidden_dim).to(device),
    'LSTM_Attention': LSTMAttentionModel(vocab_size, embed_dim, hidden_dim).to(device)
}

history = {name: {'train_loss': [], 'val_loss': [], 'val_acc': []} for name in models.keys()}

for name, model in models.items():
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            predictions = model(texts)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        history[name]['train_loss'].append(avg_train_loss)

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for texts, labels in val_loader:
                texts, labels = texts.to(device), labels.to(device)
                predictions = model(texts)
                loss = criterion(predictions, labels)
                val_loss += loss.item()

                preds = torch.round(predictions)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        history[name]['val_loss'].append(avg_val_loss)
        history[name]['val_acc'].append(val_acc)--- RNN ---
Accuracy: 0.6400
Precision: 0.6560
Recall: 0.6063
F1: 0.6301
--- LSTM ---
Accuracy: 0.7650
Precision: 0.7466
Recall: 0.8105
F1: 0.7773
--- BiLSTM ---
Accuracy: 0.7550
Precision: 0.7504
Recall: 0.7727
F1: 0.7614
--- LSTM_Attention ---
Accuracy: 0.7808
Precision: 0.7500
Recall: 0.8501
F1: 0.7969results = {}

for name, model in models.items():
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            predictions = model(texts)
            preds = torch.round(predictions)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1}

for name, metrics in results.items():
    print(f"--- {name} ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
for name in models.keys():
    plt.plot(history[name]['train_loss'], label=name)
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 2)
for name in models.keys():
    plt.plot(history[name]['val_loss'], label=name)
plt.title('Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 3)
for name in models.keys():
    plt.plot(history[name]['val_acc'], label=name)
plt.title('Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
```
