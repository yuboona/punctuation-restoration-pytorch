# pytorch搭建简单LSTM系统

## 1. 主要模块

DNN系统的简单搭建需要依赖深度学习框架进行，pytorch是一个非常好的选择，使用的逻辑比较简单易懂。

通常DNN系统包括两个大部分：

- 训练模块
- 测试（预测）模块

## 2. 训练模块搭建

一个DNN模型的训练模块的构建应当包括几个部分：

1. 数据预处理模块
2. 数据输入模块
3. 网络模型编写
4. train代码
5. test代码

其中网络模型和train的代码的构建是较为模式化的，不需要投入过量的精力。

我们需要对数据的处理和输入加以重视，数据于模型如同燃料于汽车，这一点在自然语言处理中尤甚。

我们做如下假设：

- 数据预处理中，数据被篡改
- 数据无法被处理成我们需要的格式

以上两种假设，一个导致模型脱离现实，另一个则使得训练根本无法开始，如同无米之炊。

即使你找到了开源的规范化数据集，你也会发现，数据集仍旧需要进行处理再进入你自己的DNN系统

### 2.1 数据预处理模块

预处理：

### 2.2 数据输入模块

数据再输入网络时，需要被组织成相对于pytorch来说规范的数据格式，pytorch提供了Dataset模块对于数据进行打包、提供了Dataloader对打包好的数据进行加载使用。

我们事实上对LSTM模型的输入会有如下需求：

- **确定序列长度**:LSTM的输入是一个词序列，我们需要确定一次序列的长度
- **确定batch划分**:DNN网络的输入通常是一个batch进行训练，输入的shape(batch_size, 序列长度)，最后网络输出的是batch_size个结果。一般DNN的数据集batch随即划分即可，但对于处理序列数据的LSTM，有各个batch间保持数据连续性的连续采样需求，这样训练时可以在各个batch间传递更多连续的信息。

---

通过对于Dataset,Dataloader进行定制，我们可以满足对数据的需求：

1. **自定义Dataset类的数据初始**要确定好输入数据的句子序列长度，将其分成由句子单元组成的数据集合（通过我的改造，dataset基本单元变为了100的序列，这是使用shuffle与否会产生两种random，一种强[完全随机]一种弱[由取batch导致的不连续性]）
    - 更改__getitem__()函数，pytorch调用类中的这一函数获取某个index的数据
2. **自定义Sampler对数据抽样**然后根据batch\_size确定如何通过定制采样函数，完成连续抽样和随机抽样。（参考repo：pytorch\_punctuation的代码，它对于batch的采样，几乎等于随机采样。因为当batch大于1时，将两个连续的句子作为一个batch，各batch间数据断开$batch\_size-1$的距离，训练时失去了相邻两句话之间的信息传递！！！！！）
   - batchSampler和其他Sampler类都继承的时sampler。更改__iter__()函数，将返回用于确定从dataset中getitem()的*index*列表迭代器。index被用来调用Dataset的__getitem__()获取对应数据。

## 4. train部分

1. 选择损失函数Loss
2. 选择梯度计算优化器，Adam
3. 选择check-point路径
4. 记录最佳损失Loss

定制train代码，选择mini-batch反向传递策略，还有epoch反向传递更新一次的策略？？？

$$
\text{loss}(x, class) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
                       = -x[class] + \log\left(\sum_j \exp(x[j])\right)
$$

## 画图

- viznet
- visio
