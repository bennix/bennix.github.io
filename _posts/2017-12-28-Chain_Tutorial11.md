---
ilayout: post
title: Chainer 入门教程（11）基于CIFAR数据集的卷积网络训练
date: 2017-12-28
categories: blog
tags: [Chainer,入门教程（11), 基于CIFAR数据集的卷积网络训练]
descrption: Chainer 入门教程（11）基于CIFAR数据集的卷积网络训练
---

# 基于CIFAR数据集的卷积网络训练

## CIFAR-10, CIFAR-100 数据集介绍

CIFAR-10和CIFAR-100是分类标记的小图像数据集。它在研究界被广泛用于简单的图像分类任务基准。

官方网站：

[CIFAR-10和100站点](https://www.cs.toronto.edu/~kriz/cifar.html)

在Chainer中，CIFAR-10和CIFAR-100数据集可以通过内置函数获得。


```python
from __future__ import print_function
import os
import matplotlib.pyplot as plt
import numpy as np
import chainer
 
```


```python
%matplotlib inline
```


```python
basedir = './images'
```

## CIFAR-10


在`Chainer`中准备`chainer.datasets.get_cifar10`方法来获取CIFAR-10数据集。数据集仅自动从 https://www.cs.toronto.edu 下载，第二次使用缓存。


```python
CIFAR10_LABELS_LIST = [
    'airplane', 
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]
 
train, test = chainer.datasets.get_cifar10()

```

    Downloading from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz...


数据集结构与MNIST数据集完全相同，即`TupleDataset`。`train[i]` 代表第i个数据，有50000个训练数据。测试数据结构相同，有10000个测试数据。


```python
print('len(train), type ', len(train), type(train))
print('len(test), type ', len(test), type(test))

```

    len(train), type  50000 <class 'chainer.datasets.tuple_dataset.TupleDataset'>
    len(test), type  10000 <class 'chainer.datasets.tuple_dataset.TupleDataset'>


`train[i]` 表示第i个数据，type=tuple ($x_i,y_i$), 其中$x_i$是图像数据，$y_i$是标签数据.

`train[i][0]` 表示CIFAR-10图像数据$x_i$,，这是3维阵列（3,32,32），分别表示RGB通道，宽度32px，高度32px。

`train[i][1]` 表示CIFAR-10图像数据（标量）的标签$y_i$，这是标量值，其实际标签可以被LABELS_LIST转换。ted by LABELS_LIST.

我们来看看第0个数据，`train[0]`的细节。



```python
print('train[0]', type(train[0]), len(train[0]))
 
x0, y0 = train[0]
print('train[0][0]', x0.shape, x0)
print('train[0][1]', y0.shape, y0, '->', CIFAR10_LABELS_LIST[y0])
```

    train[0] <class 'tuple'> 2
    train[0][0] (3, 32, 32) [[[ 0.23137257  0.16862746  0.19607845 ...,  0.61960787  0.59607846
        0.58039218]
      [ 0.0627451   0.          0.07058824 ...,  0.48235297  0.4666667
        0.4784314 ]
      [ 0.09803922  0.0627451   0.19215688 ...,  0.46274513  0.47058827
        0.42745101]
      ..., 
      [ 0.81568635  0.78823537  0.77647066 ...,  0.627451    0.21960786
        0.20784315]
      [ 0.70588237  0.67843139  0.72941178 ...,  0.72156864  0.38039219
        0.32549021]
      [ 0.69411767  0.65882355  0.7019608  ...,  0.84705889  0.59215689
        0.48235297]]
    
     [[ 0.24313727  0.18039216  0.18823531 ...,  0.51764709  0.49019611
        0.48627454]
      [ 0.07843138  0.          0.03137255 ...,  0.34509805  0.32549021
        0.34117648]
      [ 0.09411766  0.02745098  0.10588236 ...,  0.32941177  0.32941177
        0.28627452]
      ..., 
      [ 0.66666669  0.60000002  0.63137257 ...,  0.52156866  0.12156864
        0.13333334]
      [ 0.54509807  0.48235297  0.56470591 ...,  0.58039218  0.24313727
        0.20784315]
      [ 0.56470591  0.50588238  0.55686277 ...,  0.72156864  0.46274513
        0.36078432]]
    
     [[ 0.24705884  0.17647059  0.16862746 ...,  0.42352945  0.40000004
        0.4039216 ]
      [ 0.07843138  0.          0.         ...,  0.21568629  0.19607845
        0.22352943]
      [ 0.08235294  0.          0.03137255 ...,  0.19607845  0.19607845
        0.16470589]
      ..., 
      [ 0.37647063  0.13333334  0.10196079 ...,  0.27450982  0.02745098
        0.07843138]
      [ 0.37647063  0.16470589  0.11764707 ...,  0.36862746  0.13333334
        0.13333334]
      [ 0.45490199  0.36862746  0.34117648 ...,  0.54901963  0.32941177
        0.28235295]]]
    train[0][1] () 6 -> frog



```python
def plot_cifar(filepath, data, row, col, scale=3., label_list=None):
    fig_width = data[0][0].shape[1] / 80 * row * scale
    fig_height = data[0][0].shape[2] / 80 * col * scale
    fig, axes = plt.subplots(row, 
                             col, 
                             figsize=(fig_height, fig_width))
    for i in range(row * col):
        # train[i][0] is i-th image data with size 32x32
        image, label_index = data[i]
        image = image.transpose(1, 2, 0)
        r, c = divmod(i, col)
        axes[r][c].imshow(image)  # cmap='gray' is for black and white picture.
        if label_list is None:
            axes[r][c].set_title('label {}'.format(label_index))
        else:
            axes[r][c].set_title('{}: {}'.format(label_index, label_list[label_index]))
        axes[r][c].axis('off')  # do not show axis value
    plt.tight_layout()   # automatic padding between subplots
    plt.savefig(filepath)
```


```python
plot_cifar(os.path.join(basedir, 'cifar10_plot.png'), train, 4, 5, 
           scale=4., label_list=CIFAR10_LABELS_LIST)
```


![png](https://bennix.github.io/imgs/t11/output_12_0.png)



```python
plot_cifar(os.path.join(basedir, 'cifar10_plot_more.png'), train, 10, 10, 
           scale=4., label_list=CIFAR10_LABELS_LIST)
```


![png](https://bennix.github.io/imgs/t11/output_13_0.png)


## CIFAR-100

CIFAR-100与CIFAR-10非常相似。 chainer.datasets.get_cifar100方法在Chainer中准备得到CIFAR-100数据集。


```python
CIFAR100_LABELS_LIST = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]
 
train_cifar100, test_cifar100 = chainer.datasets.get_cifar100()
```

    Downloading from https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz...


数据集结构与MNIST数据集完全相同，即TupleDataset。

`train[i]`代表第i个数据，有50000个训练数据。列车数据总量相同，班级标签数量增加。所以每个类别标签的训练数据少于CIFAR-10数据集。

测试数据结构相同，有10000个测试数据。




```python
print('len(train_cifar100), type ', len(train_cifar100), type(train_cifar100))
print('len(test_cifar100), type ', len(test_cifar100), type(test_cifar100))
 
print('train_cifar100[0]', type(train_cifar100[0]), len(train_cifar100[0]))
 
x0, y0 = train_cifar100[0]
print('train_cifar100[0][0]', x0.shape)  # , x0
print('train_cifar100[0][1]', y0.shape, y0)
```

    len(train_cifar100), type  50000 <class 'chainer.datasets.tuple_dataset.TupleDataset'>
    len(test_cifar100), type  10000 <class 'chainer.datasets.tuple_dataset.TupleDataset'>
    train_cifar100[0] <class 'tuple'> 2
    train_cifar100[0][0] (3, 32, 32)
    train_cifar100[0][1] () 19



```python
plot_cifar(os.path.join(basedir, 'cifar100_plot_more.png'), train_cifar100,
           10, 10, scale=4., label_list=CIFAR100_LABELS_LIST)
```


![png](https://bennix.github.io/imgs/t11/output_18_0.png)


定义一个CNN架构的神经网络


```python
import chainer
import chainer.functions as F
import chainer.links as L

class CNNMedium(chainer.Chain):
    def __init__(self, n_out):
        super(CNNMedium, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 16, 3, 1)
            self.conv2 = L.Convolution2D(16, 32, 3, 2)
            self.conv3 = L.Convolution2D(32, 32, 3, 1)
            self.conv4 = L.Convolution2D(32, 64, 3, 2)
            self.conv5 = L.Convolution2D(64, 64, 3, 1)
            self.conv6 = L.Convolution2D(64, 128, 3, 2)
            self.fc7 = L.Linear(None, 100)
            self.fc8 = L.Linear(100, n_out)
 
    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = F.relu(self.conv6(h))
        h = F.relu(self.fc7(h))
        h = self.fc8(h)
        return h
```

卷积层的计算代价近似为，

$H_I \times W_I  \times  CH_I \times  CH_O \times k^2$

$CH_I$  : 输入图像通道数
$CH_O$ : 输出图像通道数
$H_I$     : 输入图像高度
$W_I$    : 输入图像宽度
$k$     : 内核大小（假设宽度和高度相同）
 

在以上的CNN定义中，对于更深的层来说，通道的尺寸更大。这可以通过计算每层的计算成本来理解。

当采用`stride=2`的`L.Convolution2D`时，图像的大小几乎变成了一半。这意味着$H_I$和$W_I$变得很小，所以$CH_I$和$CH_O$可以选择更大的数值。



## 训练 CIFAR-10

一旦你写了CNN，很容易训练这个模型


```python
from __future__ import print_function
import argparse
 
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, iterators, serializers, optimizers
from chainer.training import extensions

```


```python
archs = {
    'cnnmedium': CNNMedium,
}
```


```python
parser = argparse.ArgumentParser(description='Cifar-10 CNN example')
parser.add_argument('--arch', '-a', choices=archs.keys(),
                    default='cnnmedium', help='Convnet architecture')
parser.add_argument('--batchsize', '-b', type=int, default=64,
                    help='Number of images in each mini-batch')
parser.add_argument('--epoch', '-e', type=int, default=20,
                    help='Number of sweeps over the dataset to train')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--out', '-o', default='result-cifar10',
                    help='Directory to output the result')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the training from snapshot')
args = parser.parse_args(['-g','0'])
```


```python
print('GPU: {}'.format(args.gpu))
print('# Minibatch-size: {}'.format(args.batchsize))
print('# epoch: {}'.format(args.epoch))
print('')
```

    GPU: 0
    # Minibatch-size: 64
    # epoch: 20
    



```python
# 1. Setup model
class_num = 10
model = archs[args.arch](n_out=class_num)
classifier_model = L.Classifier(model)
if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
    classifier_model.to_gpu()  # Copy the model to the GPU
```


```python
# 2. Setup an optimizer
optimizer = optimizers.Adam()
optimizer.setup(classifier_model)
```


```python
# 3. Load the CIFAR-10 dataset
train, test = chainer.datasets.get_cifar10()
```


```python
# 4. Setup an Iterator
train_iter = iterators.SerialIterator(train, args.batchsize)
test_iter = iterators.SerialIterator(test, args.batchsize,
                                     repeat=False, shuffle=False)
```


```python
# 5. Setup an Updater
updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
```


```python
# 6. Setup a trainer (and extensions)
trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
```


```python
# Evaluate the model with the test dataset for each epoch
trainer.extend(extensions.Evaluator(test_iter, classifier_model, device=args.gpu))

trainer.extend(extensions.dump_graph('main/loss'))
trainer.extend(extensions.snapshot(), trigger=(1, 'epoch'))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(
    ['epoch', 'main/loss', 'validation/main/loss',
     'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
trainer.extend(extensions.PlotReport(
    ['main/loss', 'validation/main/loss'],
    x_key='epoch', file_name='loss.png'))
trainer.extend(extensions.PlotReport(
    ['main/accuracy', 'validation/main/accuracy'],
    x_key='epoch',
    file_name='accuracy.png'))

trainer.extend(extensions.ProgressBar())
```


```python
# Resume from a snapshot
if args.resume:
    serializers.load_npz(args.resume, trainer)
```


```python
# Run the training
trainer.run()
```

    epoch       main/loss   validation/main/loss  main/accuracy  validation/main/accuracy  elapsed_time
    [J     total [..................................................]  0.64%
    this epoch [######............................................] 12.80%
           100 iter, 0 epoch / 20 epochs
           inf iters/sec. Estimated time to finish: 0:00:00.
    [4A[J     total [..................................................]  1.28%
    this epoch [############......................................] 25.60%
           200 iter, 0 epoch / 20 epochs
        125.61 iters/sec. Estimated time to finish: 0:02:02.799182.
    [4A[J     total [..................................................]  1.92%
    this epoch [###################...............................] 38.40%
           300 iter, 0 epoch / 20 epochs
        126.41 iters/sec. Estimated time to finish: 0:02:01.228643.
    [4A[J     total [#.................................................]  2.56%
    this epoch [#########################.........................] 51.20%
           400 iter, 0 epoch / 20 epochs
        126.96 iters/sec. Estimated time to finish: 0:01:59.915494.
    [4A[J     total [#.................................................]  3.20%
    this epoch [################################..................] 64.00%
           500 iter, 0 epoch / 20 epochs
        127.31 iters/sec. Estimated time to finish: 0:01:58.799905.
    [4A[J     total [#.................................................]  3.84%
    this epoch [######################################............] 76.80%
           600 iter, 0 epoch / 20 epochs
         127.5 iters/sec. Estimated time to finish: 0:01:57.844706.
    [4A[J     total [##................................................]  4.48%
    this epoch [############################################......] 89.60%
           700 iter, 0 epoch / 20 epochs
        127.62 iters/sec. Estimated time to finish: 0:01:56.947000.
    [4A[J1           1.63966     1.37119               0.391584       0.497811                  7.3451        
    [J     total [##................................................]  5.12%
    this epoch [#.................................................]  2.40%
           800 iter, 1 epoch / 20 epochs
        112.44 iters/sec. Estimated time to finish: 0:02:11.847638.
    [4A[J     total [##................................................]  5.76%
    this epoch [#######...........................................] 15.20%
           900 iter, 1 epoch / 20 epochs
        114.07 iters/sec. Estimated time to finish: 0:02:09.084668.
    [4A[J     total [###...............................................]  6.40%
    this epoch [##############....................................] 28.00%
          1000 iter, 1 epoch / 20 epochs
        115.23 iters/sec. Estimated time to finish: 0:02:06.920558.
    [4A[J     total [###...............................................]  7.04%
    this epoch [####################..............................] 40.80%
          1100 iter, 1 epoch / 20 epochs
         116.3 iters/sec. Estimated time to finish: 0:02:04.891163.
    [4A[J     total [###...............................................]  7.68%
    this epoch [##########################........................] 53.60%
          1200 iter, 1 epoch / 20 epochs
        117.24 iters/sec. Estimated time to finish: 0:02:03.035187.
    [4A[J     total [####..............................................]  8.32%
    this epoch [#################################.................] 66.40%
          1300 iter, 1 epoch / 20 epochs
         118.1 iters/sec. Estimated time to finish: 0:02:01.294570.
    [4A[J     total [####..............................................]  8.96%
    this epoch [#######################################...........] 79.20%
          1400 iter, 1 epoch / 20 epochs
        118.83 iters/sec. Estimated time to finish: 0:01:59.703835.
    [4A[J     total [####..............................................]  9.60%
    this epoch [##############################################....] 92.00%
          1500 iter, 1 epoch / 20 epochs
        119.39 iters/sec. Estimated time to finish: 0:01:58.311100.
    [4A[J2           1.27853     1.19743               0.540013       0.573646                  14.2082       
    [J     total [#####.............................................] 10.24%
    this epoch [##................................................]  4.80%
          1600 iter, 2 epoch / 20 epochs
         113.3 iters/sec. Estimated time to finish: 0:02:03.789080.
    [4A[J     total [#####.............................................] 10.88%
    this epoch [########..........................................] 17.60%
          1700 iter, 2 epoch / 20 epochs
        114.13 iters/sec. Estimated time to finish: 0:02:02.007898.
    [4A[J     total [#####.............................................] 11.52%
    this epoch [###############...................................] 30.40%
          1800 iter, 2 epoch / 20 epochs
        114.76 iters/sec. Estimated time to finish: 0:02:00.466795.
    [4A[J     total [######............................................] 12.16%
    this epoch [#####################.............................] 43.20%
          1900 iter, 2 epoch / 20 epochs
        115.41 iters/sec. Estimated time to finish: 0:01:58.924804.
    [4A[J     total [######............................................] 12.80%
    this epoch [############################......................] 56.00%
          2000 iter, 2 epoch / 20 epochs
        116.07 iters/sec. Estimated time to finish: 0:01:57.383023.
    [4A[J     total [######............................................] 13.44%
    this epoch [##################################................] 68.80%
          2100 iter, 2 epoch / 20 epochs
        116.61 iters/sec. Estimated time to finish: 0:01:55.984400.
    [4A[J     total [#######...........................................] 14.08%
    this epoch [########################################..........] 81.60%
          2200 iter, 2 epoch / 20 epochs
        117.11 iters/sec. Estimated time to finish: 0:01:54.632769.
    [4A[J     total [#######...........................................] 14.72%
    this epoch [###############################################...] 94.40%
          2300 iter, 2 epoch / 20 epochs
        117.56 iters/sec. Estimated time to finish: 0:01:53.342540.
    [4A[J3           1.11089     1.09346               0.604273       0.616143                  21.0442       
    [J     total [#######...........................................] 15.36%
    this epoch [###...............................................]  7.20%
          2400 iter, 3 epoch / 20 epochs
        113.63 iters/sec. Estimated time to finish: 0:01:56.384466.
    [4A[J     total [########..........................................] 16.00%
    this epoch [##########........................................] 20.00%
          2500 iter, 3 epoch / 20 epochs
        114.14 iters/sec. Estimated time to finish: 0:01:54.990394.
    [4A[J     total [########..........................................] 16.64%
    this epoch [################..................................] 32.80%
          2600 iter, 3 epoch / 20 epochs
        114.61 iters/sec. Estimated time to finish: 0:01:53.649835.
    [4A[J     total [########..........................................] 17.28%
    this epoch [######################............................] 45.60%
          2700 iter, 3 epoch / 20 epochs
        115.06 iters/sec. Estimated time to finish: 0:01:52.328166.
    [4A[J     total [########..........................................] 17.92%
    this epoch [#############################.....................] 58.40%
          2800 iter, 3 epoch / 20 epochs
         115.5 iters/sec. Estimated time to finish: 0:01:51.043157.
    [4A[J     total [#########.........................................] 18.56%
    this epoch [###################################...............] 71.20%
          2900 iter, 3 epoch / 20 epochs
        115.91 iters/sec. Estimated time to finish: 0:01:49.781270.
    [4A[J     total [#########.........................................] 19.20%
    this epoch [#########################################.........] 84.00%
          3000 iter, 3 epoch / 20 epochs
        116.28 iters/sec. Estimated time to finish: 0:01:48.571255.
    [4A[J     total [#########.........................................] 19.84%
    this epoch [################################################..] 96.80%
          3100 iter, 3 epoch / 20 epochs
        116.63 iters/sec. Estimated time to finish: 0:01:47.391292.
    [4A[J4           0.993067    1.0149                0.645947       0.640625                  27.909        
    [J     total [##########........................................] 20.48%
    this epoch [####..............................................]  9.60%
          3200 iter, 4 epoch / 20 epochs
        113.76 iters/sec. Estimated time to finish: 0:01:49.218342.
    [4A[J     total [##########........................................] 21.12%
    this epoch [###########.......................................] 22.40%
          3300 iter, 4 epoch / 20 epochs
        114.13 iters/sec. Estimated time to finish: 0:01:47.991799.
    [4A[J     total [##########........................................] 21.76%
    this epoch [#################.................................] 35.20%
          3400 iter, 4 epoch / 20 epochs
        114.48 iters/sec. Estimated time to finish: 0:01:46.784048.
    [4A[J     total [###########.......................................] 22.40%
    this epoch [########################..........................] 48.00%
          3500 iter, 4 epoch / 20 epochs
        114.84 iters/sec. Estimated time to finish: 0:01:45.585070.
    [4A[J     total [###########.......................................] 23.04%
    this epoch [##############################....................] 60.80%
          3600 iter, 4 epoch / 20 epochs
        115.15 iters/sec. Estimated time to finish: 0:01:44.426766.
    [4A[J     total [###########.......................................] 23.68%
    this epoch [####################################..............] 73.60%
          3700 iter, 4 epoch / 20 epochs
        115.47 iters/sec. Estimated time to finish: 0:01:43.271093.
    [4A[J     total [############......................................] 24.32%
    this epoch [###########################################.......] 86.40%
          3800 iter, 4 epoch / 20 epochs
        115.69 iters/sec. Estimated time to finish: 0:01:42.215823.
    [4A[J     total [############......................................] 24.96%
    this epoch [#################################################.] 99.20%
          3900 iter, 4 epoch / 20 epochs
        115.94 iters/sec. Estimated time to finish: 0:01:41.131895.
    [4A[J5           0.906105    0.981679              0.679028       0.65834                   34.8258       
    [J     total [############......................................] 25.60%
    this epoch [######............................................] 12.00%
          4000 iter, 5 epoch / 20 epochs
        113.09 iters/sec. Estimated time to finish: 0:01:42.796864.
    [4A[J     total [#############.....................................] 26.24%
    this epoch [############......................................] 24.80%
          4100 iter, 5 epoch / 20 epochs
        113.24 iters/sec. Estimated time to finish: 0:01:41.777657.
    [4A[J     total [#############.....................................] 26.88%
    this epoch [##################................................] 37.60%
          4200 iter, 5 epoch / 20 epochs
        113.41 iters/sec. Estimated time to finish: 0:01:40.744638.
    [4A[J     total [#############.....................................] 27.52%
    this epoch [#########################.........................] 50.40%
          4300 iter, 5 epoch / 20 epochs
        113.73 iters/sec. Estimated time to finish: 0:01:39.575979.
    [4A[J     total [##############....................................] 28.16%
    this epoch [###############################...................] 63.20%
          4400 iter, 5 epoch / 20 epochs
        114.02 iters/sec. Estimated time to finish: 0:01:38.447348.
    [4A[J     total [##############....................................] 28.80%
    this epoch [#####################################.............] 76.00%
          4500 iter, 5 epoch / 20 epochs
         114.3 iters/sec. Estimated time to finish: 0:01:37.333849.
    [4A[J     total [##############....................................] 29.44%
    this epoch [############################################......] 88.80%
          4600 iter, 5 epoch / 20 epochs
        114.56 iters/sec. Estimated time to finish: 0:01:36.234988.
    [4A[J6           0.829003    0.962217              0.706646       0.663018                  41.9501       
    [J     total [###############...................................] 30.08%
    this epoch [..................................................]  1.60%
          4700 iter, 6 epoch / 20 epochs
        112.76 iters/sec. Estimated time to finish: 0:01:36.888072.
    [4A[J     total [###############...................................] 30.72%
    this epoch [#######...........................................] 14.40%
          4800 iter, 6 epoch / 20 epochs
        113.02 iters/sec. Estimated time to finish: 0:01:35.775456.
    [4A[J     total [###############...................................] 31.36%
    this epoch [#############.....................................] 27.20%
          4900 iter, 6 epoch / 20 epochs
        113.29 iters/sec. Estimated time to finish: 0:01:34.666533.
    [4A[J     total [################..................................] 32.00%
    this epoch [####################..............................] 40.00%
          5000 iter, 6 epoch / 20 epochs
        113.54 iters/sec. Estimated time to finish: 0:01:33.582794.
    [4A[J     total [################..................................] 32.64%
    this epoch [##########################........................] 52.80%
          5100 iter, 6 epoch / 20 epochs
        113.79 iters/sec. Estimated time to finish: 0:01:32.496581.
    [4A[J     total [################..................................] 33.28%
    this epoch [################################..................] 65.60%
          5200 iter, 6 epoch / 20 epochs
        114.06 iters/sec. Estimated time to finish: 0:01:31.402953.
    [4A[J     total [################..................................] 33.92%
    this epoch [#######################################...........] 78.40%
          5300 iter, 6 epoch / 20 epochs
        114.29 iters/sec. Estimated time to finish: 0:01:30.342284.
    [4A[J     total [#################.................................] 34.56%
    this epoch [#############################################.....] 91.20%
          5400 iter, 6 epoch / 20 epochs
        114.52 iters/sec. Estimated time to finish: 0:01:29.285742.
    [4A[J7           0.762821    0.922659              0.729133       0.682922                  48.8031       
    [J     total [#################.................................] 35.20%
    this epoch [##................................................]  4.00%
          5500 iter, 7 epoch / 20 epochs
        112.99 iters/sec. Estimated time to finish: 0:01:29.610861.
    [4A[J     total [#################.................................] 35.84%
    this epoch [########..........................................] 16.80%
          5600 iter, 7 epoch / 20 epochs
        113.22 iters/sec. Estimated time to finish: 0:01:28.540957.
    [4A[J     total [##################................................] 36.48%
    this epoch [##############....................................] 29.60%
          5700 iter, 7 epoch / 20 epochs
        113.42 iters/sec. Estimated time to finish: 0:01:27.504411.
    [4A[J     total [##################................................] 37.12%
    this epoch [#####################.............................] 42.40%
          5800 iter, 7 epoch / 20 epochs
        113.64 iters/sec. Estimated time to finish: 0:01:26.454301.
    [4A[J     total [##################................................] 37.76%
    this epoch [###########################.......................] 55.20%
          5900 iter, 7 epoch / 20 epochs
        113.87 iters/sec. Estimated time to finish: 0:01:25.401434.
    [4A[J     total [###################...............................] 38.40%
    this epoch [#################################.................] 68.00%
          6000 iter, 7 epoch / 20 epochs
        114.07 iters/sec. Estimated time to finish: 0:01:24.380654.
    [4A[J     total [###################...............................] 39.04%
    this epoch [########################################..........] 80.80%
          6100 iter, 7 epoch / 20 epochs
        114.28 iters/sec. Estimated time to finish: 0:01:23.346596.
    [4A[J     total [###################...............................] 39.68%
    this epoch [##############################################....] 93.60%
          6200 iter, 7 epoch / 20 epochs
        114.48 iters/sec. Estimated time to finish: 0:01:22.328793.
    [4A[J8           0.698332    0.957248              0.753661       0.672671                  55.6596       
    [J     total [####################..............................] 40.32%
    this epoch [###...............................................]  6.40%
          6300 iter, 8 epoch / 20 epochs
        113.13 iters/sec. Estimated time to finish: 0:01:22.426493.
    [4A[J     total [####################..............................] 40.96%
    this epoch [#########.........................................] 19.20%
          6400 iter, 8 epoch / 20 epochs
        113.34 iters/sec. Estimated time to finish: 0:01:21.394052.
    [4A[J     total [####################..............................] 41.60%
    this epoch [################..................................] 32.00%
          6500 iter, 8 epoch / 20 epochs
        113.53 iters/sec. Estimated time to finish: 0:01:20.372562.
    [4A[J     total [#####################.............................] 42.24%
    this epoch [######################............................] 44.80%
          6600 iter, 8 epoch / 20 epochs
        113.73 iters/sec. Estimated time to finish: 0:01:19.354827.
    [4A[J     total [#####################.............................] 42.88%
    this epoch [############################......................] 57.60%
          6700 iter, 8 epoch / 20 epochs
        113.93 iters/sec. Estimated time to finish: 0:01:18.340858.
    [4A[J     total [#####################.............................] 43.52%
    this epoch [###################################...............] 70.40%
          6800 iter, 8 epoch / 20 epochs
         114.1 iters/sec. Estimated time to finish: 0:01:17.342524.
    [4A[J     total [######################............................] 44.16%
    this epoch [#########################################.........] 83.20%
          6900 iter, 8 epoch / 20 epochs
        114.28 iters/sec. Estimated time to finish: 0:01:16.346479.
    [4A[J     total [######################............................] 44.80%
    this epoch [################################################..] 96.00%
          7000 iter, 8 epoch / 20 epochs
        114.46 iters/sec. Estimated time to finish: 0:01:15.350704.
    [4A[J9           0.65115     0.937813              0.768942       0.682325                  62.5173       
    [J     total [######################............................] 45.44%
    this epoch [####..............................................]  8.80%
          7100 iter, 9 epoch / 20 epochs
        113.15 iters/sec. Estimated time to finish: 0:01:15.342457.
    [4A[J     total [#######################...........................] 46.08%
    this epoch [##########........................................] 21.60%
          7200 iter, 9 epoch / 20 epochs
        113.33 iters/sec. Estimated time to finish: 0:01:14.342647.
    [4A[J     total [#######################...........................] 46.72%
    this epoch [#################.................................] 34.40%
          7300 iter, 9 epoch / 20 epochs
        113.51 iters/sec. Estimated time to finish: 0:01:13.340419.
    [4A[J     total [#######################...........................] 47.36%
    this epoch [#######################...........................] 47.20%
          7400 iter, 9 epoch / 20 epochs
        113.69 iters/sec. Estimated time to finish: 0:01:12.346160.
    [4A[J     total [########################..........................] 48.00%
    this epoch [#############################.....................] 60.00%
          7500 iter, 9 epoch / 20 epochs
        113.87 iters/sec. Estimated time to finish: 0:01:11.353393.
    [4A[J     total [########################..........................] 48.64%
    this epoch [####################################..............] 72.80%
          7600 iter, 9 epoch / 20 epochs
        114.02 iters/sec. Estimated time to finish: 0:01:10.382495.
    [4A[J     total [########################..........................] 49.28%
    this epoch [##########################################........] 85.60%
          7700 iter, 9 epoch / 20 epochs
        114.19 iters/sec. Estimated time to finish: 0:01:09.404675.
    [4A[J     total [########################..........................] 49.92%
    this epoch [#################################################.] 98.40%
          7800 iter, 9 epoch / 20 epochs
        114.34 iters/sec. Estimated time to finish: 0:01:08.434026.
    [4A[J10          0.601723    0.937862              0.786852       0.688296                  69.4373       
    [J     total [#########################.........................] 50.56%
    this epoch [#####.............................................] 11.20%
          7900 iter, 10 epoch / 20 epochs
        113.26 iters/sec. Estimated time to finish: 0:01:08.208386.
    [4A[J     total [#########################.........................] 51.20%
    this epoch [############......................................] 24.00%
          8000 iter, 10 epoch / 20 epochs
        113.41 iters/sec. Estimated time to finish: 0:01:07.235866.
    [4A[J     total [#########################.........................] 51.84%
    this epoch [##################................................] 36.80%
          8100 iter, 10 epoch / 20 epochs
        113.57 iters/sec. Estimated time to finish: 0:01:06.261301.
    [4A[J     total [##########################........................] 52.48%
    this epoch [########################..........................] 49.60%
          8200 iter, 10 epoch / 20 epochs
        113.72 iters/sec. Estimated time to finish: 0:01:05.289734.
    [4A[J     total [##########################........................] 53.12%
    this epoch [###############################...................] 62.40%
          8300 iter, 10 epoch / 20 epochs
        113.86 iters/sec. Estimated time to finish: 0:01:04.330680.
    [4A[J     total [##########################........................] 53.76%
    this epoch [#####################################.............] 75.20%
          8400 iter, 10 epoch / 20 epochs
        114.01 iters/sec. Estimated time to finish: 0:01:03.374040.
    [4A[J     total [###########################.......................] 54.40%
    this epoch [############################################......] 88.00%
          8500 iter, 10 epoch / 20 epochs
        114.14 iters/sec. Estimated time to finish: 0:01:02.421317.
    [4A[J11          0.544254    0.994541              0.806358       0.699343                  76.3034       
    [J     total [###########################.......................] 55.04%
    this epoch [..................................................]  0.80%
          8600 iter, 11 epoch / 20 epochs
        113.19 iters/sec. Estimated time to finish: 0:01:02.064211.
    [4A[J     total [###########################.......................] 55.68%
    this epoch [######............................................] 13.60%
          8700 iter, 11 epoch / 20 epochs
        113.33 iters/sec. Estimated time to finish: 0:01:01.102122.
    [4A[J     total [############################......................] 56.32%
    this epoch [#############.....................................] 26.40%
          8800 iter, 11 epoch / 20 epochs
        113.48 iters/sec. Estimated time to finish: 0:01:00.143439.
    [4A[J     total [############################......................] 56.96%
    this epoch [###################...............................] 39.20%
          8900 iter, 11 epoch / 20 epochs
        113.62 iters/sec. Estimated time to finish: 0:00:59.190595.
    [4A[J     total [############################......................] 57.60%
    this epoch [#########################.........................] 52.00%
          9000 iter, 11 epoch / 20 epochs
        113.76 iters/sec. Estimated time to finish: 0:00:58.235956.
    [4A[J     total [#############################.....................] 58.24%
    this epoch [################################..................] 64.80%
          9100 iter, 11 epoch / 20 epochs
        113.89 iters/sec. Estimated time to finish: 0:00:57.291026.
    [4A[J     total [#############################.....................] 58.88%
    this epoch [######################################............] 77.60%
          9200 iter, 11 epoch / 20 epochs
        114.03 iters/sec. Estimated time to finish: 0:00:56.345303.
    [4A[J     total [#############################.....................] 59.52%
    this epoch [#############################################.....] 90.40%
          9300 iter, 11 epoch / 20 epochs
        114.17 iters/sec. Estimated time to finish: 0:00:55.401445.
    [4A[J12          0.501473    1.00326               0.821763       0.687898                  83.1545       
    [J     total [##############################....................] 60.16%
    this epoch [#.................................................]  3.20%
          9400 iter, 12 epoch / 20 epochs
        113.28 iters/sec. Estimated time to finish: 0:00:54.952157.
    [4A[J     total [##############################....................] 60.80%
    this epoch [########..........................................] 16.00%
          9500 iter, 12 epoch / 20 epochs
        113.41 iters/sec. Estimated time to finish: 0:00:54.007632.
    [4A[J     total [##############################....................] 61.44%
    this epoch [##############....................................] 28.80%
          9600 iter, 12 epoch / 20 epochs
        113.54 iters/sec. Estimated time to finish: 0:00:53.065726.
    [4A[J     total [###############################...................] 62.08%
    this epoch [####################..............................] 41.60%
          9700 iter, 12 epoch / 20 epochs
        113.66 iters/sec. Estimated time to finish: 0:00:52.126871.
    [4A[J     total [###############################...................] 62.72%
    this epoch [###########################.......................] 54.40%
          9800 iter, 12 epoch / 20 epochs
        113.79 iters/sec. Estimated time to finish: 0:00:51.189483.
    [4A[J     total [###############################...................] 63.36%
    this epoch [#################################.................] 67.20%
          9900 iter, 12 epoch / 20 epochs
        113.91 iters/sec. Estimated time to finish: 0:00:50.256781.
    [4A[J     total [################################..................] 64.00%
    this epoch [########################################..........] 80.00%
         10000 iter, 12 epoch / 20 epochs
        114.04 iters/sec. Estimated time to finish: 0:00:49.326404.
    [4A[J     total [################################..................] 64.64%
    this epoch [##############################################....] 92.80%
         10100 iter, 12 epoch / 20 epochs
        114.15 iters/sec. Estimated time to finish: 0:00:48.401125.
    [4A[J13          0.458805    1.0016                0.835478       0.69586                   90.0334       
    [J     total [################################..................] 65.28%
    this epoch [##................................................]  5.60%
         10200 iter, 13 epoch / 20 epochs
        113.16 iters/sec. Estimated time to finish: 0:00:47.938875.
    [4A[J     total [################################..................] 65.92%
    this epoch [#########.........................................] 18.40%
         10300 iter, 13 epoch / 20 epochs
        113.16 iters/sec. Estimated time to finish: 0:00:47.056687.
    [4A[J     total [#################################.................] 66.56%
    this epoch [###############...................................] 31.20%
         10400 iter, 13 epoch / 20 epochs
        113.15 iters/sec. Estimated time to finish: 0:00:46.178907.
    [4A[J     total [#################################.................] 67.20%
    this epoch [#####################.............................] 44.00%
         10500 iter, 13 epoch / 20 epochs
        113.13 iters/sec. Estimated time to finish: 0:00:45.300128.
    [4A[J     total [#################################.................] 67.84%
    this epoch [############################......................] 56.80%
         10600 iter, 13 epoch / 20 epochs
        113.12 iters/sec. Estimated time to finish: 0:00:44.423722.
    [4A[J     total [##################################................] 68.48%
    this epoch [##################################................] 69.60%
         10700 iter, 13 epoch / 20 epochs
        113.11 iters/sec. Estimated time to finish: 0:00:43.541908.
    [4A[J     total [##################################................] 69.12%
    this epoch [#########################################.........] 82.40%
         10800 iter, 13 epoch / 20 epochs
        114.05 iters/sec. Estimated time to finish: 0:00:42.304912.
    [4A[J     total [##################################................] 69.76%
    this epoch [###############################################...] 95.20%
         10900 iter, 13 epoch / 20 epochs
        114.06 iters/sec. Estimated time to finish: 0:00:41.427012.
    [4A[J14          0.41086     1.0577                0.855374       0.688993                  96.974        
    [J     total [###################################...............] 70.40%
    this epoch [####..............................................]  8.00%
         11000 iter, 14 epoch / 20 epochs
        113.12 iters/sec. Estimated time to finish: 0:00:40.885069.
    [4A[J     total [###################################...............] 71.04%
    this epoch [##########........................................] 20.80%
         11100 iter, 14 epoch / 20 epochs
        113.13 iters/sec. Estimated time to finish: 0:00:39.999143.
    [4A[J     total [###################################...............] 71.68%
    this epoch [################..................................] 33.60%
         11200 iter, 14 epoch / 20 epochs
        113.13 iters/sec. Estimated time to finish: 0:00:39.114540.
    [4A[J     total [####################################..............] 72.32%
    this epoch [#######################...........................] 46.40%
         11300 iter, 14 epoch / 20 epochs
        113.12 iters/sec. Estimated time to finish: 0:00:38.235042.
    [4A[J     total [####################################..............] 72.96%
    this epoch [#############################.....................] 59.20%
         11400 iter, 14 epoch / 20 epochs
        113.12 iters/sec. Estimated time to finish: 0:00:37.351202.
    [4A[J     total [####################################..............] 73.60%
    this epoch [####################################..............] 72.00%
         11500 iter, 14 epoch / 20 epochs
        113.12 iters/sec. Estimated time to finish: 0:00:36.465431.
    [4A[J     total [#####################################.............] 74.24%
    this epoch [##########################################........] 84.80%
         11600 iter, 14 epoch / 20 epochs
        114.07 iters/sec. Estimated time to finish: 0:00:35.286180.
    [4A[J     total [#####################################.............] 74.88%
    this epoch [################################################..] 97.60%
         11700 iter, 14 epoch / 20 epochs
        114.06 iters/sec. Estimated time to finish: 0:00:34.412155.
    [4A[J15          0.379527    1.12376               0.864497       0.684713                  103.831       
    [J     total [#####################################.............] 75.52%
    this epoch [#####.............................................] 10.40%
         11800 iter, 15 epoch / 20 epochs
        113.11 iters/sec. Estimated time to finish: 0:00:33.815536.
    [4A[J     total [######################################............] 76.16%
    this epoch [###########.......................................] 23.20%
         11900 iter, 15 epoch / 20 epochs
         113.1 iters/sec. Estimated time to finish: 0:00:32.934948.
    [4A[J     total [######################################............] 76.80%
    this epoch [#################.................................] 36.00%
         12000 iter, 15 epoch / 20 epochs
        113.09 iters/sec. Estimated time to finish: 0:00:32.054541.
    [4A[J     total [######################################............] 77.44%
    this epoch [########################..........................] 48.80%
         12100 iter, 15 epoch / 20 epochs
        113.08 iters/sec. Estimated time to finish: 0:00:31.171342.
    [4A[J     total [#######################################...........] 78.08%
    this epoch [##############################....................] 61.60%
         12200 iter, 15 epoch / 20 epochs
        113.08 iters/sec. Estimated time to finish: 0:00:30.288670.
    [4A[J     total [#######################################...........] 78.72%
    this epoch [#####################################.............] 74.40%
         12300 iter, 15 epoch / 20 epochs
        113.08 iters/sec. Estimated time to finish: 0:00:29.404697.
    [4A[J     total [#######################################...........] 79.36%
    this epoch [###########################################.......] 87.20%
         12400 iter, 15 epoch / 20 epochs
        114.03 iters/sec. Estimated time to finish: 0:00:28.282535.
    [4A[J16          0.353034    1.18041               0.87418        0.68949                   110.713       
    [J     total [########################################..........] 80.00%
    this epoch [..................................................]  0.00%
         12500 iter, 16 epoch / 20 epochs
        113.25 iters/sec. Estimated time to finish: 0:00:27.594325.
    [4A[J     total [########################################..........] 80.64%
    this epoch [######............................................] 12.80%
         12600 iter, 16 epoch / 20 epochs
        113.07 iters/sec. Estimated time to finish: 0:00:26.753266.
    [4A[J     total [########################################..........] 81.28%
    this epoch [############......................................] 25.60%
         12700 iter, 16 epoch / 20 epochs
        113.07 iters/sec. Estimated time to finish: 0:00:25.869927.
    [4A[J     total [########################################..........] 81.92%
    this epoch [###################...............................] 38.40%
         12800 iter, 16 epoch / 20 epochs
        112.98 iters/sec. Estimated time to finish: 0:00:25.003462.
    [4A[J     total [#########################################.........] 82.56%
    this epoch [#########################.........................] 51.20%
         12900 iter, 16 epoch / 20 epochs
        112.98 iters/sec. Estimated time to finish: 0:00:24.120367.
    [4A[J     total [#########################################.........] 83.20%
    this epoch [################################..................] 64.00%
         13000 iter, 16 epoch / 20 epochs
        112.97 iters/sec. Estimated time to finish: 0:00:23.236801.
    [4A[J     total [#########################################.........] 83.84%
    this epoch [######################################............] 76.80%
         13100 iter, 16 epoch / 20 epochs
        112.97 iters/sec. Estimated time to finish: 0:00:22.350168.
    [4A[J     total [##########################################........] 84.48%
    this epoch [############################################......] 89.60%
         13200 iter, 16 epoch / 20 epochs
        113.93 iters/sec. Estimated time to finish: 0:00:21.284692.
    [4A[J17          0.328672    1.24946               0.882133       0.682623                  117.647       
    [J     total [##########################################........] 85.12%
    this epoch [#.................................................]  2.40%
         13300 iter, 17 epoch / 20 epochs
        112.99 iters/sec. Estimated time to finish: 0:00:20.577064.
    [4A[J     total [##########################################........] 85.76%
    this epoch [#######...........................................] 15.20%
         13400 iter, 17 epoch / 20 epochs
        112.99 iters/sec. Estimated time to finish: 0:00:19.692789.
    [4A[J     total [###########################################.......] 86.40%
    this epoch [##############....................................] 28.00%
         13500 iter, 17 epoch / 20 epochs
        112.97 iters/sec. Estimated time to finish: 0:00:18.810314.
    [4A[J     total [###########################################.......] 87.04%
    this epoch [####################..............................] 40.80%
         13600 iter, 17 epoch / 20 epochs
        112.98 iters/sec. Estimated time to finish: 0:00:17.924269.
    [4A[J     total [###########################################.......] 87.68%
    this epoch [##########################........................] 53.60%
         13700 iter, 17 epoch / 20 epochs
        112.97 iters/sec. Estimated time to finish: 0:00:17.039640.
    [4A[J     total [############################################......] 88.32%
    this epoch [#################################.................] 66.40%
         13800 iter, 17 epoch / 20 epochs
           113 iters/sec. Estimated time to finish: 0:00:16.150847.
    [4A[J     total [############################################......] 88.96%
    this epoch [#######################################...........] 79.20%
         13900 iter, 17 epoch / 20 epochs
        113.01 iters/sec. Estimated time to finish: 0:00:15.263648.
    [4A[J     total [############################################......] 89.60%
    this epoch [##############################################....] 92.00%
         14000 iter, 17 epoch / 20 epochs
        114.21 iters/sec. Estimated time to finish: 0:00:14.228447.
    [4A[J18          0.293841    1.30144               0.895387       0.682325                  124.524       
    [J     total [#############################################.....] 90.24%
    this epoch [##................................................]  4.80%
         14100 iter, 18 epoch / 20 epochs
        113.33 iters/sec. Estimated time to finish: 0:00:13.456653.
    [4A[J     total [#############################################.....] 90.88%
    this epoch [########..........................................] 17.60%
         14200 iter, 18 epoch / 20 epochs
         113.4 iters/sec. Estimated time to finish: 0:00:12.566670.
    [4A[J     total [#############################################.....] 91.52%
    this epoch [###############...................................] 30.40%
         14300 iter, 18 epoch / 20 epochs
        113.38 iters/sec. Estimated time to finish: 0:00:11.686217.
    [4A[J     total [##############################################....] 92.16%
    this epoch [#####################.............................] 43.20%
         14400 iter, 18 epoch / 20 epochs
        113.38 iters/sec. Estimated time to finish: 0:00:10.804028.
    [4A[J     total [##############################################....] 92.80%
    this epoch [###########################.......................] 56.00%
         14500 iter, 18 epoch / 20 epochs
        113.39 iters/sec. Estimated time to finish: 0:00:09.921756.
    [4A[J     total [##############################################....] 93.44%
    this epoch [##################################................] 68.80%
         14600 iter, 18 epoch / 20 epochs
        113.38 iters/sec. Estimated time to finish: 0:00:09.040132.
    [4A[J     total [###############################################...] 94.08%
    this epoch [########################################..........] 81.60%
         14700 iter, 18 epoch / 20 epochs
        114.32 iters/sec. Estimated time to finish: 0:00:08.091500.
    [4A[J     total [###############################################...] 94.72%
    this epoch [###############################################...] 94.40%
         14800 iter, 18 epoch / 20 epochs
        114.33 iters/sec. Estimated time to finish: 0:00:07.215960.
    [4A[J19          0.274091    1.40398               0.901248       0.678344                  131.382       
    [J     total [###############################################...] 95.36%
    this epoch [###...............................................]  7.20%
         14900 iter, 19 epoch / 20 epochs
        113.36 iters/sec. Estimated time to finish: 0:00:06.395292.
    [4A[J     total [################################################..] 96.00%
    this epoch [#########.........................................] 20.00%
         15000 iter, 19 epoch / 20 epochs
        113.37 iters/sec. Estimated time to finish: 0:00:05.512862.
    [4A[J     total [################################################..] 96.64%
    this epoch [################..................................] 32.80%
         15100 iter, 19 epoch / 20 epochs
        113.36 iters/sec. Estimated time to finish: 0:00:04.631149.
    [4A[J     total [################################################..] 97.28%
    this epoch [######################............................] 45.60%
         15200 iter, 19 epoch / 20 epochs
        113.34 iters/sec. Estimated time to finish: 0:00:03.749654.
    [4A[J     total [################################################..] 97.92%
    this epoch [#############################.....................] 58.40%
         15300 iter, 19 epoch / 20 epochs
        113.34 iters/sec. Estimated time to finish: 0:00:02.867473.
    [4A[J     total [#################################################.] 98.56%
    this epoch [###################################...............] 71.20%
         15400 iter, 19 epoch / 20 epochs
        113.34 iters/sec. Estimated time to finish: 0:00:01.985093.
    [4A[J     total [#################################################.] 99.20%
    this epoch [#########################################.........] 84.00%
         15500 iter, 19 epoch / 20 epochs
         114.3 iters/sec. Estimated time to finish: 0:00:01.093661.
    [4A[J     total [#################################################.] 99.84%
    this epoch [################################################..] 96.80%
         15600 iter, 19 epoch / 20 epochs
         114.3 iters/sec. Estimated time to finish: 0:00:00.218720.
    [4A[J20          0.263402    1.40445               0.90505        0.683519                  138.239       
    [J


```python
serializers.save_npz('{}/{}-cifar10.model'
                         .format(args.out, args.arch), model)
```

 Chainer将训练过程抽象化，因此代码可以与其他深度学习训练一起重复使用。

CIFAR-10推断部分代码:


```python
CIFAR10_LABELS_LIST = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]
```


```python
def plot_predict_cifar(filepath, model, data, row, col,
                       scale=3., label_list=None):
    fig_width = data[0][0].shape[1] / 80 * row * scale
    fig_height = data[0][0].shape[2] / 80 * col * scale
    fig, axes = plt.subplots(row,
                             col,
                             figsize=(fig_height, fig_width))
    for i in range(row * col):
        # train[i][0] is i-th image data with size 32x32
        image, label_index = data[i]
        xp = cuda.cupy
        x = Variable(xp.asarray(image.reshape(1, 3, 32, 32)))    # test data
        #t = Variable(xp.asarray([test[i][1]]))  # labels
        y = model(x)                              # Inference result
        prediction = y.data.argmax(axis=1)
        image = image.transpose(1, 2, 0)
        print('Predicted {}-th image, prediction={}, actual={}'
              .format(i, prediction[0], label_index))
        r, c = divmod(i, col)
        axes[r][c].imshow(image)  # cmap='gray' is for black and white picture.
        if label_list is None:
            axes[r][c].set_title('Predict:{}, Answer: {}'
                                 .format(label_index, prediction[0]))
        else:
            pred = int(prediction[0])
            axes[r][c].set_title('Predict:{} {}\nAnswer:{} {}'
                                 .format(label_index, label_list[label_index],
                                         pred, label_list[pred]))
        axes[r][c].axis('off')  # do not show axis value
    plt.tight_layout(pad=0.01)   # automatic padding between subplots
    plt.savefig(filepath)
    print('Result saved to {}'.format(filepath))
```


```python
from chainer import training, iterators, serializers, optimizers, Variable, cuda
basedir = 'images'
plot_predict_cifar(os.path.join(basedir, 'cifar10_predict.png'), model,
                       train, 4, 5, scale=5., label_list=CIFAR10_LABELS_LIST)
```

    Predicted 0-th image, prediction=6, actual=6
    Predicted 1-th image, prediction=9, actual=9
    Predicted 2-th image, prediction=9, actual=9
    Predicted 3-th image, prediction=4, actual=4
    Predicted 4-th image, prediction=1, actual=1
    Predicted 5-th image, prediction=1, actual=1
    Predicted 6-th image, prediction=2, actual=2
    Predicted 7-th image, prediction=7, actual=7
    Predicted 8-th image, prediction=8, actual=8
    Predicted 9-th image, prediction=5, actual=3
    Predicted 10-th image, prediction=4, actual=4
    Predicted 11-th image, prediction=7, actual=7
    Predicted 12-th image, prediction=7, actual=7
    Predicted 13-th image, prediction=2, actual=2
    Predicted 14-th image, prediction=9, actual=9
    Predicted 15-th image, prediction=9, actual=9
    Predicted 16-th image, prediction=9, actual=9
    Predicted 17-th image, prediction=4, actual=3
    Predicted 18-th image, prediction=2, actual=2
    Predicted 19-th image, prediction=6, actual=6
    Result saved to images/cifar10_predict.png



![png](https://bennix.github.io/imgs/t11/output_41_1.png)


你可以看到，我们定义的CNN，它能成功分类大部分图像。当然，这只是一个简单的例子，您可以通过调整深度神经网络来提高模型精度！
