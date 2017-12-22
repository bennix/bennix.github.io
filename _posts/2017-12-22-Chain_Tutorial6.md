---
ilayout: post
title: Chainer 入门教程（6）使用Trainer模块编写有组织的，可重复使用的清洁训练代码
date: 2017-12-22
categories: blog
tags: [Chainer,入门教程（6), 使用Trainer模块编写有组织的, 可重复使用的清洁训练代码]
descrption: Chainer 入门教程（6）使用Trainer模块编写有组织的，可重复使用的清洁训练代码
---

# 使用Trainer模块编写有组织的，可重复使用的清洁训练代码

## 用 Trainer 抽象训练代码

到目前为止，我正在以“原始”的方式实施培训代码，以解释在深度学习训练中正在进行什么样的操作。但是，使用Chainer中的Trainer模块，可以用很干净的方式编写代码。

>Trainer 模块从版本1.11开始加入，部分开源项目在没有训练器的情况下实施。因此，通过了解没有Trainer模块的训练的实施，有助于理解这些代码。 


## 使用 Trainer 的动机

例如，我们可以注意到机器学习中广泛使用的“典型”操作有很多，例如

* 在小批次随机采样的数据集迭代训练
* 训练数据和验证数据的分离，验证只用于检查训练损失，以防止过配合
* 输出日志，定期保存训练好的模型

这些操作经常被使用，Chainer在库级提供这些功能，以便用户不需要一次又一次地从零开始实现。Trainer 将为您管理训练代码！





如果使用Trainer，教程5里的代码就会变成这样


```python
from __future__ import print_function
import argparse
 
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer import serializers
 
# Network definition Chainer v2
# 1. `init_scope()` is used to initialize links for IDE friendly design.
# 2. input size of Linear layer can be omitted
class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # input size of each layer will be inferred when omitted
            self.l1 = L.Linear(n_units)  # n_in -> n_units
            self.l2 = L.Linear(n_units)  # n_units -> n_units
            self.l3 = L.Linear(n_out)    # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)
 
 

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--batchsize', '-b', type=int, default=100,
                    help='Number of images in each mini-batch')
parser.add_argument('--epoch', '-e', type=int, default=20,
                    help='Number of sweeps over the dataset to train')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--out', '-o', default='result/4',
                    help='Directory to output the result')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the training from snapshot')
parser.add_argument('--unit', '-u', type=int, default=50,
                    help='Number of units')
args = parser.parse_args(['-g','0'])

print('GPU: {}'.format(args.gpu))
print('# unit: {}'.format(args.unit))
print('# Minibatch-size: {}'.format(args.batchsize))
print('# epoch: {}'.format(args.epoch))
print('')

model = MLP(args.unit, 10)
classifier_model = L.Classifier(model)
if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
    classifier_model.to_gpu()  # Copy the model to the GPU

optimizer = chainer.optimizers.Adam()
optimizer.setup(classifier_model)

train, test = chainer.datasets.get_mnist()

train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

trainer.extend(extensions.Evaluator(test_iter, classifier_model, device=args.gpu))
trainer.extend(extensions.dump_graph('main/loss'))
trainer.extend(extensions.snapshot(), trigger=(1, 'epoch'))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(
    ['epoch', 'main/loss', 'validation/main/loss',
     'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
trainer.extend(extensions.ProgressBar())

if args.resume:
    # Resume from a snapshot
    serializers.load_npz(args.resume, trainer)

trainer.run()
serializers.save_npz('{}/mlp.model'.format(args.out), model)


```

    GPU: 0
    # unit: 50
    # Minibatch-size: 100
    # epoch: 20
    
    epoch       main/loss   validation/main/loss  main/accuracy  validation/main/accuracy  elapsed_time
    [J     total [..................................................]  0.83%
    this epoch [########..........................................] 16.67%
           100 iter, 0 epoch / 20 epochs
           inf iters/sec. Estimated time to finish: 0:00:00.
    [4A[J     total [..................................................]  1.67%
    this epoch [################..................................] 33.33%
           200 iter, 0 epoch / 20 epochs
        316.04 iters/sec. Estimated time to finish: 0:00:37.336950.
    [4A[J     total [#.................................................]  2.50%
    this epoch [#########################.........................] 50.00%
           300 iter, 0 epoch / 20 epochs
        309.43 iters/sec. Estimated time to finish: 0:00:37.811818.
    [4A[J     total [#.................................................]  3.33%
    this epoch [#################################.................] 66.67%
           400 iter, 0 epoch / 20 epochs
         306.7 iters/sec. Estimated time to finish: 0:00:37.821953.
    [4A[J     total [##................................................]  4.17%
    this epoch [#########################################.........] 83.33%
           500 iter, 0 epoch / 20 epochs
        299.77 iters/sec. Estimated time to finish: 0:00:38.362243.
    [4A[J1           0.402984    0.212212              0.888667       0.937                     4.13376       
    [J     total [##................................................]  5.00%
    this epoch [..................................................]  0.00%
           600 iter, 1 epoch / 20 epochs
        267.63 iters/sec. Estimated time to finish: 0:00:42.596603.
    [4A[J     total [##................................................]  5.83%
    this epoch [########..........................................] 16.67%
           700 iter, 1 epoch / 20 epochs
        264.76 iters/sec. Estimated time to finish: 0:00:42.680550.
    [4A[J     total [###...............................................]  6.67%
    this epoch [################..................................] 33.33%
           800 iter, 1 epoch / 20 epochs
         271.1 iters/sec. Estimated time to finish: 0:00:41.313488.
    [4A[J     total [###...............................................]  7.50%
    this epoch [#########################.........................] 50.00%
           900 iter, 1 epoch / 20 epochs
        276.08 iters/sec. Estimated time to finish: 0:00:40.206376.
    [4A[J     total [####..............................................]  8.33%
    this epoch [#################################.................] 66.67%
          1000 iter, 1 epoch / 20 epochs
        280.01 iters/sec. Estimated time to finish: 0:00:39.284346.
    [4A[J     total [####..............................................]  9.17%
    this epoch [#########################################.........] 83.33%
          1100 iter, 1 epoch / 20 epochs
        282.99 iters/sec. Estimated time to finish: 0:00:38.517941.
    [4A[J2           0.181823    0.15224               0.947384       0.9553                    6.26692       
    [J     total [#####.............................................] 10.00%
    this epoch [..................................................]  0.00%
          1200 iter, 2 epoch / 20 epochs
        274.98 iters/sec. Estimated time to finish: 0:00:39.275230.
    [4A[J     total [#####.............................................] 10.83%
    this epoch [########..........................................] 16.67%
          1300 iter, 2 epoch / 20 epochs
         272.6 iters/sec. Estimated time to finish: 0:00:39.251951.
    [4A[J     total [#####.............................................] 11.67%
    this epoch [################..................................] 33.33%
          1400 iter, 2 epoch / 20 epochs
        275.44 iters/sec. Estimated time to finish: 0:00:38.484018.
    [4A[J     total [######............................................] 12.50%
    this epoch [#########################.........................] 50.00%
          1500 iter, 2 epoch / 20 epochs
        278.08 iters/sec. Estimated time to finish: 0:00:37.758695.
    [4A[J     total [######............................................] 13.33%
    this epoch [#################################.................] 66.67%
          1600 iter, 2 epoch / 20 epochs
        280.29 iters/sec. Estimated time to finish: 0:00:37.104973.
    [4A[J     total [#######...........................................] 14.17%
    this epoch [#########################################.........] 83.33%
          1700 iter, 2 epoch / 20 epochs
         281.2 iters/sec. Estimated time to finish: 0:00:36.628781.
    [4A[J3           0.132983    0.118093              0.96105        0.9633                    8.4183        
    [J     total [#######...........................................] 15.00%
    this epoch [..................................................]  0.00%
          1800 iter, 3 epoch / 20 epochs
        276.32 iters/sec. Estimated time to finish: 0:00:36.913193.
    [4A[J     total [#######...........................................] 15.83%
    this epoch [########..........................................] 16.67%
          1900 iter, 3 epoch / 20 epochs
        274.49 iters/sec. Estimated time to finish: 0:00:36.795129.
    [4A[J     total [########..........................................] 16.67%
    this epoch [################..................................] 33.33%
          2000 iter, 3 epoch / 20 epochs
        276.38 iters/sec. Estimated time to finish: 0:00:36.182647.
    [4A[J     total [########..........................................] 17.50%
    this epoch [#########################.........................] 50.00%
          2100 iter, 3 epoch / 20 epochs
        278.16 iters/sec. Estimated time to finish: 0:00:35.591243.
    [4A[J     total [#########.........................................] 18.33%
    this epoch [#################################.................] 66.67%
          2200 iter, 3 epoch / 20 epochs
        279.64 iters/sec. Estimated time to finish: 0:00:35.045448.
    [4A[J     total [#########.........................................] 19.17%
    this epoch [#########################################.........] 83.33%
          2300 iter, 3 epoch / 20 epochs
        281.12 iters/sec. Estimated time to finish: 0:00:34.504284.
    [4A[J4           0.104697    0.11165               0.96845        0.9657                    10.5566       
    [J     total [##########........................................] 20.00%
    this epoch [..................................................]  0.00%
          2400 iter, 4 epoch / 20 epochs
        277.44 iters/sec. Estimated time to finish: 0:00:34.601534.
    [4A[J     total [##########........................................] 20.83%
    this epoch [########..........................................] 16.67%
          2500 iter, 4 epoch / 20 epochs
        275.98 iters/sec. Estimated time to finish: 0:00:34.422438.
    [4A[J     total [##########........................................] 21.67%
    this epoch [################..................................] 33.33%
          2600 iter, 4 epoch / 20 epochs
        277.29 iters/sec. Estimated time to finish: 0:00:33.899870.
    [4A[J     total [###########.......................................] 22.50%
    this epoch [#########################.........................] 50.00%
          2700 iter, 4 epoch / 20 epochs
        278.61 iters/sec. Estimated time to finish: 0:00:33.379434.
    [4A[J     total [###########.......................................] 23.33%
    this epoch [#################################.................] 66.67%
          2800 iter, 4 epoch / 20 epochs
        279.77 iters/sec. Estimated time to finish: 0:00:32.884449.
    [4A[J     total [############......................................] 24.17%
    this epoch [#########################################.........] 83.33%
          2900 iter, 4 epoch / 20 epochs
        280.96 iters/sec. Estimated time to finish: 0:00:32.388476.
    [4A[J5           0.0874788   0.0965876             0.973767       0.9704                    12.6931       
    [J     total [############......................................] 25.00%
    this epoch [..................................................]  0.00%
          3000 iter, 5 epoch / 20 epochs
        278.13 iters/sec. Estimated time to finish: 0:00:32.358470.
    [4A[J     total [############......................................] 25.83%
    this epoch [########..........................................] 16.67%
          3100 iter, 5 epoch / 20 epochs
        276.92 iters/sec. Estimated time to finish: 0:00:32.139505.
    [4A[J     total [#############.....................................] 26.67%
    this epoch [################..................................] 33.33%
          3200 iter, 5 epoch / 20 epochs
        278.04 iters/sec. Estimated time to finish: 0:00:31.650471.
    [4A[J     total [#############.....................................] 27.50%
    this epoch [#########################.........................] 50.00%
          3300 iter, 5 epoch / 20 epochs
        279.06 iters/sec. Estimated time to finish: 0:00:31.175683.
    [4A[J     total [##############....................................] 28.33%
    this epoch [#################################.................] 66.67%
          3400 iter, 5 epoch / 20 epochs
        280.07 iters/sec. Estimated time to finish: 0:00:30.706472.
    [4A[J     total [##############....................................] 29.17%
    this epoch [#########################################.........] 83.33%
          3500 iter, 5 epoch / 20 epochs
        281.02 iters/sec. Estimated time to finish: 0:00:30.247117.
    [4A[J6           0.0749929   0.0916353             0.977565       0.9734                    14.8284       
    [J     total [###############...................................] 30.00%
    this epoch [..................................................]  0.00%
          3600 iter, 6 epoch / 20 epochs
        278.61 iters/sec. Estimated time to finish: 0:00:30.149527.
    [4A[J     total [###############...................................] 30.83%
    this epoch [########..........................................] 16.67%
          3700 iter, 6 epoch / 20 epochs
        277.55 iters/sec. Estimated time to finish: 0:00:29.904001.
    [4A[J     total [###############...................................] 31.67%
    this epoch [################..................................] 33.33%
          3800 iter, 6 epoch / 20 epochs
        278.46 iters/sec. Estimated time to finish: 0:00:29.447851.
    [4A[J     total [################..................................] 32.50%
    this epoch [#########################.........................] 50.00%
          3900 iter, 6 epoch / 20 epochs
        279.34 iters/sec. Estimated time to finish: 0:00:28.997334.
    [4A[J     total [################..................................] 33.33%
    this epoch [#################################.................] 66.67%
          4000 iter, 6 epoch / 20 epochs
        280.08 iters/sec. Estimated time to finish: 0:00:28.563181.
    [4A[J     total [#################.................................] 34.17%
    this epoch [#########################################.........] 83.33%
          4100 iter, 6 epoch / 20 epochs
        280.87 iters/sec. Estimated time to finish: 0:00:28.126458.
    [4A[J7           0.0643715   0.0879003             0.979783       0.9724                    16.9707       
    [J     total [#################.................................] 35.00%
    this epoch [..................................................]  0.00%
          4200 iter, 7 epoch / 20 epochs
        278.83 iters/sec. Estimated time to finish: 0:00:27.973935.
    [4A[J     total [#################.................................] 35.83%
    this epoch [########..........................................] 16.67%
          4300 iter, 7 epoch / 20 epochs
        277.96 iters/sec. Estimated time to finish: 0:00:27.701553.
    [4A[J     total [##################................................] 36.67%
    this epoch [################..................................] 33.33%
          4400 iter, 7 epoch / 20 epochs
         278.7 iters/sec. Estimated time to finish: 0:00:27.269910.
    [4A[J     total [##################................................] 37.50%
    this epoch [#########################.........................] 50.00%
          4500 iter, 7 epoch / 20 epochs
        279.44 iters/sec. Estimated time to finish: 0:00:26.839792.
    [4A[J     total [###################...............................] 38.33%
    this epoch [#################################.................] 66.67%
          4600 iter, 7 epoch / 20 epochs
        280.12 iters/sec. Estimated time to finish: 0:00:26.417151.
    [4A[J     total [###################...............................] 39.17%
    this epoch [#########################################.........] 83.33%
          4700 iter, 7 epoch / 20 epochs
        280.78 iters/sec. Estimated time to finish: 0:00:25.999106.
    [4A[J8           0.0562755   0.0864786             0.982466       0.9738                    19.1114       
    [J     total [####################..............................] 40.00%
    this epoch [..................................................]  0.00%
          4800 iter, 8 epoch / 20 epochs
        279.01 iters/sec. Estimated time to finish: 0:00:25.805169.
    [4A[J     total [####################..............................] 40.83%
    this epoch [########..........................................] 16.67%
          4900 iter, 8 epoch / 20 epochs
        278.06 iters/sec. Estimated time to finish: 0:00:25.534438.
    [4A[J     total [####################..............................] 41.67%
    this epoch [################..................................] 33.33%
          5000 iter, 8 epoch / 20 epochs
        278.75 iters/sec. Estimated time to finish: 0:00:25.112069.
    [4A[J     total [#####################.............................] 42.50%
    this epoch [#########################.........................] 50.00%
          5100 iter, 8 epoch / 20 epochs
        279.43 iters/sec. Estimated time to finish: 0:00:24.692790.
    [4A[J     total [#####################.............................] 43.33%
    this epoch [#################################.................] 66.67%
          5200 iter, 8 epoch / 20 epochs
        280.07 iters/sec. Estimated time to finish: 0:00:24.279462.
    [4A[J     total [######################............................] 44.17%
    this epoch [#########################################.........] 83.33%
          5300 iter, 8 epoch / 20 epochs
        280.72 iters/sec. Estimated time to finish: 0:00:23.867539.
    [4A[J9           0.0483379   0.0858446             0.984882       0.9748                    21.2511       
    [J     total [######################............................] 45.00%
    this epoch [..................................................]  0.00%
          5400 iter, 9 epoch / 20 epochs
        279.17 iters/sec. Estimated time to finish: 0:00:23.641520.
    [4A[J     total [######################............................] 45.83%
    this epoch [########..........................................] 16.67%
          5500 iter, 9 epoch / 20 epochs
        278.39 iters/sec. Estimated time to finish: 0:00:23.348436.
    [4A[J     total [#######################...........................] 46.67%
    this epoch [################..................................] 33.33%
          5600 iter, 9 epoch / 20 epochs
           279 iters/sec. Estimated time to finish: 0:00:22.939125.
    [4A[J     total [#######################...........................] 47.50%
    this epoch [#########################.........................] 50.00%
          5700 iter, 9 epoch / 20 epochs
         279.6 iters/sec. Estimated time to finish: 0:00:22.531912.
    [4A[J     total [########################..........................] 48.33%
    this epoch [#################################.................] 66.67%
          5800 iter, 9 epoch / 20 epochs
        280.16 iters/sec. Estimated time to finish: 0:00:22.130334.
    [4A[J     total [########################..........................] 49.17%
    this epoch [#########################################.........] 83.33%
          5900 iter, 9 epoch / 20 epochs
        280.68 iters/sec. Estimated time to finish: 0:00:21.732882.
    [4A[J10          0.0439513   0.0861788             0.986449       0.9741                    23.3961       
    [J     total [#########################.........................] 50.00%
    this epoch [..................................................]  0.00%
          6000 iter, 10 epoch / 20 epochs
        279.23 iters/sec. Estimated time to finish: 0:00:21.487977.
    [4A[J     total [#########################.........................] 50.83%
    this epoch [########..........................................] 16.67%
          6100 iter, 10 epoch / 20 epochs
         278.5 iters/sec. Estimated time to finish: 0:00:21.185242.
    [4A[J     total [#########################.........................] 51.67%
    this epoch [################..................................] 33.33%
          6200 iter, 10 epoch / 20 epochs
        278.99 iters/sec. Estimated time to finish: 0:00:20.789242.
    [4A[J     total [##########################........................] 52.50%
    this epoch [#########################.........................] 50.00%
          6300 iter, 10 epoch / 20 epochs
        279.52 iters/sec. Estimated time to finish: 0:00:20.391839.
    [4A[J     total [##########################........................] 53.33%
    this epoch [#################################.................] 66.67%
          6400 iter, 10 epoch / 20 epochs
         279.7 iters/sec. Estimated time to finish: 0:00:20.021794.
    [4A[J     total [###########################.......................] 54.17%
    this epoch [#########################################.........] 83.33%
          6500 iter, 10 epoch / 20 epochs
        280.22 iters/sec. Estimated time to finish: 0:00:19.627598.
    [4A[J11          0.038202    0.089263              0.988148       0.9744                    25.5662       
    [J     total [###########################.......................] 55.00%
    this epoch [..................................................]  0.00%
          6600 iter, 11 epoch / 20 epochs
        278.97 iters/sec. Estimated time to finish: 0:00:19.356945.
    [4A[J     total [###########################.......................] 55.83%
    this epoch [########..........................................] 16.67%
          6700 iter, 11 epoch / 20 epochs
        278.36 iters/sec. Estimated time to finish: 0:00:19.039910.
    [4A[J     total [############################......................] 56.67%
    this epoch [################..................................] 33.33%
          6800 iter, 11 epoch / 20 epochs
        278.83 iters/sec. Estimated time to finish: 0:00:18.649145.
    [4A[J     total [############################......................] 57.50%
    this epoch [#########################.........................] 50.00%
          6900 iter, 11 epoch / 20 epochs
        279.33 iters/sec. Estimated time to finish: 0:00:18.257656.
    [4A[J     total [#############################.....................] 58.33%
    this epoch [#################################.................] 66.67%
          7000 iter, 11 epoch / 20 epochs
        279.82 iters/sec. Estimated time to finish: 0:00:17.868794.
    [4A[J     total [#############################.....................] 59.17%
    this epoch [#########################################.........] 83.33%
          7100 iter, 11 epoch / 20 epochs
        280.29 iters/sec. Estimated time to finish: 0:00:17.481805.
    [4A[J12          0.0344433   0.0895551             0.989415       0.9752                    27.7029       
    [J     total [##############################....................] 60.00%
    this epoch [..................................................]  0.00%
          7200 iter, 12 epoch / 20 epochs
        279.12 iters/sec. Estimated time to finish: 0:00:17.196764.
    [4A[J     total [##############################....................] 60.83%
    this epoch [########..........................................] 16.67%
          7300 iter, 12 epoch / 20 epochs
        278.56 iters/sec. Estimated time to finish: 0:00:16.872522.
    [4A[J     total [##############################....................] 61.67%
    this epoch [################..................................] 33.33%
          7400 iter, 12 epoch / 20 epochs
           279 iters/sec. Estimated time to finish: 0:00:16.487363.
    [4A[J     total [###############################...................] 62.50%
    this epoch [#########################.........................] 50.00%
          7500 iter, 12 epoch / 20 epochs
        279.46 iters/sec. Estimated time to finish: 0:00:16.102487.
    [4A[J     total [###############################...................] 63.33%
    this epoch [#################################.................] 66.67%
          7600 iter, 12 epoch / 20 epochs
        279.88 iters/sec. Estimated time to finish: 0:00:15.721217.
    [4A[J     total [################################..................] 64.17%
    this epoch [#########################################.........] 83.33%
          7700 iter, 12 epoch / 20 epochs
        280.31 iters/sec. Estimated time to finish: 0:00:15.339942.
    [4A[J13          0.0307968   0.0856079             0.990365       0.9744                    29.8438       
    [J     total [################################..................] 65.00%
    this epoch [..................................................]  0.00%
          7800 iter, 13 epoch / 20 epochs
        279.21 iters/sec. Estimated time to finish: 0:00:15.042426.
    [4A[J     total [################################..................] 65.83%
    this epoch [########..........................................] 16.67%
          7900 iter, 13 epoch / 20 epochs
        278.69 iters/sec. Estimated time to finish: 0:00:14.711530.
    [4A[J     total [#################################.................] 66.67%
    this epoch [################..................................] 33.33%
          8000 iter, 13 epoch / 20 epochs
        279.11 iters/sec. Estimated time to finish: 0:00:14.331201.
    [4A[J     total [#################################.................] 67.50%
    this epoch [#########################.........................] 50.00%
          8100 iter, 13 epoch / 20 epochs
         279.5 iters/sec. Estimated time to finish: 0:00:13.953544.
    [4A[J     total [##################################................] 68.33%
    this epoch [#################################.................] 66.67%
          8200 iter, 13 epoch / 20 epochs
        279.88 iters/sec. Estimated time to finish: 0:00:13.577207.
    [4A[J     total [##################################................] 69.17%
    this epoch [#########################################.........] 83.33%
          8300 iter, 13 epoch / 20 epochs
        280.27 iters/sec. Estimated time to finish: 0:00:13.201457.
    [4A[J14          0.0273088   0.0911323             0.991165       0.9749                    31.9861       
    [J     total [###################################...............] 70.00%
    this epoch [..................................................]  0.00%
          8400 iter, 14 epoch / 20 epochs
        279.27 iters/sec. Estimated time to finish: 0:00:12.890678.
    [4A[J     total [###################################...............] 70.83%
    this epoch [########..........................................] 16.67%
          8500 iter, 14 epoch / 20 epochs
        278.76 iters/sec. Estimated time to finish: 0:00:12.555650.
    [4A[J     total [###################################...............] 71.67%
    this epoch [################..................................] 33.33%
          8600 iter, 14 epoch / 20 epochs
        279.15 iters/sec. Estimated time to finish: 0:00:12.179819.
    [4A[J     total [####################################..............] 72.50%
    this epoch [#########################.........................] 50.00%
          8700 iter, 14 epoch / 20 epochs
        279.56 iters/sec. Estimated time to finish: 0:00:11.804459.
    [4A[J     total [####################################..............] 73.33%
    this epoch [#################################.................] 66.67%
          8800 iter, 14 epoch / 20 epochs
        279.92 iters/sec. Estimated time to finish: 0:00:11.431826.
    [4A[J     total [#####################################.............] 74.17%
    this epoch [#########################################.........] 83.33%
          8900 iter, 14 epoch / 20 epochs
        280.27 iters/sec. Estimated time to finish: 0:00:11.060850.
    [4A[J15          0.0249426   0.0956454             0.992082       0.9749                    34.1325       
    [J     total [#####################################.............] 75.00%
    this epoch [..................................................]  0.00%
          9000 iter, 15 epoch / 20 epochs
        279.29 iters/sec. Estimated time to finish: 0:00:10.741581.
    [4A[J     total [#####################################.............] 75.83%
    this epoch [########..........................................] 16.67%
          9100 iter, 15 epoch / 20 epochs
        278.71 iters/sec. Estimated time to finish: 0:00:10.404949.
    [4A[J     total [######################################............] 76.67%
    this epoch [################..................................] 33.33%
          9200 iter, 15 epoch / 20 epochs
        279.07 iters/sec. Estimated time to finish: 0:00:10.033392.
    [4A[J     total [######################################............] 77.50%
    this epoch [#########################.........................] 50.00%
          9300 iter, 15 epoch / 20 epochs
        279.41 iters/sec. Estimated time to finish: 0:00:09.663085.
    [4A[J     total [#######################################...........] 78.33%
    this epoch [#################################.................] 66.67%
          9400 iter, 15 epoch / 20 epochs
        279.75 iters/sec. Estimated time to finish: 0:00:09.293894.
    [4A[J     total [#######################################...........] 79.17%
    this epoch [#########################################.........] 83.33%
          9500 iter, 15 epoch / 20 epochs
        280.08 iters/sec. Estimated time to finish: 0:00:08.926009.
    [4A[J16          0.0215279   0.0915036             0.993231       0.9759                    36.2947       
    [J     total [########################################..........] 80.00%
    this epoch [..................................................]  0.00%
          9600 iter, 16 epoch / 20 epochs
        279.17 iters/sec. Estimated time to finish: 0:00:08.596850.
    [4A[J     total [########################################..........] 80.83%
    this epoch [########..........................................] 16.67%
          9700 iter, 16 epoch / 20 epochs
        278.64 iters/sec. Estimated time to finish: 0:00:08.254477.
    [4A[J     total [########################################..........] 81.67%
    this epoch [################..................................] 33.33%
          9800 iter, 16 epoch / 20 epochs
        278.97 iters/sec. Estimated time to finish: 0:00:07.886277.
    [4A[J     total [#########################################.........] 82.50%
    this epoch [#########################.........................] 50.00%
          9900 iter, 16 epoch / 20 epochs
        278.75 iters/sec. Estimated time to finish: 0:00:07.533553.
    [4A[J     total [#########################################.........] 83.33%
    this epoch [#################################.................] 66.67%
         10000 iter, 16 epoch / 20 epochs
        278.61 iters/sec. Estimated time to finish: 0:00:07.178550.
    [4A[J     total [##########################################........] 84.17%
    this epoch [#########################################.........] 83.33%
         10100 iter, 16 epoch / 20 epochs
        278.71 iters/sec. Estimated time to finish: 0:00:06.817097.
    [4A[J17          0.0187092   0.0988569             0.994448       0.9747                    38.6819       
    [J     total [##########################################........] 85.00%
    this epoch [..................................................]  0.00%
         10200 iter, 17 epoch / 20 epochs
        277.01 iters/sec. Estimated time to finish: 0:00:06.497974.
    [4A[J     total [##########################################........] 85.83%
    this epoch [########..........................................] 16.67%
         10300 iter, 17 epoch / 20 epochs
        276.03 iters/sec. Estimated time to finish: 0:00:06.158807.
    [4A[J     total [###########################################.......] 86.67%
    this epoch [################..................................] 33.33%
         10400 iter, 17 epoch / 20 epochs
         275.6 iters/sec. Estimated time to finish: 0:00:05.805548.
    [4A[J     total [###########################################.......] 87.50%
    this epoch [#########################.........................] 50.00%
         10500 iter, 17 epoch / 20 epochs
        275.23 iters/sec. Estimated time to finish: 0:00:05.450053.
    [4A[J     total [############################################......] 88.33%
    this epoch [#################################.................] 66.67%
         10600 iter, 17 epoch / 20 epochs
        276.02 iters/sec. Estimated time to finish: 0:00:05.072159.
    [4A[J     total [############################################......] 89.17%
    this epoch [#########################################.........] 83.33%
         10700 iter, 17 epoch / 20 epochs
        276.04 iters/sec. Estimated time to finish: 0:00:04.709460.
    [4A[J18          0.0173003   0.0992972             0.994682       0.9747                    41.3301       
    [J     total [#############################################.....] 90.00%
    this epoch [..................................................]  0.00%
         10800 iter, 18 epoch / 20 epochs
         274.1 iters/sec. Estimated time to finish: 0:00:04.377921.
    [4A[J     total [#############################################.....] 90.83%
    this epoch [########..........................................] 16.67%
         10900 iter, 18 epoch / 20 epochs
        272.91 iters/sec. Estimated time to finish: 0:00:04.030630.
    [4A[J     total [#############################################.....] 91.67%
    this epoch [################..................................] 33.33%
         11000 iter, 18 epoch / 20 epochs
         272.5 iters/sec. Estimated time to finish: 0:00:03.669707.
    [4A[J     total [##############################################....] 92.50%
    this epoch [#########################.........................] 50.00%
         11100 iter, 18 epoch / 20 epochs
        272.15 iters/sec. Estimated time to finish: 0:00:03.306946.
    [4A[J     total [##############################################....] 93.33%
    this epoch [#################################.................] 66.67%
         11200 iter, 18 epoch / 20 epochs
         272.8 iters/sec. Estimated time to finish: 0:00:02.932570.
    [4A[J     total [###############################################...] 94.17%
    this epoch [#########################################.........] 83.33%
         11300 iter, 18 epoch / 20 epochs
        273.03 iters/sec. Estimated time to finish: 0:00:02.563774.
    [4A[J19          0.015928    0.107987              0.995332       0.9725                    43.8311       
    [J     total [###############################################...] 95.00%
    this epoch [..................................................]  0.00%
         11400 iter, 19 epoch / 20 epochs
        271.39 iters/sec. Estimated time to finish: 0:00:02.210802.
    [4A[J     total [###############################################...] 95.83%
    this epoch [########..........................................] 16.67%
         11500 iter, 19 epoch / 20 epochs
        270.38 iters/sec. Estimated time to finish: 0:00:01.849222.
    [4A[J     total [################################################..] 96.67%
    this epoch [################..................................] 33.33%
         11600 iter, 19 epoch / 20 epochs
        269.95 iters/sec. Estimated time to finish: 0:00:01.481766.
    [4A[J     total [################################################..] 97.50%
    this epoch [#########################.........................] 50.00%
         11700 iter, 19 epoch / 20 epochs
        269.71 iters/sec. Estimated time to finish: 0:00:01.112312.
    [4A[J     total [#################################################.] 98.33%
    this epoch [#################################.................] 66.67%
         11800 iter, 19 epoch / 20 epochs
        270.29 iters/sec. Estimated time to finish: 0:00:00.739940.
    [4A[J     total [#################################################.] 99.17%
    this epoch [#########################################.........] 83.33%
         11900 iter, 19 epoch / 20 epochs
        270.73 iters/sec. Estimated time to finish: 0:00:00.369376.
    [4A[J20          0.0160734   0.109226              0.995048       0.9739                    46.2573       
    [J     total [##################################################] 100.00%
    this epoch [..................................................]  0.00%
         12000 iter, 20 epoch / 20 epochs
        269.42 iters/sec. Estimated time to finish: 0:00:00.
    [4A[J

比较前面教程的代码，看看代码是干净的！该代码甚至不显式包含for循环，以及随机排列的小批量和保存功能。

代码长度也缩短了近一半，甚至比之前的代码支持更多的功能。

* 计算在验证数据集上的损失，准确性
* 定期保存训练快照（包括优化器和模型数据）。您可以暂停和恢复训练。
* 打印记录格式化的训练状态的进度条。
* 将训练结果以json格式文本输出到日志文件。
 
然而，它与以前的代码有很大的不同，用户可能不知道发生了什么。训练器使用了若干个模块。让我们看看每个模块的功能概述。


## 数据集

输入数据应该以数据集格式准备，以便迭代器可以处理。

在这个例子中，数据集没有显式出现，但已经准备好了


```python
train, test = chainer.datasets.get_mnist()
```

这列训练集和测试集形成了`TupleDataset`，具体可以回顾前面提及的MNIST数据集介绍。

有几个数据集类，`TupleDataset`，`ImageDataset`等，甚至你可以通过使用`DatasetMixin`定义您的自定义数据集类。

所有的数据集遵循的通用规则是当数据是数据集实例数据`[i]`指向第i个数据。

通常它由输入数据和目标数据（答案）组成，其中`data[i][0]`是第i个输入数据，`data[i][1]`是第`i`个目标数据。但是，根据问题，它可以只有一个元素，甚至可以是两个以上的元素。


作用：用于准备输入值以提供数据的索引访问。具体来说，第i个数据可以通过`data[i]`来访问，以便Iterator可以处理。


## Iterator (迭代器)

迭代器管理着循环训练数据的小批量数据集。


```python
train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
```

这一行代码替代了下面的代码：
```
# Learning loop
for epoch in six.moves.range(1, n_epoch + 1):
    # training
    perm = np.random.permutation(N)
    for i in six.moves.range(0, N, batchsize):
        x = chainer.Variable(xp.asarray(train[perm[i:i + batchsize]][0]))
        t = chainer.Variable(xp.asarray(train[perm[i:i + batchsize]][1]))

        # Pass the loss function (Classifier defines it) and its arguments
        optimizer.update(classifier_model, x, t)
```

对于测试（验证）数据集也是如此


```python
 test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)
```

顶替了如下代码：

```
for i in six.moves.range(0, N_test, batchsize):
    index = np.asarray(list(range(i, i + batchsize)))
    x = chainer.Variable(xp.asarray(test[index][0]), volatile='on')
    t = chainer.Variable(xp.asarray(test[index][1]), volatile='on')</strong>
    loss = classifier_model(x, t)
```

由np.permutation实现的小批量的随机采样被替换为仅仅设定 shuffle 标志为 True 或者 False (缺省是 True)。

目前提供了2个Iterator类，

* SerialIterator是最基本的类。
* MultiProcessIterator在后台提供多进程数据准备支持。

作用：从数据集中构建小批量（包括使用多进程的后台准备支持），并将其传递给更新器。

## Updater （更新器）

创建迭代器后，将其与optmizer一起设置为Updater，


```python
updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
```

更新器负责调用优化器的更新函数，这意味着它对应于调用
```
optimizer.update(classifier_model, x, t)
```

目前提供了2个更新程序类（和1个更新程序）

* StandardUpdater是基本类。
* ParallelUpdater是为了同时使用多个GPU。

作用：接收来自Iterator的小批量，计算损失并调用优化器的更新。


## Trainer（训练器）

最后，训练器实例可以通过更新程序创建


```python
trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
```

如果要开始训练，只需执行，

```
trainer.run()
```

通常在开始调用训练器的运行之前注册扩展，见下文

角色：管理训练生命周期以及扩展的注册。

## 训练器扩展

训练器扩展可以通过trainer.extend（）函数进行注册。

在这个例子中使用了这些扩展， 

* Evaluator（评估器）
计算验证损失和准确性，并将其打印并记录到文件中。
* LogReport（日志报告器）
在训练器中由out参数指定的目录中以json格式打印日志文件。
* PrintReport（打印报告器）
打印出标准输出（控制台）显示训练状态。
* ProgressBar（进度条）
显示训练的当前进度。
* snapshot（快照）
定期保存训练器状态（包括模型，优化器信息）。通过设置这个扩展，你可以暂停和恢复训练。
* dump_graph
将神经网络计算图保存至dot格式的文件
 
角色：挂钩触发器使得让训练器在特定的时间做特定动作


## 训练器架构总结

![](trainer3-800x245.png)

上图对于训练器模块进行了抽象和概括。


## 使用训练器模块的好处

- 使用MultiProcessIterator进行多进程数据准备

Python具有GIL特性，所以即使使用多线程，它的线程也不会在“并行”中执行。如果代码包含大量的数据预处理（例如样本增强，在输入之前添加噪声），则可以使用MultiProcessIterator获得好处。

- 使用多个GPU

- ParallelUpdater 或者 MultiProcessParallelUpdater

- 一旦你做了自己的扩展，这些扩展将是有用的和可重用的

- PrintReport

- ProgressBar

- LogReport

- 日志格式为json格式，易于加载和绘制学习曲线图等

- 快照

等等······ 有这么多好处，为什么不使用它呢!

## 实例化训练好的模型并加载模型


```python
model_saved_name='{}/mlp.model'.format(args.out)
print(model_saved_name)
serializers.load_npz(model_saved_name, model)
```

    result/4/mlp.model


在这里，请注意模型可以在实例化模型之后加载。在训练阶段保存模型时，该模型必须具有相同的结构（隐藏单元大小，网络层深度等）。

## 将输入数据送入加载的模型以获得推断结果

下面的代码是从测试输入数据x得到推断结果y。


```python
from chainer import Variable
from chainer import cuda
xp = np if args.gpu < 0 else cuda.cupy
for i in range(len(test)):
    x = Variable(xp.asarray([test[i][0]]))    # test data
    # t = Variable(xp.asarray([test[i][1]]))  # labels
    y = model(x)                              # Inference result
```

## 可视化结果

您可能希望看到把推断结果与输入图像放在一起更准确地反映训练结果。此代码绘制测试输入图像及其推断结果的图形。



```python
import matplotlib.pyplot as plt
import numpy as np
```


```python
%matplotlib inline
```


```python
ROW = 4
COLUMN = 5
# show graphical results of first 20 data to understand what's going on in inference stage
plt.figure(figsize=(15, 10))
for i in range(ROW * COLUMN):
    # Example of predicting the test input one by one.
    x = Variable(xp.asarray([test[i][0]]))  # test data
    # t = Variable(xp.asarray([test[i][1]]))  # labels
    y = model(x)
    np.set_printoptions(precision=2, suppress=True)
    print('{}-th image: answer = {}, predict = {}'.format(i, test[i][1], F.softmax(y).data))
    prediction = y.data.argmax(axis=1)
    example = (test[i][0] * 255).astype(np.int32).reshape(28, 28)
    plt.subplot(ROW, COLUMN, i+1)
    plt.imshow(example, cmap='gray')
    plt.title("No.{0} / Answer:{1}, Predict:{2}".format(i, test[i][1], prediction))
    plt.axis("off")
plt.tight_layout()
plt.savefig('inference.png')
```

    0-th image: answer = 7, predict = [[ 0.  0.  0.  0.  0.  0.  0.  1.  0.  0.]]
    1-th image: answer = 2, predict = [[ 0.  0.  1.  0.  0.  0.  0.  0.  0.  0.]]
    2-th image: answer = 1, predict = [[ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.]]
    3-th image: answer = 0, predict = [[ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]]
    4-th image: answer = 4, predict = [[ 0.  0.  0.  0.  1.  0.  0.  0.  0.  0.]]
    5-th image: answer = 1, predict = [[ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.]]
    6-th image: answer = 4, predict = [[ 0.  0.  0.  0.  1.  0.  0.  0.  0.  0.]]
    7-th image: answer = 9, predict = [[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  1.]]
    8-th image: answer = 5, predict = [[ 0.    0.    0.    0.    0.    0.01  0.99  0.    0.    0.  ]]
    9-th image: answer = 9, predict = [[ 0.    0.    0.    0.    0.14  0.    0.    0.    0.    0.86]]
    10-th image: answer = 0, predict = [[ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]]
    11-th image: answer = 6, predict = [[ 0.  0.  0.  0.  0.  0.  1.  0.  0.  0.]]
    12-th image: answer = 9, predict = [[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  1.]]
    13-th image: answer = 0, predict = [[ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]]
    14-th image: answer = 1, predict = [[ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.]]
    15-th image: answer = 5, predict = [[ 0.  0.  0.  0.  0.  1.  0.  0.  0.  0.]]
    16-th image: answer = 9, predict = [[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  1.]]
    17-th image: answer = 7, predict = [[ 0.  0.  0.  0.  0.  0.  0.  1.  0.  0.]]
    18-th image: answer = 3, predict = [[ 0.    0.    0.    0.99  0.    0.    0.    0.    0.01  0.  ]]
    19-th image: answer = 4, predict = [[ 0.  0.  0.  0.  1.  0.  0.  0.  0.  0.]]



![png](output_32_1.png)


这就是基于MNIST数据集进行深度学习教程的全部内容。现在您已经学习了如何使用深度学习框架的基础知识。如何编写训练码，如何用Chainer编写推断码。
