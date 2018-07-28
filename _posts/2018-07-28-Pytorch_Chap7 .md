---
layout: post
title: PyTorch 生成网络
date: 2018-07-28
categories: blog
tags: [PyTorch,生成网络]
description: PyTorch 生成网络
---
# 生成网络
我们在前面章节中看到的所有示例都侧重于解决诸如分类或回归之类的问题。 本章对于理解深度学习如何在无监督学习中解决问题非常有意义和重要。
在本章中，我们将培训学习如何创建的网络：

* 基于内容和特定艺术风格的图像，通常称为风格迁移
* 使用特定类型的生成对抗来生成新的面孔网络（GAN）
* 使用语言建模生成新文本


这些技术构成了深度学习领域中发生的大多数高级研究的基础。 进入每个子部分的具体细节，例如GAN和语言建模，都不属于本书的范围，因为它们本身应该有一本单独的书。 我们将了解它们如何工作以及在PyTorch中构建它们的过程。

## 神经风格迁移

我们人类可以生成具有不同精度和复杂程度的艺术作品。虽然创作艺术的过程可能是一个非常复杂的过程，但它可以看作是两个最重要因素的结合，即画什么和如何画。绘制的内容受到我们周围所看到的内容的启发，我们绘制的内容也会受到我们周围某些事物的影响。从艺术家的角度来看，这可能过于简单了，但是为了理解我们如何使用深度学习算法创建艺术作品，它非常有用。我们将训练深度学习算法从一个图像中获取内容，然后根据特定的艺术风格绘制它。如果您是艺术家或创意产业，您可以直接使用近年来的令人惊叹的研究来改进这一点并在您工作的领域内创造一些很酷的东西。即使您不是，它仍然会向您介绍生成模型领域，网络生成的新内容。

让我们了解在高级别的神经风格转移中做了什么，然后深入细节，以及构建它所需的PyTorch代码。 样式转移算法具有内容图像（C）和样式图像（S）; 该算法必须生成新图像（O），其具有来自内容图像的内容和来自样式图像的样式。 这个创建神经风格转移的过程由Leon Gates和其他人在2015年（A Neural Algorithm of Artistic style）引入。 以下是我们将使用的内容图像（C）：
![](https://bennix.github.io/imgs/7_1.png)
以下是样式图像（S）：
![](https://bennix.github.io/imgs/7_2.png)

这是我们要生成的图像：
![](https://bennix.github.io/imgs/7_3.png)


```python
from torchvision.models import vgg19
from torch.autograd import Variable
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
%pylab inline

```

    Populating the interactive namespace from numpy and matplotlib


风格迁移背后的想法从理解卷积神经网络（CNN）如何工作变得直观。当CNN被训练用于对象识别时，训练好的的CNN的早期层学习非常通用的信息，如线，曲线和形状。 CNN中的最后一层从图像（例如眼睛，建筑物和树木）捕获更高级别的概念。因此，相似图像的最后层的值往往更接近。我们采用相同的概念并将其应用于内容丢失。内容图像和生成的图像的最后一层应该相似，我们使用均方误差（MSE）计算相似度。我们使用我们的优化算法来降低损失值。

通常通过称为Gram矩阵的技术在CNN中跨多个层捕获图像的样式。 Gram矩阵计算跨多个层捕获的要素图之间的相关性。 Gram矩阵给出了计算样式的度量。类似的样式图像具有类似于Gram 矩阵的值。还使用样式图像的Gram 矩阵与生成的图像之间的MSE来计算样式损失。

我们将使用torchvision模型中提供的预训练VGG19模型。训练样式转移模型所需的步骤与任何其他深度学习模型类似，除了计算损失比分类或回归模型更复杂的事实。神经风格算法的训练可以分解为以下步骤：

1.加载数据。

2.创建VGG19模型。

3.定义内容丢失。

4.定义风格损失。

5.从VGG模型中提取跨层的损失。

6.创建优化程序。

7.训练 - 生成类似于内容图像的图像，并且样式类似于风格形象。

## 加载数据

加载数据类似于我们在第5章“计算机视觉的深度学习”中解决图像分类问题所看到的。 我们将使用预训练的VGG模型，因此我们必须使用训练预训练模型的相同值来标准化图像。

以下代码显示了我们如何做到这一点。 代码大多是不言自明的，因为我们已在前面的章节中详细讨论过它：


```python
imsize = 512 
is_cuda = torch.cuda.is_available()

prep = transforms.Compose([transforms.Resize(imsize),
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                           transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], #subtract imagenet mean
                                                std=[1,1,1]),
                           transforms.Lambda(lambda x: x.mul_(255)),
                          ])
postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1./255)),
                           transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], #add imagenet mean
                                                std=[1,1,1]),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to RGB
                           ])
postpb = transforms.Compose([transforms.ToPILImage()])
def postp(tensor): # to clip results in the range [0,1]
    t = postpa(tensor)
    t[t>1] = 1    
    t[t<0] = 0
    img = postpb(t)
    return img
```

在此代码中，我们定义了三个功能，prep执行所需的所有预处理，并使用与VGG模型训练的标准化相同的值进行标准化。 模型的输出需要归一化回原始值; postpa函数执行所需的处理。 生成的模型可能超出了可接受值的范围，并且postp函数将所有大于1的值限制为小于0到0的值。最后，image_loader函数加载图像，应用预处理转换， 并将其转换为变量。 以下函数加载样式和内容图像：


```python
def image_loader(image_name):
    image = Image.open(image_name)
    image = Variable(prep(image))
    # fake batch dimension required to fit network's input dimensions
    image = image.unsqueeze(0)
    return image
```


```python
Image.open('Images/amrut1.jpg').resize((600,600))
```




![png](https://bennix.github.io/imgs/output_8_0.png)



我们将使用优化器调整opt_img的值，以使图像更接近内容图像和样式图像。 出于这个原因，我们要求PyTorch通过提及requires_grad = True来维持梯度渐变。我们可以创建带有噪声的图像（随机数），也可以使用相同的内容图像。 在这种情况下，我们将使用内容图像。 以下代码创建内容图像：


```python
style_img = image_loader("Images/vangogh_starry_night.jpg")
content_img = image_loader("Images/amrut1.jpg")
vgg = vgg19(pretrained=True).features
for param in vgg.parameters():
    param.requires_grad = False
if is_cuda:
    style_img = style_img.cuda()
    content_img = content_img.cuda()
    vgg = vgg.cuda()

opt_img = Variable(content_img.data.clone(),requires_grad=True)

```

## 创建VGG模型

我们将从torchvisions.models加载一个预训练模型。 我们将仅使用此模型来提取特征，并且PyTorch VGG模型以这样的方式定义：所有卷积块将在特征模块中，并且完全连接或线性的层在分类器模块中。 由于我们不会训练VGG模型中的任何权重或参数，我们也会冻结模型。 以上代码演示了相同的内容。在这段代码中，我们创建了一个VGG模型，仅使用其卷积块并冻结模型的所有参数，因为我们仅将其用于提取特征。


```python


```

## 风格损失
样式损失是跨多个层计算的。 样式丢失是为每个要素图生成的gram 矩阵的MSE。 gram 矩阵表示其特征的相关值。 让我们通过使用下图和代码实现来理解gram矩阵是如何工作的。
下表显示了维度[2,3,3,3]的要素图的输出，其中包含列属性Batch_size，Channels和Values：
![](https://bennix.github.io/imgs/7_4.png)
为了计算Gram 矩阵，我们将每个通道的所有值展平，然后通过乘以其转置来找到相关性，如下表所示：
![](https://bennix.github.io/imgs/7_5.png)
我们所做的就是将每个通道的所有值平坦化为单个向量或张量。 以下代码实现了这个：


```python
class GramMatrix(nn.Module):
    
    def forward(self,input):
        b,c,h,w = input.size()
        features = input.view(b,c,h*w)
        gram_matrix =  torch.bmm(features,features.transpose(1,2))
        gram_matrix.div_(h*w)
        return gram_matrix
```

我们将GramMatrix实现为具有前向功能的另一个PyTorch模块，以便我们可以像PyTorch一样使用它。我们从这一行的输入图像中提取不同的维度：

```python
b，c，h，w = input.size()
```

这里，`b`表示批次，`c`表示过滤器或通道，`h`表示高度，`w`代表宽度。在下一步中，我们将使用以下代码来保持批次和通道尺寸的完整性，并沿高度和宽度尺寸展平所有值，如上图所示：
```python
features = input.view(b，c，h * w)
```

通过将平坦值与其转置矢量相乘来计算克矩阵。我们可以使用PyTorch批处理矩阵乘法函数来实现，该函数以torch.bmm（）的形式提供，如下面的代码所示：
```python
gram_matrix = torch.bmm(features，features.transpose(l，2))
```

我们通过将它除以元素的数量来完成对克矩阵的值的标准化。这可以防止具有大量值的特定特征图占据分数。计算GramMatrix后，计算样式丢失变得简单，这在以下代码中实现：


```python
class StyleLoss(nn.Module):
    
    def forward(self,inputs,targets):
        out = nn.MSELoss()(GramMatrix()(inputs),targets)
        return (out)
```

StyleLoss实现为另一个PyTorch层。 它计算输入GramMatrix值和样式图像GramMatrix值之间的MSE。

## 提取损失
就像我们使用第5章“计算机视觉的深度学习”中的register_forward_hook()函数提取卷积层的激活一样，我们可以提取计算样式丢失和内容丢失所需的不同卷积层的损失。 在这种情况下的一个区别是，我们需要提取多个层的输出，而不是从一个层中提取。 以下类集成了所需的更改：


```python
style_layers = [1,6,11,20,25]
content_layers = [21]
loss_layers = style_layers + content_layers

class LayerActivations():
    features=[]
    
    def __init__(self,model,layer_nums):
        
        self.hooks = []
        for layer_num in layer_nums:
            self.hooks.append(model[layer_num].register_forward_hook(self.hook_fn))
    
    def hook_fn(self,module,input,output):
        self.features.append(output)

    
    def remove(self):
        for hook in self.hooks:
            hook.remove()
        
```

`__init__`方法采用我们需要调用`register_forward_hook`方法的模型以及我们需要提取输出的层数。 `__init__`方法中的for循环遍历层数并注册提取输出所需的前向钩子。
传递给`register_forward_hook`方法的`hook_fn`之后由PyTorch调用
`hook_fn`函数注册的层。 在函数内部，我们捕获输出并将其存储在`features`数组中。

当我们不想捕获输出时，我们需要调用一次remove函数。忘记调用remove方法可能会导致内存不足异常输出累积。

让我们编写另一个实用函数，它可以提取样式和所需的输出内容图片。 以下功能也是如此：


```python
def extract_layers(layers,img,model=None):
    la = LayerActivations(model,layers)
    #Clearing the cache 
    la.features = []
    out = model(img)
    la.remove()
    return la.features
```

在extract_layers函数内部，我们通过传入模型和图层编号为LayerActivations类创建对象。 功能列表可能包含先前运行的输出，因此我们将重新启动到空列表。 然后我们通过模型传递图像，我们不会使用输出。 我们对features数组中生成的输出更感兴趣。 我们调用remove方法从模型中删除所有已注册的钩子并返回功能。 一旦我们提取目标，我们需要从创建它们的图形中分离输出。 请记住，所有这些输出都是PyTorch变量，它们保存有关如何创建它们的信息。 但是，对于我们的情况，我们只对输出值而不是图表感兴趣，因为我们不会更新样式图像或内容图像。 以下代码显示了我们如何提取样式和内容图像所需的目标：


```python
content_targets = extract_layers(content_layers,content_img,model=vgg)
content_targets = [t.detach() for t in content_targets]
style_targets = extract_layers(style_layers,style_img,model=vgg)
style_targets = [GramMatrix()(t).detach() for t in style_targets]
targets = style_targets + content_targets
```

一旦我们分离了，让我们将所有目标添加到一个列表中。

在计算样式丢失和内容丢失时，我们传递了两个名为content的列表
图层和样式图层。不同的层选择将对质量产生影响
图像生成。让我们选择与论文作者提到的相同的层次。以下代码显示了我们在此处使用的图层选择：

```python
style_layers = [1,6,11,20,25]
content_layers = [21]
loss_layers = style_layers + content_layers
```

优化器期望单个标量数量最小化。为了获得单个标量值，我们总结了到达不同层的所有损失。通常的做法是对这些损失进行加权求和，并再次选择与GitHub存储库中的文件实现中使用的相同权重（https://github.com/leongatys/PytorchNeuralstyleTransfer ）。我们的实现是作者实现的略微修改版本。以下代码描述了所使用的权重，这些权重是通过所选层中的过滤器数量计算得出的：


```python
#these are good weights settings:
style_weights = [1e3/n**2 for n in [64,128,256,512,512]]
content_weights = [1e0]
weights = style_weights + content_weights
```

为了使其可视化，我们可以打印VGG图层。 花一点时间观察我们正在抽取的图层，您可以尝试不同的图层组合。 我们将使用以下代码打印VGG图层：


```python
print(vgg)
```

    Sequential(
      (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU(inplace)
      (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (3): ReLU(inplace)
      (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (6): ReLU(inplace)
      (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (8): ReLU(inplace)
      (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (11): ReLU(inplace)
      (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (13): ReLU(inplace)
      (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (15): ReLU(inplace)
      (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (17): ReLU(inplace)
      (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (20): ReLU(inplace)
      (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (22): ReLU(inplace)
      (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (24): ReLU(inplace)
      (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (26): ReLU(inplace)
      (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (29): ReLU(inplace)
      (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (31): ReLU(inplace)
      (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (33): ReLU(inplace)
      (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (35): ReLU(inplace)
      (36): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )


## 为每个层创建损失函数

我们已经将损失函数定义为PyTorch层。 因此，让我们为不同的风格损失和内容损失创建损失层。 loss_fns是一个列表，包含一堆样式丢失对象和基于所创建数组长度的内容丢失对象。以下代码定义了该函数：


```python
loss_fns = [StyleLoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)
if is_cuda:
    loss_fns = [fn.cuda() for fn in loss_fns]
```

## 创建优化程序
通常，我们传递像VGG这样的网络参数进行训练。 但是，在此示例中，我们使用VGG模型作为特征提取器，因此，我们无法传递VGG参数。 在这里，我们将仅提供我们将优化的opt_img变量的参数，以使图像具有所需的内容和样式。 以下代码创建优化其值的优化器：


```python
optimizer = optim.LBFGS([opt_img])
```

现在我们有了训练所需的所有组件。

## 训练

与我们迄今为止训练的其他模型相比，训练方法不同。 在这里，我们需要计算多层的损失，以及每次优化器
调用后，它将更改输入图像，使其内容和样式接近目标内容和风格。 让我们看一下用于训练的代码，然后我们将逐步介绍
培训的重要步骤：


```python

#run style transfer
max_iter = 500
show_iter = 50

n_iter=[0]

while n_iter[0] <= max_iter:

    def closure():
        optimizer.zero_grad()
        
        out = extract_layers(loss_layers,opt_img,model=vgg)
        layer_losses = [weights[a] * loss_fns[a](A, targets[a]) for a,A in enumerate(out)]
        loss = sum(layer_losses)
        loss.backward()
        n_iter[0]+=1
        #print loss
        if n_iter[0]%show_iter == (show_iter-1):
            print('Iteration: %d, loss: %f'%(n_iter[0]+1, loss.data.item()))

        return loss
    
    optimizer.step(closure)
    

```

    Iteration: 50, loss: 26043.576172
    Iteration: 100, loss: 13674.988281
    Iteration: 150, loss: 11287.914062
    Iteration: 200, loss: 10558.317383
    Iteration: 250, loss: 10203.455078
    Iteration: 300, loss: 10011.093750
    Iteration: 350, loss: 9884.095703
    Iteration: 400, loss: 9795.749023
    Iteration: 450, loss: 9725.958008
    Iteration: 500, loss: 9673.065430


我们正在运行500次迭代的训练循环。 对于每次迭代，我们使用extract_layers函数计算VGG模型的不同层的输出。 在这种情况下，唯一改变的是opt_img的值，它将包含我们的样式图像。 一旦计算出输出，我们就会通过迭代输出并将它们与各自的目标一起传递给相应的损失函数来计算损失。 我们汇总所有损失并调用backward函数。 在closure函数结束时，返回损失。 调用closure方法以及max_iter的optimizer.step方法。 如果您在GPU上运行，可能需要几分钟才能运行; 如果您在CPU上运行，请尝试缩小图像的大小以使其运行得更快。

运行500个 epoch 后，我机器上生成的图像如下所示。 尝试不同的内容和样式组合，以生成有趣的图像：


```python
#display result
out_img_hr = postp(opt_img.data[0].cpu().squeeze())

imshow(out_img_hr)
gcf().set_size_inches(10,10)
```


![png](https://bennix.github.io/imgs/output_35_0.png)


在下一节中，让我们继续使用深度卷积生成对抗网络（DCGAN）生成图像。
