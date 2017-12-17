---
ilayout: post
title: Chainer 导言和安装
date: 2017-12-17
categories: blog
tags: [Chainer,导言和安装]
descrption: Chainer 导言和安装
---

# 导言和安装

Chainer是“灵活的深度学习框架”。与其他着名的tensorflow或caffe深度学习框架的不同之处在于，Chainer动态地构建了神经网络，使您能够在学习时更灵活地编写您的神经网络。

特别是，我想推荐Chainer

* 深度学习框架初学者
 - 简单的环境设置：一个命令，pip install chainer，用于安装chainer。
 - 易于调试：当发生错误时，您可以看到Python堆栈跟踪日志。

* 研究人员
 - 灵活性：Chainer由于其基本概念，“边定义边运行”的架构非常灵活。您可以轻松定义复杂的网络。
 - 可扩展/可定制：也很容易开发自己的功能，你自己的神经网络层只使用python与Chainer。适合研究快速尝试新想法。

* 学生
 - 教育：Chainer是开源的，也适合学习深度学习框架。由于chainer是用python编写的，所以如果你想挖掘内部行为，你可以跳转到函数定义并读取python docstring。


# 在MacOS上如何安装Chainer

我们讲的MacOS版本是macOS Sierra （10.12.6）更高阶的版本可能在安装时有些不同，敬请注意。

## 1. 首先安装CUDA环境

没有CUDA支持的深度学习在训练过程基本会很慢（慢的令人发指），所以强烈建议安装CUDA。首先我们到Nvidia网站上下载针对macOS的CUDA环境：https://developer.nvidia.com/compute/cuda/9.1/Prod/local_installers/cuda_9.1.85_mac

遵循着安装器的指令一步步安装以后，再安装CuDNN运行库，访问链接https://developer.nvidia.com/rdp/cudnn-download 以获得CuDNN，建议7，6和5 三个版本的CuDNN都下载，按照版本号从低到高的顺序依次将对应头文件和动态链接库放到`/usr/local/cuda/include` 或者`/usr/local/cuda/lib`中。然后修改`~/.bash_profile`如下

```
export CUDA_HOME=/usr/local/cuda
export DYLD_LIBRARY_PATH="$DYLD_LIBRARY_PATH:$CUDA_HOME/lib"
export PATH="$CUDA_HOME/bin:$PATH"
export PATH="/usr/local/sbin:$PATH"
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/usr/local/cuda/lib
```

## 2. 安装Anaconda环境

国内的用户可以访问到 https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/ 下载对应的安装包，建议选择5.0.0版本https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-5.0.0-MacOSX-x86_64.pkg 。

## 3. 安装Xcode 8.2.1

通过Apple Developer官方网站下载Xcode 8.2.1 的xip格式的压缩包，解压后放在`应用程序`文件夹里，并在`终端`中执行以下命令：
```
sudo xcode-select -s /Applications/Xcode8.2.1.app/Contents/Developer/
```

## 4.  编译安装Cupy

在终端中执行如下命令：

```
git clone https://github.com/cupy/cupy.git
cd cupy
MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py install
```

如果一切正常的话，应该会把cupy正确安装。

## 5.编译安装Chainer

在终端中执行如下命令：
```
wget https://github.com/chainer/chainer/archive/v3.2.0.tar.gz
tar xzf v3.2.0.tar.gz
cd chainer-3.2.0/
MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py install
```

如果一切正常，chainer会在你的系统上正确安装。

安装以后，可以切换到`chainer-3.2.0/examples/mnist/`目录下，运行：

```
python train_mnist.py -g 0
```

如果，系统抱怨找不到CUDA 8.0的动态链接库，你可以切换到`\usr\local\cuda\lib`目录下，用`ln`命令构造如下符号链接：

```
 libcublas.8.0.dylib -> /Developer/NVIDIA/CUDA-9.0/lib/libcublas.9.0.dylib
 libcudart.8.0.dylib -> /Developer/NVIDIA/CUDA-9.0/lib/libcudart.9.0.dylib
 libcurand.8.0.dylib -> /Developer/NVIDIA/CUDA-9.0/lib/libcurand.9.0.dylib
 libcusparse.8.0.dylib -> /Developer/NVIDIA/CUDA-9.0/lib/libcusparse.9.0.dylib
 libnvrtc.8.0.dylib -> /Developer/NVIDIA/CUDA-9.0/lib/libnvrtc.9.0.dylib
```

然后再执行
```
python train_mnist.py -g 0
```

系统就不应该报错了。

现在可以开始Chainer的学习了！
