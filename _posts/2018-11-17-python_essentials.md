---
title: 'Python 精要'
date: 2018-11-17
permalink: /posts/2018/11/python_essentials
tags:
  - python
---

掌握了其他编程语言，再来学习 Python，需要知道什么？

## 1. 变量及数据类型

基本数据类型

* int
* bool
* float
* str

type 函数获取类型，类型名函数做类型转换，如int()。

List 是 Python 重要的数据结构，重点总结以下用法
* 下标查找 subindex 
* 截取 slice 
* 增加 + 
* 删除 del (支持下标or slice修改)
* 复制
    * 指针赋值 y=x
    * 复制复制 y = list(x) OR y = x[:]

## 2. 函数

* type、max、round、len、help(👍)
* 帮助，例：
    * help(max) , help(str) 
    * ?max

## 3. 包及环境

包：一系列代码的目录，包含functions, types, methods, constants

包导入方法:

* import numpy 基本
* import numpy as np 重命名导入
* from numpy import array 只导入一部分

包管理工具： pip etc.

环境管理工具：pyenv etc.

集包管理和环境管理与一身的工具：conda.

个人目前还是比较习惯于 pip + pyenv 的方式，可能因为从 ruby 转过来的原因吧。
个人觉得 ruby 包和环境管理的工具 bundler + rvm 比 python 方便许多。
很多 python 开源项目不指明 python 版本，也不写明依赖的第三方包列表，总是运行了代码才知道 python 代码合不合适，一个个安装缺失的包。
对于 ruby 项目，一般都会有一个 Gemfile 说明该项目使用的 ruby 版本及依赖了哪些第三方包，并且会使用 Gemfile.lock 锁定包的版本，以防第三方包更新导致的不兼容。

## 4. 调试

可以使用集成环境，但是作为一个极简主义的程序员，还是喜欢使用自带的命令行调试工具 pdb。

```
python -m pdb test.py
```
常用的调试指令如下：

* b 设置断点
* l 列出当前位置的代码
* c 继续
* s 跳入
* n 下一步

## 5. 重要数据分析工具

### Numpy

相比于List的优点：

* 方便的数学计算操作
* 速度、性能有保证

注意：

* 元素只能是一种类型
* 会改变一些方法

### Matplotlib

用于可是化，简单的绘图类型：

* 折线图
* 散点图
* 柱状图

### Pandas

用于复杂数据的分析


### Jupyter notebook

* 安装  pip3 install jupyter
* 启动服务 jupyter notebook

之后就可以在本地浏览器内写kernel了，很赞

不过，关键还是需要我从数据中发现有用的信息，利用好工具，掌握数据分析的方法

### TensorFlow 

机器学习、深度学习神器，有了它，你也可以很容易做到语音转文字、图片风格迁移、图片标注、生成假人头像等看似高端的深度学习应用。
TensorFlow 的官方 tutorials 讲了很多有趣的应用，感兴趣赶紧去试试吧。
戳 https://www.tensorflow.org/tutorials/ 。
