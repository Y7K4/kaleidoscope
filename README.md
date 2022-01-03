# kaleidoscope

本项目是太极图形课S1的大作业，由 [Y7K4](https://github.com/Y7K4) 与 [507C](https://github.com/507C) 二人合作完成。

## 背景简介

本项目的目标是实现万花筒的效果。与现实中简单的万花筒类似，我们假设在筒底 cap 处有许多彩色的装饰物，并且可以通过旋转筒底使其运动，而镜筒 tube 内侧则有数个平面镜对装饰物发出的光线反复反射。据此，项目主要由两部分组成，一是对装饰物基于 MPM 的二维仿真，二是以类似光追的方式求出镜面反射后的图案。

```
+---+
|   -----------------------+
| c |                      |
| a |         tube         -->  eye
| p |                      |
|   -----------------------+
+---+
```

## 成功效果展示

TODO

## 整体结构

```
.
├── LICENSE
├── main.py
├── mpm.py
├── obj.png
├── README.md
├── reflect.py
└── requirements.txt
```

`mpm.py` 中参考 [mpm99](https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/mpm99.py) 实现了 MPM 类，增加了从图片导入装饰物初态，镜筒旋转带动装饰物等功能。

`reflect.py` 中实现了一个反射类，其中有一个由正 N 边形组成的万花筒镜组，并以类似光追的方式对视野内的每个像点求出其反射前的原像点。

`main.py` 中实例化了两个类中的对象，实现了用户交互，并将万花筒成像结果打印到屏幕上。装饰物粒子到图片原像之间的转换接口也main中实现。

`obj.png` 提供了万花筒筒底装饰物的初态。


## 运行方式

在安装了 `taichi==0.8.5` 的前提下，执行

```bash
python3 main.py
```

在运行过程中，
* 按 SPACE 键可以在原像（万花筒cap部分内的物体）与像（万花筒成像的视觉效果）之间切换。
* 按 LEFT/RIGHT 键可以调整万花筒筒底转速。
* 按 ESCAPE 键可以退出程序。

## 合作分工

[Y7K4](https://github.com/Y7K4) 完成了 `mpm.py` 及 `reflect.py` 的小部分计算函数。

[507C](https://github.com/507C) 完成了 `reflect.py` 的主体以及 `main.py`。
