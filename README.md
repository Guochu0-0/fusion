# fusion
# 个人做的一些小实验

#### 文件说明

- fusion/Code/Model/fusiongan.py 是我个人对[Fusion_Gan](https://github.com/jiayi-ma/FusionGAN)的pytorch版本实现。
- fusion/Code/Model/rebuild.py 是我个人自定义的一个小模型，尝试从特征图重建图像。
- fusion/Code/Model/pretrainer.py 用于预训练rebuild中的生成器



#### 运行

1. cd 到fusion/Code/data下，生成txt文件

   ```
   python gen_txt.py
   ```

2. cd 到fusion/Code/utils下，训练，train.py中可以选择训练哪个模块

   ```
   python train.py
   ```

   

#### 参考

- 数据集来自TNO数据集，一个红外与可见光对齐的数据集。训练集与测试集的划分和[Fusion_Gan](https://github.com/jiayi-ma/FusionGAN)一致。
- 可视化代码部分取自[《动手学深度学习》](https://zh.d2l.ai/index.html)
- 生成txt文件代码参考 [余霆嵩《pytorch模型实用教程》](https://github.com/TingsongYu/PyTorch_Tutorial)

- 代码的组织格式参考[目标跟踪代码siamfc的一种实现](https://github.com/huanglianghua/siamfc-pytorch)，这是我见过最清晰的pytorch机器学习代码了。但是随着个人添加的代码越来越多，代码也变的有点拥挤，没有那么精简优美了。

