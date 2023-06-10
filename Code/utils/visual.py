import torch
from IPython import display
from matplotlib import pyplot as plt


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib.

    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim), axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


class Animator:
    """
    原代码来自李沐的d2l教程，这里借过来用了
    在动画中绘制数据
    """

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


class Accumulator:
    """在n个变量上累加"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def accuracy(pred_y, label, device):
    """
    计算正确的数量
    由于是soft label，0~0.3视为0，0.7~1视为1
    自己随便设置的

    """

    if label:
        cmp = pred_y >= torch.FloatTensor(pred_y.shape).fill_(0.7).to(device)

    else:
        cmp = pred_y <= torch.FloatTensor(pred_y.shape).fill_(0.3).to(device)
    return float(cmp.sum())


def evaluate_accuracy(AE, D, data_iter, device):
    """计算在指定数据集上模型的精度"""
    if isinstance(AE, torch.nn.Module):
        AE.eval()  # 将模型设置为评估模式
    if isinstance(D, torch.nn.Module):
        D.eval()

    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for i, (vi_imgs, _) in enumerate(data_iter):
            vi_imgs = vi_imgs.to(device)
            metric.add(accuracy(D(vi_imgs), 1, device), vi_imgs.shape[0])
            metric.add(accuracy(D(AE(vi_imgs)), 0, device), vi_imgs.shape[0])
    return metric[0] / metric[1]

def evaluate_accuracy1(G, D, data_iter, device):
    """
    fusion_gan
    计算在指定数据集上模型的精度
    """
    if isinstance(G, torch.nn.Module):
        G.eval()  # 将模型设置为评估模式
    if isinstance(D, torch.nn.Module):
        D.eval()

    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for i, (vi_imgs, ir_imgs) in enumerate(data_iter):
            vi_imgs = vi_imgs.to(device)
            ir_imgs = ir_imgs.to(device)
            cat_imgs = torch.cat([vi_imgs, ir_imgs], 1)
            metric.add(accuracy(D(vi_imgs), 1, device), vi_imgs.shape[0])
            metric.add(accuracy(D(G(cat_imgs, ir_imgs)), 0, device), vi_imgs.shape[0])
    return metric[0] / metric[1]