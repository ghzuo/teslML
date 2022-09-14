#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
Copyright (c) 2022
Wenzhou Institute, University of Chinese Academy of Sciences.
See the accompanying Manual for the contributors and the way to
cite this work. Comments and suggestions welcome. Please contact
Dr. Guanghong Zuo <ghzuo@ucas.ac.cn>

@Author: Dr. Guanghong Zuo
@Date: 2022-09-06 20:05:29
@Last Modified By: Dr. Guanghong Zuo
@Last Modified Time: 2022-09-06 21:28:44
'''

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch


from cycler import cycler
sns.set(font="DejaVu Sans", font_scale=1.8,
        rc={'figure.figsize': (10, 10),
            'lines.markersize': 15,
            "animation.embed_limit": 100})
sns.mpl.rc("axes", prop_cycle=cycler('color',
                                     ['#E24A33', '#348ABD', '#988ED5',
                                      '#777777', '#FBC15E', '#8EBA42',
                                      '#FFB5B8']))
sns.set_style('darkgrid', {'axes.facecolor': "0.8"})


def ln3plot(X, y):
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], y)
    plt.show()


def fit(net, data_iter, loss=torch.nn.MSELoss(), opt=None,
        n_epochs=3, lr=0.03):
    if opt is None:
        opt = torch.optim.SGD(net.parameters(), lr=lr)

    for epoch in range(1, n_epochs + 1):
        for X, y in data_iter:
            output = net(X)
            l = loss(output, y.view(-1, 1))
            opt.zero_grad()
            l.backward()
            opt.step()
        print('epoch %d, loss: %f' % (epoch, l.item()))


def pack_data(features, labels, batch_size=10, shuffle=True, njobs=2):
    # 将训练数据的特征和标签组合
    dataset = torch.utils.data.TensorDataset(features, labels)

    # 把 dataset 放入 DataLoader
    data_iter = torch.utils.data.DataLoader(
        dataset=dataset,      # torch TensorDataset format
        batch_size=batch_size,      # mini batch size
        shuffle=shuffle,               # 要不要打乱数据 (打乱比较好)
        num_workers=njobs,              # 多线程来读数据
    )

    return data_iter
