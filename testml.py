#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
Copyright (c) 2022
Wenzhou Institute, University of Chinese Academy of Sciences.
See the accompanying Manual for the contributors and the way to
cite this work. Comments and suggestions welcome. Please contact
Dr. Guanghong Zuo <ghzuo@ucas.ac.cn>

@Author: Dr. Guanghong Zuo
@Date: 2022-09-06 11:33:58
@Last Modified By: Dr. Guanghong Zuo
@Last Modified Time: 2022-09-06 16:19:27
'''
# Basic set of Python Data Analysis
from distutils.fancy_getopt import fancy_getopt
from errno import EBADMSG
from math import factorial
from pickletools import markobject
from sqlite3 import SQLITE_UPDATE
from tkinter import Grid
import numpy as np
import pandas as pd

# for plot by matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

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


def srcplot(data, label):
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    delta = 0.6/(np.max(label) - np.min(label))
    axs[0].scatter(data[:, 0], data[:, 1], marker='o',
                   color=plt.get_cmap('jet')(label*delta+0.2))
    plt.subplot(122)
    pd.value_counts(label).plot.bar(rot=0)


def clplot(score, cl, lab, data):
    # plot data
    plt.figure(13, figsize=(18, 6))
    ax = plt.subplot(131)
    score.plot(marker="o", ax=ax)

    plt.subplot(132)
    delta = 0.6/(np.max(cl) - np.min(cl))
    plt.scatter(data[:, 0], data[:, 1], marker='o', label=lab,
                color=plt.get_cmap('jet')((cl-np.min(cl))*delta+0.2))
    plt.legend()

    plt.subplot(133)
    pd.value_counts(cl).plot.bar(rot=0)


def cmplot(cmat):
    sns.heatmap(cmat, square=True, annot=True, cbar=False, fmt='.0f')
    plt.xlabel("predicted Value")
    plt.ylabel("Tree value")


def svcplot(X, y, model, plot_support=True):
    delta = 0.6/(np.max(y) - np.min(y))
    plt.scatter(X[:, 0], X[:, 1], marker='o',
                color=plt.get_cmap('jet')(y*delta+0.2))
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # create grid to evaluate model
    xs = np.linspace(xlim[0], xlim[1], 30)
    ys = np.linspace(ylim[0], ylim[1], 30)
    Ys, Xs = np.meshgrid(ys, xs)
    xy = np.vstack([Xs.ravel(), Ys.ravel()]).T
    Ps = model.decision_function(xy).reshape(Xs.shape)
    # plot decision boundary and margins
    ax.contour(Xs, Ys, Ps, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=800, linewidth=2, edgecolor='black',
                   facecolor='none')
    ax.set(xlim=xlim, ylim=ylim)


def vclplot(model, X, y, ax=None, cmap='rainbow'):
    ax = ax or plt.gca()
    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap,
               clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # fit the estimator
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    # Create a color plot with the results
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           cmap=cmap, clim=(y.min(), y.max()), zorder=1)
    ax.set(xlim=xlim, ylim=ylim)


def lrplot(lrdata):
    pd.DataFrame(lrdata).plot(grid=True)
    plt.gca().set_ylim(0, 1)
