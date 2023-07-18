# -*- coding: utf-8 -*-
"""
Created on 2023-07-18 (Tue) 16:45:36

Plotting utils for TopoPathology (under development)

@author: I.Azuma
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
color_list = list(mcolors.TABLEAU_COLORS.keys())

# %%
# accuracy
def plot_acc(train_res,valid_res):
    fig,ax = plt.subplots()
    plt.plot([t[0] for t in train_res],label='train')
    plt.plot([t[0] for t in valid_res],label='valid')
    plt.xlabel("epochs")
    plt.ylabel("Accuracy")

    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')
    ax.set_axisbelow(True)
    ax.grid(color="#ababab",linewidth=0.5)
    plt.legend(shadow=True,loc='best')
    plt.show()

# loss
def plot_loss(loss_res):
    fig,ax = plt.subplots()
    plt.plot(loss_res,label='train')
    plt.xlabel("epochs")
    plt.ylabel("Loss")

    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')
    ax.set_axisbelow(True)
    ax.grid(color="#ababab",linewidth=0.5)
    plt.legend(shadow=True,loc='best')
    plt.show()

# precision
def plot_each_prec(train_res,valid_res):
    fig,ax = plt.subplots()
    prect_res = [t[3] for t in train_res]
    prect_t = np.array(prect_res).T
    precv_res = [t[3] for t in valid_res]
    precv_t = np.array(precv_res).T

    for i in range(len(prect_t)):
        plt.plot(prect_t[i],label=str(i)+": train",linestyle="solid",color=color_list[i])
        plt.plot(precv_t[i],label=str(i)+": valid",linestyle="dashed",color=color_list[i])

    plt.xlabel("epochs")
    plt.ylabel("Precision")
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')
    ax.set_axisbelow(True)
    ax.grid(color="#ababab",linewidth=0.5)
    plt.legend(loc='upper right',shadow=True,bbox_to_anchor=(1.21, 1.01))
    plt.show()
