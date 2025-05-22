import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from sklearn.metrics import balanced_accuracy_score

"""
Adapted from https://colab.research.google.com/drive/1H_XlTbNvjxlAXMW5NuBDWhxF3F2Osg1F?usp=sharing
"""
def calc_bins(preds,true):
  # Assign each prediction to a bin
    num_bins = 5
    bins = np.linspace(0.0, 1, num_bins+1)
    
    binned = np.digitize(preds, bins)-1
    print(bins)

    # Save the accuracy, confidence and size of each bin
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)

    for bin in range(num_bins):
        bin_sizes[bin] = len(preds[binned == bin])
        if bin_sizes[bin] > 0:
            true_bin = true[binned == bin]
            pred_bin = preds[binned==bin]
            bin_accs[bin] = bin_acc(true_bin)
            bin_confs[bin] = confidence(pred_bin,true_bin)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    return bin_centers, binned, bin_accs, bin_confs, bin_sizes

def calc_bins_nb(preds,true):
  # Assign each prediction to a bin
    num_bins = 10
    bins = np.linspace(0.0, 1, num_bins+1)
    
    binned = np.digitize(preds, bins)-1
    print(bins)

    # Save the accuracy, confidence and size of each bin
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)

    for bin in range(num_bins):
        bin_sizes[bin] = len(preds[binned == bin])
        if bin_sizes[bin] > 0:
            true_bin = true[binned == bin]
            pred_bin = preds[binned==bin]
            bin_accs[bin] = not_bind_bin_acc(true_bin)
            bin_confs[bin] = confidence(pred_bin,true_bin)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    return bin_centers, binned, bin_accs, bin_confs, bin_sizes

def get_metrics(preds,true):
    ECE = 0
    ABCE = 0
    MCE = 0
    bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(preds,true)

    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        print("Bin:", i,"Bin size:",bin_sizes[i],"Sum of bin sizes:",sum(bin_sizes),"Acc-Conf:",abs_conf_dif)
        print("Acc:",bin_accs[i],"Conf:",bin_confs[i])
        ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
        ABCE += (abs_conf_dif / (len(bins)-1))
        MCE = max(MCE, abs_conf_dif)

    return ECE, MCE,ABCE

def get_metrics_nb(preds,true):
    ECE = 0
    ABCE = 0
    MCE = 0
    bins, _, bin_accs, bin_confs, bin_sizes = calc_bins_nb(preds,true)

    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        print("Bin:", i,"Bin size:",bin_sizes[i],"Sum of bin sizes:",sum(bin_sizes),"Acc-Conf:",abs_conf_dif)
        print("Acc:",bin_accs[i],"Conf:",bin_confs[i])
        ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
        ABCE += (abs_conf_dif / (len(bins)-1))
        MCE = max(MCE, abs_conf_dif)

    return ECE, MCE,ABCE

def draw_reliability_graph(preds,true,name):
    print(name)
    ECE, MCE,ABCE = get_metrics(preds,true)
    bins, _, bin_accs, _, _ = calc_bins(preds,true)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()

    # x/y limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # x/y labels
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')

    # Create grid
    ax.set_axisbelow(True) 
    ax.grid(color='gray', linestyle='dashed')

    # Error bars
    plt.bar(bins, bins,  width=0.1, alpha=0.3, edgecolor='red', color='r', hatch='\\')

    # Draw bars and identity line
    plt.bar(bins, bin_accs, width=0.1, alpha=1, edgecolor='black', color='b')
    difference = bin_accs - bins
    difference = np.where(difference > 0, difference, 0)

    # 3) Plot only the exceeding portion in red, stacked on top of the diagonal
    #    So "bottom" is the diagonal, and height is the difference
    plt.bar(bins, difference, bottom=bins, width=0.1, alpha=0.3, edgecolor='red', color='r', 
            hatch='\\')
    plt.plot([0,1],[0,1], '--', color='gray', linewidth=2)

    # Equally spaced axes
    plt.gca().set_aspect('equal', adjustable='box')

    # ECE and MCE legend
    ECE_patch = mpatches.Patch(color='green', label=f'ECE = {ECE*100:.2f}% ABCE={ABCE*100:.2f}%')
    MCE_patch = mpatches.Patch(color='red', label='MCE = {:.2f}%'.format(MCE*100))
    plt.legend(handles=[ECE_patch, MCE_patch])
    plt.title(f"{name}")
    plt.tight_layout()
    #plt.show()

    plt.savefig(f'./results/{name}_metrics.png', bbox_inches='tight')

def draw_reliability_graph_marginal(preds, true, name):
    # Calculate metrics and bin-related data
    ECE, MCE, ABCE = get_metrics(preds, true)
    bin_centers, _, bin_accs, _, _ = calc_bins(preds, true)
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(
        2,
        2,
        width_ratios=[4, 1],
        height_ratios=[1, 4],
        wspace=0.00,
        hspace=0.00,
    )

    # Bottom-left: reliability diagram
    ax_main = fig.add_subplot(gs[1, 0])
    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, 1)
    ax_main.set_xlabel('Confidence', size=20)
    
    ax_main.tick_params(axis='both',labelsize=20)
    
    ax_main.set_ylabel('Accuracy',size=20)
    ax_main.grid(color='gray', linestyle='dashed')
    ax_main.set_axisbelow(True)

    # Plot the reliability bars
    ax_main.bar(bin_centers, bin_centers, width=0.2, alpha=0.3, edgecolor='red', color='r', hatch='\\',label=f'Calibration Error')
    ax_main.bar(bin_centers, bin_accs,    width=0.2, alpha=1,   edgecolor='black', color='b')
    difference = bin_accs - bin_centers
    difference = np.where(difference > 0, difference, 0)
    ax_main.bar(bin_centers, difference, bottom=bin_centers, width=0.2, alpha=0.3, edgecolor='red', color='r', hatch='\\')
    ax_main.plot([0, 1], [0, 1], '--', color='gray', linewidth=2)
    ax_main.set_aspect('equal', adjustable='box')
    fig.suptitle(name,x=.5,size=30)
    ECE_patch = ax_main.plot([], [], ' ',label=f'ECE = {ECE*100:.2f}%\nMCE = {MCE*100:.2f}%\nCECE = {ABCE*100:.2f}%')
    

    ax_main.legend(fontsize=20)
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_top.set_xlim(0, 1)
    
    ax_top.hist(preds,bins=[0,.2,.4,.6,.8,1], range=(0, 1), color='gray', fill=True,alpha=0.7, histtype=u'stepfilled')
    
    ax_top.set_yticks([])
    ax_top.axis('off')
    
    
    plt.tight_layout()
    name = name.replace(" ","_")
    plt.savefig(f'./results/{name}_metrics.png', bbox_inches='tight')

def draw_reliability_graph_nb(preds, true, name):
    # Calculate metrics and bin-related data
    ECE, MCE, ABCE = get_metrics_nb(preds, true)
    bin_centers, _, bin_accs, _, _ = calc_bins_nb(preds, true)
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(
        2,
        2,
        width_ratios=[4, 1],
        height_ratios=[1, 4],
        wspace=0.00,
        hspace=0.00,
    )

    # Bottom-left: reliability diagram
    ax_main = fig.add_subplot(gs[1, 0])
    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, 1)
    ax_main.set_xlabel('Confidence', size=20)
    
    ax_main.tick_params(axis='both',labelsize=20)
    
    ax_main.set_ylabel('Accuracy',size=20)
    ax_main.grid(color='gray', linestyle='dashed')
    ax_main.set_axisbelow(True)

    # Plot the reliability bars
    ax_main.bar(bin_centers, bin_centers, width=0.1, alpha=0.3, edgecolor='red', color='r', hatch='\\',label=f'Calibration Error')
    ax_main.bar(bin_centers, bin_accs,    width=0.1, alpha=1,   edgecolor='black', color='b')
    difference = bin_accs - bin_centers
    difference = np.where(difference > 0, difference, 0)
    ax_main.bar(bin_centers, difference, bottom=bin_centers, width=0.1, alpha=0.3, edgecolor='red', color='r', hatch='\\')
    ax_main.plot([0, 1], [0, 1], '--', color='gray', linewidth=2)
    ax_main.set_aspect('equal', adjustable='box')
    fig.suptitle(name,x=.4,size=30)
    ECE_patch = ax_main.plot([], [], ' ',label=f'ECE = {ECE*100:.2f}%\nMCE = {MCE*100:.2f}%\nCECE = {ABCE*100:.2f}%')
    

    ax_main.legend(fontsize=20)
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_top.set_xlim(0, 1)
    
    ax_top.hist(preds,bins=[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1], range=(0, 1), color='gray', fill=True,alpha=0.7, histtype=u'stepfilled')
    
    ax_top.set_yticks([])
    ax_top.axis('off')
    
    
    plt.tight_layout()
    name = name.replace(" ","_")
    plt.savefig(f'./results/{name}_metrics.png', bbox_inches='tight')

def confidence(preds,true):
    #print("preds:",preds,"true:",true)
    #print(preds)
    return preds.sum()/len(preds)

def bin_acc(true):
    true = np.where(true>.5,1,0)
    preds = np.ones(len(true))
    return np.mean(preds == true)
def not_bind_bin_acc(true):
    true = np.where(true<=.5,1,0)
    preds = np.ones(len(true))
    return np.mean(preds == true)
def accuracy(preds,true):
    
    true = np.where(true>.5,1,0)
    preds_binary = np.where(preds>.5,1,0)
    return (preds_binary == true).sum() / len(true)

def bacc(true,preds):
    true = np.where(true>.5,1,0)
    preds_binary = np.where(preds>.5,1,0)
    return balanced_accuracy_score(true,preds_binary)