from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterable, List, Optional, Tuple
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


def window_stats(y, window_size):
    y = np.asarray(y)
    n = (len(y) // window_size) * window_size
    y2 = y[:n].reshape(-1, window_size)
    return y2.mean(axis=1), y2.min(axis=1), y2.max(axis=1)


def plot_powertrace_decrease_eps():
    params_file = "experimental-setups/RQ1-google-decrease-eps.json"
    with open(params_file, "r") as fp:
        params_vec = json.load(fp)
    quantile_numbers = np.arange(0.05,1.05,0.05)
    quantile_values = []
    labels = []
    for i in range(len(params_vec)):
        params = params_vec[i]
        datafile = params.get("data_results")
        eps = params.get("eps")
        df = pd.read_csv(datafile)
        quant_to_plot = []
        for quant in quantile_numbers:
            quant_to_plot.append(float(df.dropna().active_monitors_normalized.quantile(quant)))
        quantile_values.append(quant_to_plot)
        labels.append(r"$\varepsilon$ = " + f"{eps:.4f}")
    

    plt.figure(figsize=(8,7))
    # choose a single color map
    cmap = plt.cm.Blues  # try Reds, Greens, Purples, etc.
    colors = cmap(np.linspace(0.3, 0.9, len(quantile_values)))  # light → dark

    for (y, lbl, color) in zip(quantile_values, labels, colors):
        plt.plot(
            quantile_numbers,
            y,
            marker='o',
            color=color,
            label=lbl
        )
    plt.xlabel('Quantile', fontsize=22)
    plt.ylabel('Normalized active registers', fontsize=22)
    plt.legend(fontsize=22)
    plt.grid(True)
    plt.ylim(0.0, 1.0)

    outpath = "experimental-results/images/powertrace_decrease_eps.pdf"
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    print(f"Saved plot to {outpath}")



def plot_powertrace_boxplots_eps():
    params_file = "experimental-setups/RQ1-google-decrease-eps.json"
    with open(params_file, "r") as fp:
        params_vec = json.load(fp)

    quantile_numbers = np.arange(0.05, 1.05, 0.05)
    boxplot_data = []
    labels = []

    for params in params_vec:
        datafile = params.get("data_results")
        eps = params.get("eps")

        df = pd.read_csv(datafile)

        # collect quantiles as a "distribution"
        quantiles = [
            float(df.dropna().active_monitors_normalized.quantile(q))
            for q in quantile_numbers
        ]

        boxplot_data.append(quantiles)
        # labels.append(f"eps = {eps:.4f}")
        labels.append(" ")

    plt.figure(figsize=(8,7))

    cmap = plt.cm.Blues
    colors = cmap(np.linspace(0.3, 0.9, len(boxplot_data)))

    bp = plt.boxplot(
    boxplot_data,
    tick_labels=labels,
    showfliers=False,
    patch_artist=True
    )

    for box, color in zip(bp['boxes'], colors):
        box.set_facecolor(color)

    for median in bp['medians']:
        median.set_color('black')

    # plt.boxplot(
    #     boxplot_data,
    #     tick_labels=labels,
    #     showfliers=False,     # cleaner when quantiles are dense
    #     patch_artist=True
    # )

    # plt.ylabel("Normalized active monitors")
    plt.xlabel(r"$\varepsilon$",fontsize=22)
    plt.ylim(0.0, 1.0)
    plt.grid(axis="y")

    plt.xticks(rotation=30, ha="right")

    outpath = "experimental-results/images/powertrace_decrease_eps_boxplots.pdf"
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    print(f"Saved plot to {outpath}")


def plot_powertrace_decrease_interval_length():
    params_file = "experimental-setups/RQ1-google-decrease-target-interval.json"
    with open(params_file, "r") as fp:
        params_vec = json.load(fp)
    quantile_numbers = np.arange(0.05,1.05,0.05)
    quantile_values = []
    labels = []
    for i in range(len(params_vec)):
        params = params_vec[i]
        datafile = params.get("data_results")
        eps = params.get("eps")
        L = params.get("L")
        U = params.get("U")
        df = pd.read_csv(datafile)
        quant_to_plot = []
        for quant in quantile_numbers:
            quant_to_plot.append(float(df.dropna().active_monitors_normalized.quantile(quant)))
        quantile_values.append(quant_to_plot)
        labels.append(r"$|\mathcal{I}| = $" + f"{U-L:.4f}")
    

    plt.figure(figsize=(8,7))
    # choose a single color map
    cmap = plt.cm.Blues  # try Reds, Greens, Purples, etc.
    colors = cmap(np.linspace(0.3, 0.9, len(quantile_values)))  # light → dark

    for (y, lbl, color) in zip(quantile_values, labels, colors):
        plt.plot(
            quantile_numbers,
            y,
            marker='o',
            color=color,
            label=lbl
        )
    plt.xlabel('Quantile', fontsize=22)
    plt.ylabel('Normalized active monitors', fontsize=22)
    plt.legend(fontsize=22)
    plt.grid(True)
    plt.ylim(0.0, 1.0)

    outpath = "experimental-results/images/powertrace_decrease_target_length.pdf"
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    print(f"Saved plot to {outpath}")    


def plot_powertrace_boxplots_interval_length():
    params_file = "experimental-setups/RQ1-google-decrease-target-interval.json"
    with open(params_file, "r") as fp:
        params_vec = json.load(fp)

    quantile_numbers = np.arange(0.05, 1.05, 0.05)
    boxplot_data = []
    labels = []

    for params in params_vec:
        datafile = params.get("data_results")
        L = params.get("L")
        U = params.get("U")

        df = pd.read_csv(datafile)

        # collect quantiles as a "distribution"
        quantiles = [
            float(df.dropna().active_monitors_normalized.quantile(q))
            for q in quantile_numbers
        ]

        boxplot_data.append(quantiles)
        labels.append(f" ")

    plt.figure(figsize=(8,7))

    cmap = plt.cm.Blues
    colors = cmap(np.linspace(0.3, 0.9, len(boxplot_data)))

    bp = plt.boxplot(
    boxplot_data,
    tick_labels=labels,
    showfliers=False,
    patch_artist=True
    )

    for box, color in zip(bp['boxes'], colors):
        box.set_facecolor(color)

    for median in bp['medians']:
        median.set_color('black')

    plt.xlabel("Interval length",fontsize=22)
    plt.ylim(0.0, 1.0)
    plt.grid(axis="y")

    plt.xticks(rotation=30, ha="right")

    outpath = "experimental-results/images/powertrace_decrease_target_length_boxplots.pdf"
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    print(f"Saved plot to {outpath}")


def plot_mnist_quantiles():
    params_file = "experimental-setups/RQ1-mnist-traces-increase-noise.json"
    with open(params_file, "r") as fp:
        params_vec = json.load(fp)
    quantile_numbers = np.arange(0.05,1.05,0.05)
    quantile_values = []
    labels = []
    for i in range(len(params_vec)):
        params = params_vec[i]
        datafile = params.get("data_results")
        eps = params.get("eps")
        aux = params.get("data_source").split("_")[-1].split(".")
        noise = aux[0] + "." + aux[1]
        df = pd.read_csv(datafile)
        quant_to_plot = []
        for quant in quantile_numbers:
            quant_to_plot.append(float(df.dropna().active_monitors_normalized.quantile(quant)))
        quantile_values.append(quant_to_plot)
        labels.append(r"$\mathtt{max\_noise} =  $" + f"{noise}")
    

    plt.figure(figsize=(8,6))
    # choose a single color map
    cmap = plt.cm.Blues  # try Reds, Greens, Purples, etc.
    colors = cmap(np.linspace(0.3, 0.9, len(quantile_values)))  # light → dark

    for (y, lbl, color) in zip(quantile_values, labels, colors):
        plt.plot(
            quantile_numbers,
            y,
            marker='o',
            color=color,
            label=lbl
        )
    plt.xlabel('Quantile',fontsize=18)
    plt.ylabel('Normalized active registers', fontsize=18)
    # plt.legend( fontsize = 18)
    plt.grid(True)
    plt.ylim(0.0, 1.0)

    outpath = "experimental-results/images/mnist_increase_noise.pdf"
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    print(f"Saved plot to {outpath}")


def plot_mnist_execution_active_monitors():
    params_file = "experimental-setups/RQ1-mnist-traces-increase-noise.json"
    with open(params_file, "r") as fp:
        params_vec = json.load(fp)
    
    active_monitor_values = []

    labels = []
    for i in range(len(params_vec)):
        params = params_vec[i]
        datafile = params.get("data_results")
        eps = params.get("eps")
        aux = params.get("data_source").split("_")[-1].split(".")
        noise = aux[0] + "." + aux[1]
        df = pd.read_csv(datafile)
        active_monitor_values.append(df.active_monitors_normalized.values)
        labels.append(f"max_noise = {noise}")

    window_size = 50

    plt.figure(figsize=(8,6))

    cmap = plt.cm.Blues
    colors = cmap(np.linspace(0.3, 0.9, len(active_monitor_values)))

    for (y, lbl, color) in zip(active_monitor_values, labels, colors):
        m, lo, hi = window_stats(y, window_size)
        x = np.arange(len(m)) * window_size
        plt.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0)  # light band
        plt.plot(x, m, color=color, label=lbl)

    plt.xlabel('Time', fontsize=18)
    plt.ylabel('Normalized active registers', fontsize=18)
    # plt.legend()
    plt.grid(True)

    outpath = 'experimental-results/images/mnst_active_monitors_execution.pdf'
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    print(f"Saved plot to {outpath}")


def plot_mnist_execution_values():
    params_file = "experimental-setups/RQ1-mnist-traces-increase-noise.json"
    with open(params_file, "r") as fp:
        params_vec = json.load(fp)
    
    active_monitor_values = []

    labels = []
    for i in range(len(params_vec)):
        params = params_vec[i]
        datafile = params.get("data_results")
        eps = params.get("eps")
        aux = params.get("data_source").split("_")[-1].split(".")
        noise = aux[0] + "." + aux[1]
        df = pd.read_csv(datafile)
        df["actual_vals"] = df["valL"] # valL is actual val, because infR=0
        active_monitor_values.append(df.actual_vals.values)
        # labels.append(f"max_noise = {noise}")
        labels.append(r"$\mathtt{max\_noise} =  $" + f"{noise}")

    window_size = 50

    plt.figure(figsize=(8,6))

    cmap = plt.cm.Blues
    colors = cmap(np.linspace(0.3, 0.9, len(active_monitor_values)))

    for (y, lbl, color) in zip(active_monitor_values, labels, colors):
        m, lo, hi = window_stats(y, window_size)
        x = np.arange(len(m)) * window_size
        plt.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0)  # light band
        plt.plot(x, m, color=color, label=lbl)

    plt.axhline(0.96, color = "green", linestyle = "-.", label="target interval")
    plt.axhline(1.0, color = "green", linestyle = "-.")

    plt.xlabel('Time', fontsize=18)
    plt.ylabel('Accuracy', fontsize=18)
    plt.legend(fontsize=18)
    plt.grid(True)

    outpath = 'experimental-results/images/mnst_accuracy_values_execution.pdf'
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    print(f"Saved plot to {outpath}")


def plot_adult_decrease_eps():
    
    params_file = "experimental-setups/ffb-adult-race-decrease-eps.json"
    with open(params_file, "r") as fp:
        params_vec = json.load(fp)
    quantile_numbers = np.arange(0.05,1.05,0.05)
    quantile_values_sync = []
    quantile_values_async = []
    labels_sync = []
    labels_async = []
    for i in range(len(params_vec)):
        params = params_vec[i]
        
        datafile = params.get("data_results")
        eps = params.get("eps")
        df = pd.read_csv(datafile)
        quant_to_plot = []
        for quant in quantile_numbers:
            quant_to_plot.append(float(df.dropna().active_monitors.quantile(quant)))
        if params["synchronous"]:
            quantile_values_sync.append(quant_to_plot)
            labels_sync.append(r"$\varepsilon = $" + f"{eps:.4f},  sync")
        else:
            quantile_values_async.append(quant_to_plot)
            labels_async.append(r"$\varepsilon = $" + f"{eps:.4f},  async")

    

    plt.figure(figsize=(10,6))
    # choose a single color map
    cmap_sync = plt.cm.Blues  # try Reds, Greens, Purples, etc.
    colors_sync = cmap_sync(np.linspace(0.3, 0.9, len(quantile_values_sync)))  # light → dark
    cmap_async = plt.cm.Greens  # try Reds, Greens, Purples, etc.
    colors_async = cmap_async(np.linspace(0.3, 0.9, len(quantile_values_async)))  # light → dark

    for (y, lbl, color) in zip(quantile_values_sync, labels_sync, colors_sync):
        plt.plot(
            quantile_numbers,
            y,
            marker='o',
            color=color,
            label=lbl
        )
    for (y, lbl, color) in zip(quantile_values_async, labels_async, colors_async):
        plt.plot(
            quantile_numbers,
            y,
            marker='o',
            color=color,
            label=lbl
        )
    plt.xlabel('Quantile',fontsize=18)
    plt.ylabel('Number of active registers', fontsize=18)
    plt.legend(fontsize=14,ncol=2)
    plt.grid(True)
    # plt.ylim(0.0, 1.0)

    outpath = "experimental-results/images/adult_decrease_eps.pdf"
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    print(f"Saved plot to {outpath}")


def plot_adult_decrease_interval_length():
    params_file = "experimental-setups/ffb-adult-race-decrease-interval-length.json"
    with open(params_file, "r") as fp:
        params_vec = json.load(fp)
    quantile_numbers = np.arange(0.05,1.05,0.05)
    quantile_values_sync = []
    quantile_values_async = []
    labels_sync = []
    labels_async = []
    for i in range(len(params_vec)):
        params = params_vec[i]
        L = params.get("L")
        U = params.get("U")
        datafile = params.get("data_results")
        eps = params.get("eps")
        df = pd.read_csv(datafile)
        quant_to_plot = []
        for quant in quantile_numbers:
            quant_to_plot.append(float(df.dropna().active_monitors.quantile(quant)))
        if params["synchronous"]:
            quantile_values_sync.append(quant_to_plot)
            labels_sync.append(r"$|\mathcal{I}| = $" + f"{U-L:.4f}, sync")
        else:
            quantile_values_async.append(quant_to_plot)
            labels_async.append(r"$|\mathcal{I}| = $" + f"{U-L:.4f}, async")

    

    plt.figure(figsize=(10,6))
    # choose a single color map
    cmap_sync = plt.cm.Blues  # try Reds, Greens, Purples, etc.
    colors_sync = cmap_sync(np.linspace(0.3, 0.9, len(quantile_values_sync)))  # light → dark
    cmap_async = plt.cm.Greens  # try Reds, Greens, Purples, etc.
    colors_async = cmap_async(np.linspace(0.3, 0.9, len(quantile_values_async)))  # light → dark

    for (y, lbl, color) in zip(quantile_values_sync, labels_sync, colors_sync):
        plt.plot(
            quantile_numbers,
            y,
            marker='o',
            color=color,
            label=lbl
        )
    for (y, lbl, color) in zip(quantile_values_async, labels_async, colors_async):
        plt.plot(
            quantile_numbers,
            y,
            marker='o',
            color=color,
            label=lbl
        )
    plt.xlabel('Quantile',fontsize=18)
    plt.ylabel('Number of active registers',fontsize=18)
    plt.legend(fontsize=14, ncol=2)
    plt.grid(True)
    # plt.ylim(0.0, 1.0)

    outpath = "experimental-results/images/adult_decrease_interval_lenght.pdf"
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    print(f"Saved plot to {outpath}")



def plot_adult_decrease_eps_boxplot():
    params_file = "experimental-setups/ffb-adult-race-decrease-eps.json"
    with open(params_file, "r") as fp:
        params_vec = json.load(fp)
    quantile_numbers = np.arange(0.05,1.05,0.05)
    boxplot_data_sync = []
    boxplot_data_async = []

    labels_sync = []
    labels_async = []
    for i in range(len(params_vec)):
        params = params_vec[i]
        
        datafile = params.get("data_results")
        eps = params.get("eps")
        df = pd.read_csv(datafile)
        quant_to_plot = []
        for quant in quantile_numbers:
            quant_to_plot.append(float(df.dropna().active_monitors.quantile(quant)))
        if params["synchronous"]:
            boxplot_data_sync.append(quant_to_plot)
            labels_sync.append(f" ")
        else:
            boxplot_data_async.append(quant_to_plot)
            labels_async.append(f" ")    


    fig, ax = plt.subplots(figsize=(10, 6))
    n = len(labels_sync)
    x = np.arange(1, n + 1)
    offset = 0.0   # horizontal nudge so both are visible
    width  = 0.28   # box width

    bp_sync = ax.boxplot(
        boxplot_data_sync,
        positions=x - offset,
        widths=width,
        tick_labels=None,      # we set ticks manually
        showfliers=False,
        patch_artist=True
    )

    bp_async = ax.boxplot(
        boxplot_data_async,
        positions=x + offset,
        widths=width,
        tick_labels=None,
        showfliers=False,
        patch_artist=True
    )


    # Style so you can distinguish them (no specific colors required, but helpful)
    for b in bp_sync["boxes"]:
        b.set_alpha(0.6)
    for b in bp_async["boxes"]:
        b.set_alpha(0.6)

    cmap_sync = plt.cm.Blues
    colors_sync = cmap_sync(np.linspace(0.3, 0.9, len(boxplot_data_sync)))
    cmap_async = plt.cm.Greens
    colors_async = cmap_async(np.linspace(0.3, 0.9, len(boxplot_data_async)))

    for box, color in zip(bp_sync['boxes'], colors_sync):
        box.set_facecolor(color)
    for box, color in zip(bp_async['boxes'], colors_async):
        box.set_facecolor(color)

    for median in bp_sync['medians']:
        median.set_color('black')
    for median in bp_async['medians']:
        median.set_color('black')

    ax.set_xticks(x)
    ax.set_xticklabels(labels_sync, rotation=30, ha="right")

    ax.set_xlabel(r"$\varepsilon$",fontsize=18)
    ax.grid(axis="y")
    # ax.set_ylim(0.0, 1.0)
    
    
    outpath = "experimental-results/images/adult_decrease_eps_boxplot.pdf"
    plt.tight_layout()
    # plt.show()
    plt.savefig(outpath, bbox_inches="tight")
    
    print(f"Saved plot to {outpath}")


    

def plot_adult_decrease_interval_length_boxplot():
    params_file = "experimental-setups/ffb-adult-race-decrease-interval-length.json"
    with open(params_file, "r") as fp:
        params_vec = json.load(fp)
    quantile_numbers = np.arange(0.05,1.05,0.05)
    boxplot_data_sync = []
    boxplot_data_async = []

    labels_sync = []
    labels_async = []
    for i in range(len(params_vec)):
        params = params_vec[i]
        
        datafile = params.get("data_results")
        eps = params.get("eps")
        L = params.get("L")
        U = params.get("U")
        df = pd.read_csv(datafile)
        quant_to_plot = []
        for quant in quantile_numbers:
            quant_to_plot.append(float(df.dropna().active_monitors.quantile(quant)))
        if params["synchronous"]:
            boxplot_data_sync.append(quant_to_plot)
            labels_sync.append(f" ")
        else:
            boxplot_data_async.append(quant_to_plot)
            labels_async.append(f" ")    


    fig, ax = plt.subplots(figsize=(10, 6))
    n = len(labels_sync)
    x = np.arange(1, n + 1)
    offset = 0.0   # horizontal nudge so both are visible
    width  = 0.28   # box width

    bp_sync = ax.boxplot(
        boxplot_data_sync,
        positions=x - offset,
        widths=width,
        tick_labels=None,      # we set ticks manually
        showfliers=False,
        patch_artist=True
    )

    bp_async = ax.boxplot(
        boxplot_data_async,
        positions=x + offset,
        widths=width,
        tick_labels=None,
        showfliers=False,
        patch_artist=True
    )


    # Style so you can distinguish them (no specific colors required, but helpful)
    for b in bp_sync["boxes"]:
        b.set_alpha(0.6)
    for b in bp_async["boxes"]:
        b.set_alpha(0.6)

    cmap_sync = plt.cm.Blues
    colors_sync = cmap_sync(np.linspace(0.3, 0.9, len(boxplot_data_sync)))
    cmap_async = plt.cm.Greens
    colors_async = cmap_async(np.linspace(0.3, 0.9, len(boxplot_data_async)))

    for box, color in zip(bp_sync['boxes'], colors_sync):
        box.set_facecolor(color)
    for box, color in zip(bp_async['boxes'], colors_async):
        box.set_facecolor(color)

    for median in bp_sync['medians']:
        median.set_color('black')
    for median in bp_async['medians']:
        median.set_color('black')

    ax.set_xticks(x)
    ax.set_xticklabels(labels_sync, rotation=30, ha="right")

    ax.set_xlabel("Interval width", fontsize=18)
    ax.grid(axis="y")
    # ax.set_ylim(0.0, 1.0)
    
    
    outpath = "experimental-results/images/adult_decrease_interval_boxplot.pdf"
    plt.tight_layout()
    # plt.show()
    plt.savefig(outpath, bbox_inches="tight")
    
    print(f"Saved plot to {outpath}")



def plot_stochastic_decrease_interval():
    params_file = "experimental-setups/stochastic-decrease-interval-length.json"
    with open(params_file, "r") as fp:
        params_vec = json.load(fp)
    quantile_numbers = np.arange(0.05,1.05,0.05)
    boxplot_data_sync = []
    boxplot_data_async = []

    labels_sync = []
    labels_async = []

    accuracies = {}

    for i in tqdm(range(len(params_vec))):
        params = params_vec[i]
        width = params.get("data_results").split("_")[-2]
        j = params.get("data_results").split("_")[-1].split(".")[0]
        s = params.get("s")
        r = params.get("r")
        L = params.get("L")
        U = params.get("U")
        if width not in accuracies:
            accuracies[width] = []
        df = pd.read_csv(params.get("data_results"))
        orig_df = pd.read_csv(f"stochastic_traces/sin1000_{j}.csv")
        df["p"] = orig_df["p"]
        df["discounted_p"] = orig_df["discounted_p"]
        df["ground_truth_verdict"] = ((df["p"] >= L) & (df["p"] <= U)).astype(int)
        df_nonan = df.dropna()
        accuracy=(df_nonan.verdict == df_nonan.ground_truth_verdict).sum()/len(df_nonan)
        accuracies[width].append(float(accuracy))
    # print(accuracies)
    

    # Sort keys numerically
    xs = sorted(accuracies.keys(), key=float)
    data = [accuracies[k] for k in xs]

    plt.figure(figsize=(8, 4))

    plt.boxplot(
        data,
        positions=np.arange(len(xs)),
        widths=0.6,
        showfliers=True
    )

    plt.xticks(
        ticks=np.arange(len(xs)),
        labels=xs,
        rotation=45,
        ha="right"
    )

    plt.xlabel("key")
    plt.ylabel("accuracy")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()
         

    


def plot_stochastic_decrease_interval_rand():
    params_file = "experimental-setups/stochastic-decrease-interval-length_rand.json"
    with open(params_file, "r") as fp:
        params_vec = json.load(fp)
    quantile_numbers = np.arange(0.05,1.05,0.05)
    boxplot_data_sync = []
    boxplot_data_async = []

    labels_sync = []
    labels_async = []

    accuracies = {}

    for i in tqdm(range(len(params_vec))):
        params = params_vec[i]
        width = params.get("data_results").split("_")[-2]
        j = params.get("data_results").split("_")[-1].split(".")[0]
        s = params.get("s")
        r = params.get("r")
        L = params.get("L")
        U = params.get("U")
        if width not in accuracies:
            accuracies[width] = []
        df = pd.read_csv(params.get("data_results"))
        orig_df = pd.read_csv(f"stochastic_traces/sin1000_{j}.csv")
        df["p"] = orig_df["p"]
        df["discounted_p"] = orig_df["discounted_p"]
        df["ground_truth_verdict"] = ((df["p"] >= L) & (df["p"] <= U)).astype(int)
        df_nonan = df.dropna()
        accuracy=(df_nonan.verdict == df_nonan.ground_truth_verdict).sum()/len(df_nonan)
        accuracies[width].append(float(accuracy))
    # print(accuracies)
    

    # Sort keys numerically
    xs = sorted(accuracies.keys(), key=float)
    data = [accuracies[k] for k in xs]

    print(accuracies)

    # plt.figure(figsize=(8, 4))

    # plt.boxplot(
    #     data,
    #     positions=np.arange(len(xs)),
    #     widths=0.6,
    #     showfliers=True
    # )

    # plt.xticks(
    #     ticks=np.arange(len(xs)),
    #     labels=xs,
    #     rotation=45,
    #     ha="right"
    # )

    # plt.xlabel("key")
    # plt.ylabel("accuracy")
    # plt.grid(True, axis="y", alpha=0.3)
    # plt.tight_layout()
    # plt.show()


def table_stochastic_decrease_interval():
    params_file = "experimental-setups/stochastic-decrease-interval-length_rand.json"
    with open(params_file, "r") as fp:
        params_vec = json.load(fp)
    quantile_numbers = np.arange(0.05,1.05,0.05)
    boxplot_data_sync = []
    boxplot_data_async = []

    labels_sync = []
    labels_async = []

    accuracies_rand = {}
    accuracies_rand = {}


    for i in tqdm(range(len(params_vec))):
        params = params_vec[i]
        width = params.get("data_results").split("_")[-2]
        j = params.get("data_results").split("_")[-1].split(".")[0]
        s = params.get("s")
        r = params.get("r")
        L = params.get("L")
        U = params.get("U")
        if width not in accuracies_rand:
            accuracies_rand[width] = []
        if os.path.exists(params.get("data_results")):
            df = pd.read_csv(params.get("data_results"))
            orig_df = pd.read_csv(f"stochastic_traces/sin1000_{j}.csv")
            df["p"] = orig_df["p"]
            df["discounted_p"] = orig_df["discounted_p"]
            df["ground_truth_verdict"] = ((df["p"] >= L) & (df["p"] <= U)).astype(int)
            df_nonan = df.dropna()
            accuracy=(df_nonan.verdict == df_nonan.ground_truth_verdict).sum()/len(df_nonan)
            accuracies_rand[width].append(float(accuracy))
         

    params_file = "experimental-setups/stochastic-decrease-interval-length.json"
    with open(params_file, "r") as fp:
        params_vec = json.load(fp)
    quantile_numbers = np.arange(0.05,1.05,0.05)
    boxplot_data_sync = []
    boxplot_data_async = []

    labels_sync = []
    labels_async = []

    accuracies_sin = {}

    for i in tqdm(range(len(params_vec))):
        params = params_vec[i]
        width = params.get("data_results").split("_")[-2]
        j = params.get("data_results").split("_")[-1].split(".")[0]
        s = params.get("s")
        r = params.get("r")
        L = params.get("L")
        U = params.get("U")
        if width not in accuracies_sin:
            accuracies_sin[width] = []
        if os.path.exists(params.get("data_results")):
            df = pd.read_csv(params.get("data_results"))
            orig_df = pd.read_csv(f"stochastic_traces/sin1000_{j}.csv")
            df["p"] = orig_df["p"]
            df["discounted_p"] = orig_df["discounted_p"]
            df["ground_truth_verdict"] = ((df["p"] >= L) & (df["p"] <= U)).astype(int)
            df_nonan = df.dropna()
            accuracy=(df_nonan.verdict == df_nonan.ground_truth_verdict).sum()/len(df_nonan)
            accuracies_sin[width].append(float(accuracy))
        

    mean_sin = {k: np.nanmean(v) for k, v in accuracies_sin.items()}
    mean_rand = {k: np.nanmean(v) for k, v in accuracies_rand.items()}

    # build dataframe
    df = pd.DataFrame([mean_sin, mean_rand], index=["sin", "rand"])

    # drop columns where both rows are NaN
    df = df.dropna(axis=1, how="all")

    # optional: sort columns numerically
    df = df[sorted(df.columns, key=float)]

    # export to LaTeX
    latex_table = df.to_latex(float_format="%.4f")

    print(latex_table)



def main():
    # plot_powertrace_decrease_eps()
    # plot_powertrace_boxplots_eps()
    # plot_powertrace_decrease_interval_length()
    # plot_powertrace_boxplots_interval_length()
    # plot_mnist_quantiles()
    # plot_mnist_execution_active_monitors()
    # plot_mnist_execution_values()
    # plot_adult_decrease_eps()
    # plot_adult_decrease_eps_boxplot()
    # plot_adult_decrease_interval_length()
    # plot_adult_decrease_interval_length_boxplot()
    # plot_stochastic_decrease_interval()
    # plot_stochastic_decrease_interval_rand()
    table_stochastic_decrease_interval()
    

if __name__ == "__main__":
    main()