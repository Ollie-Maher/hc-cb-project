# import os
# import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import scipy.io as io
# import seaborn as sns
# from scipy import stats
# from sklearn.metrics import mean_squared_error

model_color = {
    "hc-cb": "darkturquoise",
    "hc-cb abl": "pink",
    "hc": "sienna",
    "plastic-hc": "olive",
    "animal": "forestgreen",
    "plastic-hc-cb": "cyan",
}


def plot_from_df(path1, path2, save_filename=None):

    data_path_p = path1
    data_path_f = path2
    model_color = {
        "hc-cb": "darkturquoise",
        "hc-cb abl": "pink",
        "hc": "sienna",
        "plastic-hc": "olive",
        "animal": "forestgreen",
        "plastic-hc-cb": "cyan",
    }

    # data_path_p = "/home/rh19400/neuro-rl/plots/data/plastic_rew.csv"

    df_plastic = pd.read_csv(data_path_p)
    columns = [
        "Step",
        "plastic-hc",
        "plastic-hc_min",
        "plastic-hc_max",
        "plastic-hc-cb",
        "plastic-hc-cb_min",
        "plastic-hc-cb_max",
    ]
    df_plastic.columns = columns

    # for fixed ordered is swapped
    df_fixed = pd.read_csv(data_path_f)
    columns = [
        "Step",
        "hc-cb",
        "hc-cb_min",
        "hc-cb_max",
        "hc",
        "hc_min",
        "hc_max",
    ]
    df_fixed.columns = columns

    df = pd.concat([df_plastic, df_fixed], axis=1, ignore_index=False, sort=False)

    start = 0  # 4000
    ind = 8000
    df_ewma = df.ewm(alpha=0.02).mean()[
        start:ind
    ]  # .plot(x='Step', y=['hcc','drqn'])[:4000]
    df_ewma_var = df.ewm(alpha=0.02).var()[start:ind]
    X = np.linspace(0, ind, num=ind - start)

    # df_ewma = df_ewma * 100
    # df_ewma_var = df_ewma_var * 100

    fig = plt.figure(figsize=(16, 4))
    gs = plt.GridSpec(1, 3, wspace=0.2, hspace=0.1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # print(df_ewma.describe())
    ax1.plot(
        X,
        df_ewma["plastic-hc"],
        color=model_color["plastic-hc"],
        alpha=1,
        label="plastic-hc",
    )
    # ax1.fill_between(
    #     X,
    #     df_ewma["plastic-hc"] - df_ewma_var["plastic-hc"].sem(),
    #     df_ewma["plastic-hc"] + df_ewma_var["plastic-hc"].sem(),
    #     color=model_color["plastic-hc"],
    #     alpha=0.6,
    # )

    ax1.plot(
        X,
        df_ewma["plastic-hc-cb"],
        color=model_color["plastic-hc-cb"],
        alpha=0.8,
        label="plastic-hc-cb",
    )
    # ax1.fill_between(
    #     X,
    #     df_ewma["plastic-hc-cb"] - df_ewma_var["plastic-hc-cb"].sem(),
    #     df_ewma["plastic-hc-cb"] + df_ewma_var["plastic-hc-cb"].sem(),
    #     color=model_color["plastic-hc-cb"],
    #     alpha=0.6,
    # )

    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set(title="Episode steps", xlabel="Trials", ylabel="Steps")
    ax1.legend()

    ax2.plot(X, df_ewma["hc"], color=model_color["hc"], alpha=1, label="hc")
    # ax2.fill_between(
    #     X,
    #     df_ewma["hc"] - df_ewma_var["hc"].sem(),
    #     df_ewma["hc"] + df_ewma_var["hc"].sem(),
    #     color=model_color["hc"],
    #     alpha=0.6,
    # )

    ax2.plot(X, df_ewma["hc-cb"], color=model_color["hc-cb"], alpha=0.8, label="hc-cb")
    # ax2.fill_between(
    #     X,
    #     df_ewma["hc-cb"] - df_ewma_var["hc-cb"].sem(),
    #     df_ewma["hc-cb"] + df_ewma_var["hc-cb"].sem(),
    #     color=model_color["hc-cb"],
    #     alpha=0.6,
    # )

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set(title="Episode steps", xlabel="Trials", ylabel="Steps")
    ax2.legend()

    width = 0.3

    ax3.bar(
        ["plastic-hc-cb"],
        df_ewma["plastic-hc-cb"].mean(),
        yerr=df_ewma["plastic-hc-cb"].sem(),
        # yerr=df_ewma_var["plastic-hc-cb"].sem(),
        width=width,
        color=model_color["plastic-hc-cb"],
    )
    ax3.bar(
        ["plastic-hc"],
        df_ewma["plastic-hc"].mean(),
        yerr=df_ewma["plastic-hc"].sem(),
        width=width,
        color=model_color["plastic-hc"],
    )
    ax3.bar(
        ["hc-cb"],
        df_ewma["hc-cb"].mean(),
        yerr=df_ewma["hc-cb"].sem(),
        width=width,
        color=model_color["hc-cb"],
    )
    ax3.bar(
        ["hc"],
        df_ewma["hc"].mean(),
        yerr=df_ewma["hc"].sem(),
        # yerr=df_ewma_var["hc"].sem(),
        width=width,
        color=model_color["hc"],
    )

    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.set(title="Average", ylabel="Steps")
    # ax3.set_ylim([0, 100])
    ax3.tick_params(axis="x", labelrotation=30)
    # ax3.hlines(y=50, xmin=-0.2, xmax=4.5, linewidth=2, color="black", linestyle="--")

    plt.tight_layout()
    plt.show()
    fig.savefig(f"{save_filename}.pdf", bbox_inches="tight")


data_path_p_steps = "/home/rh19400/neuro-rl/plots/data/plastic_steps.csv"
data_path_f_steps = "/home/rh19400/neuro-rl/plots/data/fixed_steps.csv"
plot_from_df(data_path_p_steps, data_path_f_steps, "steps")
