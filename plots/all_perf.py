from dataclasses import fields

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catppuccin import Flavour

# model_color = {
#     "plastic-hc": "olive",
#     "plastic-hc-cb": "cyan",
#     # "plastic-hc-cb abl": "cyan",
#     "hc": "sienna",
#     "hc-cb": "darkturquoise",
#     # "hc-cb abl": "pink",
#     # "animal": "forestgreen",
#     "bio-plastic-hc-cb": "pink",
#     # "bio-plastic-hc-cb abl": "cyan",
#     "bio-hc-cb": "forestgreen",
#     # "bio-hc-cb abl": "pink",
# }
names = [
    "hc",
    "hc-cb",
    "plastic-hc",
    "plastic-hc-cb",
    "bio-plastic-hc-cb",
    "bio-hc-cb",
    "hc-cb abl",
    "plastic-hc-cb abl",
    "bio-plastic-hc-cb abl",
    "bio-hc-cb abl",
]
model_color = {}
flavour = Flavour.frappe()
for field, name in zip(fields(flavour), names):
    colour = getattr(flavour, field.name)
    print(f"{name} - {field.name}: #{colour.hex}")
    model_color[name] = f"#{colour.hex}"


def update_cols_name(path, model):
    df = pd.read_csv(path)
    df.columns = [model + "-" + col for col in df.columns]
    columns = [
        "Step",
        model,
        model + "_min",
        model + "_max",
    ]
    df.columns = columns
    return df


def plot_from_df(path1, path2, path3, path4, save_filename=None):

    data_path_p = path1
    data_path_f = path2
    data_path_bio_p = path3
    data_path_bio_f = path4

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

    df_bio_plastic = update_cols_name(data_path_bio_p, "bio-plastic-hc-cb")
    df_bio_fixed = update_cols_name(data_path_bio_f, "bio-hc-cb")

    df = pd.concat(
        [df_plastic, df_fixed, df_bio_plastic, df_bio_fixed],
        axis=1,
        ignore_index=False,
        sort=False,
    )

    ind = 2000
    df_ewma = df.ewm(alpha=0.02).mean()[
        :ind
    ]  # .plot(x='Step', y=['hcc','drqn'])[:4000]
    df_ewma_var = df.ewm(alpha=0.02).var()[:ind]
    X = np.linspace(0, ind, num=ind)

    df_ewma = df_ewma * 100
    df_ewma_var = df_ewma_var * 100

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
    ax1.fill_between(
        X,
        df_ewma["plastic-hc"] - df_ewma_var["plastic-hc"],
        df_ewma["plastic-hc"] + df_ewma_var["plastic-hc"],
        color=model_color["plastic-hc"],
        alpha=0.5,
    )

    ax1.plot(
        X,
        df_ewma["plastic-hc-cb"],
        color=model_color["plastic-hc-cb"],
        alpha=1,
        label="plastic-hc-cb",
    )
    ax1.fill_between(
        X,
        df_ewma["plastic-hc-cb"] - df_ewma_var["plastic-hc-cb"],
        df_ewma["plastic-hc-cb"] + df_ewma_var["plastic-hc-cb"],
        color=model_color["plastic-hc-cb"],
        alpha=0.5,
    )
    ax1.plot(
        X,
        df_ewma["bio-plastic-hc-cb"],
        color=model_color["bio-plastic-hc-cb"],
        alpha=1,
        label="bio-plastic-hc-cb",
    )
    ax1.fill_between(
        X,
        df_ewma["bio-plastic-hc-cb"] - df_ewma_var["bio-plastic-hc-cb"],
        df_ewma["bio-plastic-hc-cb"] + df_ewma_var["bio-plastic-hc-cb"],
        color=model_color["bio-plastic-hc-cb"],
        alpha=0.5,
    )

    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set(title="Episode reward", xlabel="Trials", ylabel="Performance (%)")
    ax1.legend()

    ax2.plot(X, df_ewma["hc"], color=model_color["hc"], alpha=1, label="hc")
    ax2.fill_between(
        X,
        df_ewma["hc"] - df_ewma_var["hc"],
        df_ewma["hc"] + df_ewma_var["hc"],
        color=model_color["hc"],
        alpha=0.5,
    )

    ax2.plot(X, df_ewma["hc-cb"], color=model_color["hc-cb"], alpha=1, label="hc-cb")
    ax2.fill_between(
        X,
        df_ewma["hc-cb"] - df_ewma_var["hc-cb"],
        df_ewma["hc-cb"] + df_ewma_var["hc-cb"],
        color=model_color["hc-cb"],
        alpha=0.5,
    )
    ax2.plot(
        X,
        df_ewma["bio-hc-cb"],
        color=model_color["bio-hc-cb"],
        alpha=1,
        label="bio-hc-cb",
    )
    ax2.fill_between(
        X,
        df_ewma["bio-hc-cb"] - df_ewma_var["bio-hc-cb"],
        df_ewma["bio-hc-cb"] + df_ewma_var["bio-hc-cb"],
        color=model_color["bio-hc-cb"],
        alpha=0.5,
    )

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set(title="Episode reward", xlabel="Trials", ylabel="Performance (%)")
    ax2.legend()

    width = 0.3

    ax3.bar(
        ["plastic-hc-cb"],
        df_ewma["plastic-hc-cb"].mean(),
        yerr=df_ewma_var["plastic-hc-cb"].mean(),
        width=width,
        color=model_color["plastic-hc-cb"],
    )
    ax3.bar(
        ["plastic-hc"],
        df_ewma["plastic-hc"].mean(),
        yerr=df_ewma_var["plastic-hc"].mean(),
        width=width,
        color=model_color["plastic-hc"],
    )
    ax3.bar(
        ["hc-cb"],
        df_ewma["hc-cb"].mean(),
        yerr=df_ewma_var["hc-cb"].mean(),
        width=width,
        color=model_color["hc-cb"],
    )
    ax3.bar(
        ["hc"],
        df_ewma["hc"].mean(),
        yerr=df_ewma_var["hc"].mean(),
        width=width,
        color=model_color["hc"],
    )
    ax3.bar(
        ["bio-plastic-hc-cb"],
        df_ewma["bio-plastic-hc-cb"].mean(),
        yerr=df_ewma_var["bio-plastic-hc-cb"].mean(),
        width=width,
        color=model_color["bio-plastic-hc-cb"],
    )
    ax3.bar(
        ["bio-hc-cb"],
        df_ewma["bio-hc-cb"].mean(),
        yerr=df_ewma_var["bio-hc-cb"].mean(),
        width=width,
        color=model_color["bio-hc-cb"],
    )

    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.set(title="Test", ylabel="Performance (%)")
    ax3.set_ylim([0, 100])
    ax3.tick_params(axis="x", labelrotation=30)
    ax3.hlines(y=50, xmin=-0.2, xmax=5.2, linewidth=2, color="black", linestyle="--")

    plt.tight_layout()
    plt.show()
    fig.savefig(f"{save_filename}.pdf", bbox_inches="tight")


data_path_p = "/home/rh19400/neuro-rl/plots/data/plastic_rew.csv"
data_path_f = "/home/rh19400/neuro-rl/plots/data/fixed_rew.csv"
data_path_bio_p = "/home/rh19400/neuro-rl/plots/data/bio_plastic_rew.csv"
data_path_bio_f = "/home/rh19400/neuro-rl/plots/data/bio_rew.csv"
plot_from_df(data_path_p, data_path_f, data_path_bio_p, data_path_bio_f, "all_reward")
