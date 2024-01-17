from dataclasses import fields

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catppuccin import Flavour
from numpy.random import rand, randint, randn

#### Bio model ####
# plastic hc-cb ##
data_path_bio_hc_cb_plastic = "/home/rh19400/neuro-rl/exp_local/2023.12.02/200235_hccbio_agent=hccbio,experiment=wm_plastic,save_stats=true,seed=1,use_wandb=false/locations"
# plastic hc-cb-abl ##
data_path_bio_hc_cb_plastic_abl = "/home/rh19400/neuro-rl/exp_local/2023.12.02/201451_hccbio_agent=hccbio,experiment=wm_plastic,save_stats=true,seed=1,use_wandb=false/locations"
# fixed hc-cb ##
data_path_bio_hc_cb = "/home/rh19400/neuro-rl/exp_local/2023.12.11/110514_hccbio_agent=hccbio,experiment=wm,save_stats=true,seed=1/locations"
# fixed hc-cb-abl ##
data_path_bio_hc_cb_abl = "/home/rh19400/neuro-rl/exp_local/2023.12.11/111353_hccbio_agent=hccbio,experiment=wm,save_stats=true,seed=1/locations"

#### Fixed model ####
# hc fixed 10k trained steps
data_path_hc = "/home/rh19400/neuro-rl/exp_local/2023.11.28/091550_drqn_agent=drqn,experiment=wm_fixed_s,save_stats=true,seed=5,use_wandb=false/locations"
# hc-cb fixed 10k trained steps
data_path_hc_cb = "/home/rh19400/neuro-rl/exp_local/2023.11.28/091619_hcc_agent=hcc,experiment=wm_s,save_stats=true,seed=1,use_wandb=false/locations"
# hc_cb ablation, 10k trained steps
data_path_hc_cb_abl = "/home/rh19400/neuro-rl/exp_local/2023.11.28/104213_hcc_agent=hcc,experiment=wm_s,save_stats=true,seed=1,use_wandb=false/locations"

### Plastic model ###
# drqn plastic
data_path_hc_plastic = "/home/rh19400/neuro-rl/exp_local/2023.12.11/115628_drqn_agent=drqn,experiment=wm_plastic,save_stats=true,seed=10/locations"
# hc-cb plastic 10k trained steps
data_path_hc_cb_plastic = "/home/rh19400/neuro-rl/exp_local/2023.11.28/182923_hcc_agent=hcc,experiment=wm_plastic,save_stats=true,seed=15,use_wandb=false/locations"
# hc-cb plastic ablation, 10k trained steps
data_path_hc_cb_plastic_abl = "/home/rh19400/neuro-rl/exp_local/2023.11.28/183532_hcc_agent=hcc,experiment=wm_plastic,save_stats=true,seed=15,use_wandb=false/locations"

# # # drqn plastic
# data_path_drqn_2 = "/home/rh19400/neuro-rl/exp_local/2023.11.27/163557_drqn_agent=drqn,experiment=wm_plastic_s2,save_stats=true,seed=1,use_wandb=false/locations"
# data_path_drqn_1 = "/home/rh19400/neuro-rl/exp_local/2023.11.27/163047_drqn_agent=drqn,experiment=wm_plastic_s,save_stats=true,seed=1,use_wandb=false/locations"
# # # hcc fixed 2k trained steps
# # data_path_hcc = "/home/rh19400/neuro-rl/exp_local/2023.11.27/174214_hcc_agent=hcc,experiment=wm_s,save_stats=true,seed=1,use_wandb=false/locations"

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
width = 0.3
flavour = Flavour.frappe()
for field, name in zip(fields(flavour), names):
    colour = getattr(flavour, field.name)
    print(f"{name} - {field.name}: #{colour.hex}")
    model_color[name] = f"#{colour.hex}"

print(model_color)

# model_color = {
#     "hc-cb": "darkturquoise",
#     "hc-cb abl": "pink",
#     "hc": "sienna",
#     "plastic-hc": "olive",
#     "animal": "forestgreen",
#     "plastic-hc-cb": "cyan",
# }


def get_df(data_path):
    df = pd.read_csv(f"{data_path}/agent_pos.csv")
    return df


# columns = [f'neuron_{i}' for i in range(512)]
# df.head()


def get_reward_and_steps(model, episode_num=100):

    df_filter = model.loc[model["episode"] < episode_num]
    df_group = df_filter.groupby("episode")
    steps_mean = df_group.describe()["step"]["max"].mean()  # mean=124.41 sem=11.298
    steps_sem = df_group.describe()["step"]["max"].sem()
    rew = df_group.describe()["reward"]["max"].mean()  # 97%
    rew_sem = df_group.describe()["reward"]["max"].sem()  # 97%
    return steps_mean, steps_sem, rew, rew_sem


def get_ax_trajectory(
    ax1, model1, trials=100, title="Trajectory", s_start=10**2, s_reward=11**4
):
    reward = [5.5, 6.5]
    colors = {
        (1.5, 7.5): "red",
        (7.5, 3.5): "green",
        (3.5, 4.5): "blue",
        (2.5, 1.5): "orange",
    }
    for i in range(0, trials):
        model1_1 = model1.loc[model1["episode"] == i].reset_index(drop=True)
        ax1.scatter(
            model1_1["position_x"][1],
            model1_1["position_y"][1],
            marker="x",
            linewidth=8,
            s=s_start,
            color=colors[
                (
                    round(model1_1["position_x"][1], 1),
                    round(model1_1["position_y"][1], 1),
                )
            ],
        )
        ax1.plot(
            model1_1["position_x"][1:],
            model1_1["position_y"][1:],
            linewidth=1,
            alpha=0.9,
            color=colors[
                (
                    round(model1_1["position_x"][1], 1),
                    round(model1_1["position_y"][1], 1),
                )
            ],
        )
    ax1.scatter(reward[0], reward[1], marker="o", linewidth=1, s=s_reward, alpha=0.5)
    ax1.set_ylim([0.5, 9.5])
    ax1.set_xlim([0.5, 9.5])
    ax1.hlines(5, 0, 9, linewidth=1, alpha=0.5)
    ax1.vlines(5, 0, 9, linewidth=1, alpha=0.5)
    ax1.set(title=title, xlabel="x", ylabel="y")

    return ax1


def plot_all(models):

    fig = plt.figure(figsize=(24, 9))
    gs = plt.GridSpec(3, 6, wspace=0.3, hspace=0.5)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 3])
    ax5 = fig.add_subplot(gs[0, 4])
    ax6 = fig.add_subplot(gs[0, 5])
    ax7 = fig.add_subplot(gs[1, 1])
    ax8 = fig.add_subplot(gs[1, 3])
    ax9 = fig.add_subplot(gs[1, 4])
    ax10 = fig.add_subplot(gs[1, 5])
    axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10]
    ax11 = fig.add_subplot(gs[2, 0:3])
    ax12 = fig.add_subplot(gs[2, 3:6])
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

    for i in range(len(models)):
        ax = get_ax_trajectory(
            axes[i],
            models[i],
            trials=100,
            title=names[i],
            s_start=10**1,
            s_reward=11**3,
        )

        steps_m, steps_sem, rew_m, rew_sem = get_reward_and_steps(models[i], 100)
        ax11.bar(
            names[i], rew_m, yerr=rew_sem, width=width, color=model_color[names[i]]
        )
        ax12.bar(
            names[i], steps_m, yerr=steps_sem, width=width, color=model_color[names[i]]
        )
    # remove top, left and right spines
    ax11.spines["top"].set_visible(False)
    ax11.spines["right"].set_visible(False)
    ax12.spines["top"].set_visible(False)
    ax12.spines["right"].set_visible(False)
    ax11.set(title="Average reward")
    ax11.tick_params(axis="x", rotation=45)
    ax12.set(title="Average steps")
    ax12.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()
    # fig.savefig("all_trajec.pdf", bbox_inches="tight")


def plot_together(df_drqn, df_hcc):
    fig, ax = plt.subplots(figsize=(8, 8))
    reward = [5.5, 6.5]
    colors = {
        (1.5, 7.5): "red",
        (7.5, 3.5): "green",
        (3.5, 4.5): "blue",
        (2.5, 1.5): "orange",
    }

    for i in range(0, 25):
        df_drqn_1 = df_drqn.loc[df_drqn["episode"] == i].reset_index(drop=True)
        df_hcc_1 = df_hcc.loc[df_hcc["episode"] == i].reset_index(drop=True)
        ax.scatter(reward[0], reward[1], marker="o", linewidth=1, s=9**4, alpha=0.5)
        ax.scatter(
            df_drqn_1["position_x"][1],
            df_drqn_1["position_y"][1],
            marker="x",
            linewidth=8,
            s=10**2,
            color=colors[
                (
                    round(df_drqn_1["position_x"][1], 1),
                    round(df_drqn_1["position_y"][1], 1),
                )
            ],
        )
        # ax.plot(
        #     df_drqn_1["position_x"][1:],
        #     df_drqn_1["position_y"][1:],
        #     linewidth=1,
        #     alpha=0.9,
        #     color=colors[
        #         (round(df_drqn_1["position_x"][1], 1), round(df_drqn_1["position_y"][1], 1))
        #     ],
        # )
        ax.plot(
            df_hcc_1["position_x"][1:],
            df_hcc_1["position_y"][1:],
            linewidth=1,
            linestyle="--",
            color=colors[
                (
                    round(df_hcc_1["position_x"][1], 1),
                    round(df_hcc_1["position_y"][1], 1),
                )
            ],
            alpha=0.9,
        )
        ax.set_ylim([0.5, 9.5])
        ax.set_xlim([0.5, 9.5])
        ax.hlines(5, 0, 9, linewidth=1, alpha=0.5)
        ax.vlines(5, 0, 9, linewidth=1, alpha=0.5)

    plt.show()


# plt.savefig('trajec_tog.pdf')


def plot_separately(model1, model2):

    fig = plt.figure(figsize=(16, 6))
    gs = plt.GridSpec(2, 5, wspace=0.5, hspace=0.3)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    ax2 = fig.add_subplot(gs[0:2, 2:4])
    ax3 = fig.add_subplot(gs[0, 4])
    ax4 = fig.add_subplot(gs[1, 4])

    reward = [5.5, 6.5]
    colors = {
        (1.5, 7.5): "red",
        (7.5, 3.5): "green",
        (3.5, 4.5): "blue",
        (2.5, 1.5): "orange",
    }

    for i in range(0, 100):
        model1_1 = model1.loc[model1["episode"] == i].reset_index(drop=True)
        model2_1 = model2.loc[model2["episode"] == i].reset_index(drop=True)
        ax1.scatter(
            model1_1["position_x"][1],
            model1_1["position_y"][1],
            marker="x",
            linewidth=8,
            s=10**2,
            color=colors[
                (
                    round(model1_1["position_x"][1], 1),
                    round(model1_1["position_y"][1], 1),
                )
            ],
        )
        ax2.scatter(
            model2_1["position_x"][1],
            model2_1["position_y"][1],
            marker="x",
            linewidth=8,
            s=10**2,
            color=colors[
                (
                    round(model2_1["position_x"][1], 1),
                    round(model2_1["position_y"][1], 1),
                )
            ],
        )
        ax1.plot(
            model1_1["position_x"][1:],
            model1_1["position_y"][1:],
            linewidth=1,
            alpha=0.9,
            color=colors[
                (
                    round(model1_1["position_x"][1], 1),
                    round(model1_1["position_y"][1], 1),
                )
            ],
        )
        ax2.plot(
            model2_1["position_x"][1:],
            model2_1["position_y"][1:],
            linewidth=1,
            linestyle="--",
            alpha=0.9,
            color=colors[
                (
                    round(model2_1["position_x"][1], 1),
                    round(model2_1["position_y"][1], 1),
                )
            ],
        )
    ax1.scatter(reward[0], reward[1], marker="o", linewidth=1, s=11**4, alpha=0.5)
    ax2.scatter(reward[0], reward[1], marker="o", linewidth=1, s=11**4, alpha=0.5)
    ax1.set_ylim([0.5, 9.5])
    ax1.set_xlim([0.5, 9.5])
    ax1.hlines(5, 0, 9, linewidth=1, alpha=0.5)
    ax1.vlines(5, 0, 9, linewidth=1, alpha=0.5)
    ax2.set_ylim([0.5, 9.5])
    ax2.set_xlim([0.5, 9.5])
    ax2.hlines(5, 0, 9, linewidth=1, alpha=0.5)
    ax2.vlines(5, 0, 9, linewidth=1, alpha=0.5)
    ax1.set(title="hc-cb abl")
    ax2.set(title="hc-cb")

    steps_mean, steps_sem, rew_mean, rew_sem = get_reward_and_steps(model1, 100)
    steps_mean2, steps_sem2, rew_mean2, rew_sem2 = get_reward_and_steps(model2, 100)
    ax3.bar(["hc-cb abl", "hc-cb"], [rew_mean, rew_mean2], yerr=[rew_sem, rew_sem2])
    ax4.bar(
        ["hc-cb abl", "hc-cb"], [steps_mean, steps_mean2], yerr=[steps_sem, steps_sem2]
    )
    # remove top, left and right spines
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    ax3.set(title="Average reward")
    ax4.set(title="Average steps")

    plt.tight_layout()
    plt.show()
    # fig.savefig("trajec_sep_bio.pdf", bbox_inches="tight")


# df_hc_cb_abl = get_df(data_path_hc_cb_abl)
# df_hc_cb = get_df(data_path_hc_cb)
df_hc_cb_plastic_abl = get_df(data_path_hc_cb_plastic_abl)
df_hc_cb_plastic = get_df(data_path_hc_cb_plastic)
# plot_separately(model1=df_hc_cb_plastic_abl, model2=df_hc_cb_plastic)
# plot_single(model1=df_hc_cb_plastic_abl)
all_models_list = []
all_data_paths = [
    data_path_hc,
    data_path_hc_cb,
    data_path_hc_plastic,
    data_path_hc_cb_plastic,
    data_path_bio_hc_cb_plastic,
    data_path_bio_hc_cb,
    data_path_hc_cb_abl,
    data_path_hc_cb_plastic_abl,
    data_path_bio_hc_cb_plastic_abl,
    data_path_bio_hc_cb_abl,
]
for data_path in all_data_paths:
    all_models_list.append(get_df(data_path))
plot_all(models=all_models_list)
