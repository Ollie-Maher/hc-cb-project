import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import rand, randint, randn

# # drqn plastic
# data_path_drqn = "/home/rh19400/neuro-rl/exp_local/2023.11.27/163557_drqn_agent=drqn,experiment=wm_plastic_s2,save_stats=true,seed=1,use_wandb=false/locations"

# # drqn plastic, bigger maze and smaller reward
# data_path_drqn = "/home/rh19400/neuro-rl/exp_local/2023.11.27/163047_drqn_agent=drqn,experiment=wm_plastic_s,save_stats=true,seed=1,use_wandb=false/locations"
# hc fixed 10k trained steps
data_path_drqn = "/home/rh19400/neuro-rl/exp_local/2023.11.28/091550_drqn_agent=drqn,experiment=wm_fixed_s,save_stats=true,seed=5,use_wandb=false/locations"
# hc_cb ablation, 10k trained steps
data_path_hc_cb_abl = "/home/rh19400/neuro-rl/exp_local/2023.11.28/104213_hcc_agent=hcc,experiment=wm_s,save_stats=true,seed=1,use_wandb=false/locations"
# hcc fixed 10k trained steps
data_path_hc_cb = "/home/rh19400/neuro-rl/exp_local/2023.11.28/091619_hcc_agent=hcc,experiment=wm_s,save_stats=true,seed=1,use_wandb=false/locations"
# hc plastic ablation, 10k trained steps
data_path_hc_cb_plastic_abl = "/home/rh19400/neuro-rl/exp_local/2023.11.28/183532_hcc_agent=hcc,experiment=wm_plastic,save_stats=true,seed=15,use_wandb=false/locations"
# hc plastic 10k trained steps
data_path_hc_cb_plastic = "/home/rh19400/neuro-rl/exp_local/2023.11.28/182923_hcc_agent=hcc,experiment=wm_plastic,save_stats=true,seed=15,use_wandb=false/locations"
# # hcc fixed 2k trained steps
# data_path_hcc = "/home/rh19400/neuro-rl/exp_local/2023.11.27/174214_hcc_agent=hcc,experiment=wm_s,save_stats=true,seed=1,use_wandb=false/locations"

# df_drqn = pd.read_csv(f"{data_path_drqn}/agent_pos.csv")
# df_hcc = pd.read_csv(f"{data_path_hcc}/agent_pos.csv")
# df = pd.read_csv(f'{data_path}/agent_pos.csv')


def get_df(data_path):
    df = pd.read_csv(f"{data_path}/agent_pos.csv")
    return df


# columns = [f'neuron_{i}' for i in range(512)]
# df.head()


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

    fig = plt.figure(figsize=(16, 8))
    gs = plt.GridSpec(1, 2, wspace=0.15, hspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
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
    plt.show()

    # plt.savefig('trajec_sep.pdf')


df_hc_cb_abl = get_df(data_path_hc_cb_abl)
df_hc_cb = get_df(data_path_hc_cb)
df_hc_cb_plastic_abl = get_df(data_path_hc_cb_plastic_abl)
df_hc_cb_plastic = get_df(data_path_hc_cb_plastic)
plot_separately(model1=df_hc_cb_plastic_abl, model2=df_hc_cb_plastic)
