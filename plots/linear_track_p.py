import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# df_ca3 = pd.read_csv(f"{data_path}/ca3.csv")
# df_ca1 = pd.read_csv(f"{data_path}/ca1.csv")


def get_neurons_hm(df, linear_track=False):

    df = df.loc[df["episode"] <= 50]
    # df = df.loc[df["episode"] == 6]
    # # filter between 100 and 120
    # df = df.loc[df["episode"] <= 130]
    # df = df.loc[df["episode"] >= 110]
    df["position_x"] = df["position_x"].round()
    df["position_y"] = df["position_y"].round()

    # df = df.groupby(['position_x', 'position_y']).mean().reset_index()
    # df = df.pivot(index='position_x', columns='position_y', values='firing_rate')

    if linear_track:
        # filter by posx, posy and reward = 1
        # [6, 6]  [3, 6], [9, 6]
        # after filtering, get the episodes
        eps = df.loc[(df['reward'] == 1) & (df['position_x'] ==3)]['episode'].values
        df = df[df['episode'].isin(eps)]

    df_pos = pd.DataFrame(columns=df.columns)
    df_pos = df_pos.drop(columns=["step", "episode", "reward", "env_id"])
    i = 0
    neurons = np.zeros((512, 13, 13))
    for x in range(1, 14):
        df_posx = df.loc[df["position_x"] == x]
        for y in range(1, 14):
            df_posxy = df_posx.loc[df_posx["position_y"] == y]
            df_posxy = df_posxy.drop(columns=["step", "episode", "reward", "env_id"])
            df_pos.loc[i] = df_posxy.mean()
            neurons[:, x - 1, y - 1] = df_posxy.drop(
                columns=["position_x", "position_y"]
            ).mean()
            i += 1

    neurons[np.isnan(neurons)] = 0

    # norm_neurons = neurons / np.linalg.norm(neurons)
    norm_neurons = neurons 
    m_vals = norm_neurons.mean(axis=(1,2))
    # Get the indices that would sort the mean values in descending order
    # sorted_indices = np.argsort(m_vals)#[::-1]
    sorted_indices = np.argsort(m_vals)[::-1]
    # Use take_along_axis to reorder the original array based on the sorted indices
    sorted_neurons = np.take_along_axis(norm_neurons, sorted_indices[:, None, None], axis=0)
    # sorted_neurons = norm_neurons
    return sorted_neurons 
    # norm_neurons = neurons
    return norm_neurons


def plot_neurons_hm(data, name="ca3"):
    # Set the number of rows and columns for the subplot grid
    num_rows = 1
    num_cols = 2

    # Create a figure and a grid of subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 12),subplot_kw={"projection": "3d"})
    # fig, axs = plt.subplots(num_rows, num_cols)

    # Flatten the 2D array of Axes objects into a 1D array for easier indexing
    axs = axs.flatten()

    cmap = "bwr"
    # interp = "none"
    interp = "gaussian"
    vmin = -0.008
    vmax = 0.008
    origin = "lower"
    offset = 200

    X = np.arange(0, 13, 1)
    Y = np.arange(0, 13, 1)
    X, Y = np.meshgrid(X, Y)
        

    # Loop through each subplot and create a heatmap
    for i in range(num_rows * num_cols):
        # im = axs[i].imshow(data[i], cmap="viridis", extent=[0, 100, 0, 100], origin="lower")
        # im = axs[i].imshow(
        #     data[i+offset][:,4:7],
        #     cmap=cmap,
        #     origin=origin,
        #     interpolation=interp,
        #     vmin=vmin,
        #     vmax=vmax,
        # )
        # axs[i].set_title(f"Neuron {i+1}")
        # axs[i].set_axis_off()
        # print(data[i+offset].shape)

        surf = axs[i].plot_surface(
            X, 
            Y, 
            data[i+offset][:,:],
            # data[i+offset][:,4:7],
            cmap=cmap,
            # origin=origin,
            # interpolation=interp,
            # vmin=vmin,
            # vmax=vmax,
        )

    # Add a colorbar for reference
    # cbar = fig.colorbar(im, ax=axs[-1], orientation="vertical", fraction=0.05, pad=0.05)
    # plt.axis("off")

    # Adjust layout to prevent clipping of titles
    # plt.tight_layout()

    # Show the plot
    plt.show()
    # fig.savefig(f"neurons_{name}_hm.pdf", bbox_inches="tight")



# Linear track
# # linear track hc-cb model
# data_path = '/home/rh19400/neuro-rl/exp_local/2024.01.03/213708_hcc_agent=hcc,experiment=lt,save_stats=true,seed=1,use_wandb=false/activations'
# data_path = "/home/rh19400/neuro-rl/exp_local/2024.01.03/233936_hcc_agent=hcc,experiment=lt,save_stats=true,seed=1,use_wandb=false/activations"
# hc-cb abl
# data_path = "/home/rh19400/neuro-rl/exp_local/2024.01.03/232917_hcc_agent=hcc,experiment=lt,save_stats=true,seed=1,use_wandb=false/activations"

# hc fixed
# data_path = "/home/rh19400/neuro-rl/exp_local/2024.01.03/235324_drqn_agent=drqn,experiment=lt,save_stats=true,seed=1,use_wandb=false/activations"
data_path = "/home/rh19400/neuro-rl/exp_local/2024.01.12/155138_hcc_agent=hcc,experiment=lt_1,save_stats=true,seed=1/activations"
# df = pd.read_csv(f'{data_path}/ca3.csv')
df = pd.read_csv(f'{data_path}/ca1.csv')
# model = "hc-cb-linear"
model = "hc-linear"
# model = "hc-cb-linear-abl"
data_ca1 = get_neurons_hm(df, linear_track=False)
plot_neurons_hm(data_ca1, name=f"ca1_{model}")


