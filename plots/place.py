import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# bio hc-cb plastic abl
# data_path = "/home/rh19400/neuro-rl/exp_local/2024.01.01/130425_hccbio_agent=hccbio,experiment=wm_plastic,save_stats=true,seed=1/activations"
# bio hc-cb plastic
# data_path = "/home/rh19400/neuro-rl/exp_local/2024.01.01/175354_hccbio_agent=hccbio,experiment=wm_plastic,save_stats=true,seed=1/activations"

# # bio hc-cb fixed
data_path = "/home/rh19400/neuro-rl/exp_local/2024.01.02/111128_hccbio_agent=hccbio,experiment=wm,save_stats=true,seed=1/activations"
# # bio hc-cb fixed abl
# data_path = "/home/rh19400/neuro-rl/exp_local/2024.01.02/113012_hccbio_agent=hccbio,experiment=wm,save_stats=true,seed=1/activations"

# hc fixed
# data_path = "/home/rh19400/neuro-rl/exp_local/2024.01.01/181903_drqn_agent=drqn,experiment=wm_fixed_s,save_stats=true,seed=5/activations"

# hc plastic
# data_path = "/home/rh19400/neuro-rl/exp_local/2024.01.01/183435_drqn_agent=drqn,experiment=wm_plastic,save_stats=true,seed=10/activations"

# hc cb plastic abl
# data_path = "/home/rh19400/neuro-rl/exp_local/2024.01.02/094617_hcc_agent=hcc,experiment=wm_plastic,save_stats=true,seed=15/activations"

# hc cb plastic
# data_path = "/home/rh19400/neuro-rl/exp_local/2024.01.02/100138_hcc_agent=hcc,experiment=wm_plastic,save_stats=true,seed=15/activations"

# hc cb fixed
# data_path = "/home/rh19400/neuro-rl/exp_local/2024.01.02/103144_hcc_agent=hcc,experiment=wm_s,save_stats=true,seed=1/activations"

# hc cb fixed abl
# data_path = "/home/rh19400/neuro-rl/exp_local/2024.01.02/104355_hcc_agent=hcc,experiment=wm_s,save_stats=true,seed=1/activations"

# model = "bio-plastic-hc-cb"
# model = "bio-plastic-hc-cb-abl"
model = "bio-hc-cb"
# model = "bio-hc-cb-abl"
# model = "hc"
# model = "hc-plastic"
# model = "plastic-hc-cb"
# model = "plastic-hc-cb-abl"
# model = "hc-cb"
# model = "hc-cb-abl"

df_ca3 = pd.read_csv(f"{data_path}/ca3.csv")
df_ca1 = pd.read_csv(f"{data_path}/ca1.csv")


def get_neurons_hm(df):

    df = df.loc[df["episode"] <= 90]
    # df = df.loc[df["episode"] == 6]
    # # filter between 100 and 120
    # df = df.loc[df["episode"] <= 130]
    # df = df.loc[df["episode"] >= 110]
    df["position_x"] = df["position_x"].round()
    df["position_y"] = df["position_y"].round()

    # df = df.groupby(['position_x', 'position_y']).mean().reset_index()
    # df = df.pivot(index='position_x', columns='position_y', values='firing_rate')

    df_pos = pd.DataFrame(columns=df.columns)
    df_pos = df_pos.drop(columns=["step", "episode", "reward", "env_id"])
    i = 0
    neurons = np.zeros((512, 9, 9))
    for x in range(1, 10):
        df_posx = df.loc[df["position_x"] == x]
        for y in range(1, 10):
            df_posxy = df_posx.loc[df_posx["position_y"] == y]
            df_posxy = df_posxy.drop(columns=["step", "episode", "reward", "env_id"])
            df_pos.loc[i] = df_posxy.mean()
            neurons[:, x - 1, y - 1] = df_posxy.drop(
                columns=["position_x", "position_y"]
            ).mean()
            i += 1

    neurons[np.isnan(neurons)] = 0

    norm_neurons = neurons / np.linalg.norm(neurons)
    # norm_neurons = neurons 
    # m_vals = norm_neurons.mean(axis=(1,2))
    # # Get the indices that would sort the mean values in descending order
    # sorted_indices = np.argsort(m_vals)#[::-1]
    # # sorted_indices = np.argsort(m_vals)[::-1]
    # # Use take_along_axis to reorder the original array based on the sorted indices
    # sorted_neurons = np.take_along_axis(norm_neurons, sorted_indices[:, None, None], axis=0)
    sorted_neurons = norm_neurons
    return sorted_neurons 


def plot_neurons_hm(data, name="ca3", axs=None):
    # Set the number of rows and columns for the subplot grid
    num_rows = 1
    num_cols = 5

    # # Create a figure and a grid of subplots
    # fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 12))
    # # fig, axs = plt.subplots(num_rows, num_cols)

    # # Flatten the 2D array of Axes objects into a 1D array for easier indexing
    # axs = axs.flatten()

    cmap = "bwr"
    interp = "none"
    interp = "gaussian"
    vmin = -0.008
    vmax = 0.008
    origin = "lower"
    offset = 9

    # Loop through each subplot and create a heatmap
    for i in range(num_rows * num_cols):
        # im = axs[i].imshow(data[i], cmap="viridis", extent=[0, 100, 0, 100], origin="lower")
        im = axs[i].imshow(
            data[i + offset],
            cmap=cmap,
            origin=origin,
            interpolation=interp,
            vmin=vmin,
            vmax=vmax,
        )
        axs[i].set_title(f"Neuron {i+1}")
        # axs[i].set_axis_off()

    # Add a colorbar for reference
    cbar = fig.colorbar(im, ax=axs[0], orientation="vertical", fraction=0.05, pad=0.05)
    # plt.axis("off")

    # Adjust layout to prevent clipping of titles
    plt.tight_layout()

    # Show the plot
    # plt.show()
    # fig.savefig(f"neurons_{name}_hm.pdf", bbox_inches="tight")

# Set the number of rows and columns for the subplot grid
num_rows = 2
num_cols = 5

# Create a figure and a grid of subplots
fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 12))
# fig, axs = plt.subplots(num_rows, num_cols)

# Flatten the 2D array of Axes objects into a 1D array for easier indexing
axs = axs.flatten()

data_ca3 = get_neurons_hm(df_ca3)
plot_neurons_hm(data_ca3, name=f"ca3_{model}", axs=axs[:5])

data_ca1 = get_neurons_hm(df_ca1)
plot_neurons_hm(data_ca1, name=f"ca1_{model}", axs=axs[5:])


# Add a colorbar for reference
# cbar = fig.colorbar(im, ax=axs[0], orientation="vertical", fraction=0.05, pad=0.05)
# plt.axis("off")
# Adjust layout to prevent clipping of titles
plt.tight_layout()
plt.show()
# fig.savefig(f"neurons_{model}_hm.pdf", bbox_inches="tight")
