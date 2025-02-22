import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def transform_into_data_dict(slice):

    sumDeltaTime = 0
    dict = {}
    for frame in slice:
        sumDeltaTime += frame.secondsSinceLastUpdate
        for key, value in frame.data.items():
            if key not in dict:
                dict[key] = []
            dict[key].append(value)

    return dict, sumDeltaTime

def print_info_from_graph_dict(dict: dict):
    st.write("Containing the following data streams:")
    if(dict == None or len(dict) == 0):
        st.write("No data streams found.")
        return

def get_graphs_from_slice(slice) -> dict:

    dict, deltaTime = transform_into_data_dict(slice)
    print_info_from_graph_dict(dict)
    return dict, deltaTime


def split_by_identifierGroups(dataDict, groups, exclude=""):
    splitDictIdentified = []
    for group in groups:
        subDict = {}
        for key, value in dataDict.items():
            # compare key against all elements in group
            for identifier in group:
                if identifier in key  and exclude not in key:
                    subDict[key] = value
        splitDictIdentified.append(subDict)
    return splitDictIdentified

def plot_all_groups_dark(graphgroups, time, deltaTime, title="Set Title", yMin=-1.0, yMax = 1.0):

    plt.style.use('dark_background')

    # We'll assume each group has consistent channels. Grab channel names from first group:
    first_group = graphgroups[0]
    channel_names = list(first_group.keys())

    # Also assume each channel has the same number of samples to define time axis:
    first_channel_data = first_group[channel_names[0]]
    n_samples = len(first_channel_data)

    # For demonstration, set sampling_rate = 1.0
    sampling_rate = 1.0
    time = np.linspace(0, n_samples / sampling_rate, n_samples)

    # Plot all groups/channels on the same Axes
    for g_idx, group_dict in enumerate(graphgroups):
     # Create ONE figure and ONE set of axes
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot all groups/channels on the same Axes
        for chan_name, channel_data in group_dict.items():
            if channel_data is None:
                continue

            # Label with group index + channel name
            ax.plot(time, channel_data, label=f"G{g_idx+1}-{chan_name}")

                # Optional: set a fixed y range (adjust as needed)
            ax.set_ylim(yMin, yMax)

            # Label axes and show legend
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.set_title( title + " over " + str(deltaTime) + " seconds")
            ax.legend(loc="upper right")


    st.pyplot(fig)