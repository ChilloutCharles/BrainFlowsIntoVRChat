import streamlit as st
from streamlit_autorefresh import st_autorefresh
import matplotlib.pyplot as plt
import numpy as np
import time
import threading

import osc_server  
from pyplot_graphs import get_graphs_and_deltaTime_from_slice, split_by_identifierGroups, plot_all_groups_dark

MILLISECONDS_REFRESH = 2000
TIMESTEPS_WINDOW = 1024/4

def run_osc_server(ip, port):
    osc_server.run_server(ip, port)  

def start_server_once():
    if "server_thread" not in st.session_state:
        thread = threading.Thread(
            target=run_osc_server,
            args=("127.0.0.1", 9000),
            daemon=True
        )
        thread.start()
        print("Server started")
        st.session_state["server_thread"] = thread
        st.write("OSC Server started on port 9000.")

start_server_once()

st.title("BrainFlows Streamlit Viewer")

if "Button Pressed last" not in st.session_state:
    st.session_state["Button Pressed last"] = "None"



if st.button("NeuroFB") or st.session_state["Button Pressed last"] == "NeuroFB":
    slice = osc_server.get_neurofb_dataframes().get_latest_frames(TIMESTEPS_WINDOW)

    groups = [['FocusLeft', 'FocusRight', 'FocusAvg'], ['RelaxLeft', 'RelaxRight', 'RelaxAvg']]

    graphs, deltaTime = get_graphs_and_deltaTime_from_slice(slice)
    graphgroups = split_by_identifierGroups(graphs, groups, exclude="Pos")

    plot_all_groups_dark(graphgroups, deltaTime)
    st.session_state["Button Pressed last"] = "NeuroFB"


if st.button("PowerBands") or st.session_state["Button Pressed last"] == "PowerBands":
    sliceLeft = osc_server.get_pwrbands_dataframes_left().get_latest_frames(TIMESTEPS_WINDOW)
    sliceRight = osc_server.get_pwrbands_dataframes_right().get_latest_frames(TIMESTEPS_WINDOW)
    sliceAvg = osc_server.get_pwrbands_dataframes_avg().get_latest_frames(TIMESTEPS_WINDOW)

    groups = [['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']]
    
    for slice in [sliceLeft, sliceRight, sliceAvg]:
        graphs, deltaTime = get_graphs_and_deltaTime_from_slice(slice)
        graphgroups = split_by_identifierGroups(graphs, groups, exclude="Pos")
        plot_all_groups_dark(graphgroups, deltaTime, title="Power Bands", yMin=-0.1, yMax=1.1)
    
    st.session_state["Button Pressed last"] = "PowerBands"

if st.button("Biometrics") or st.session_state["Button Pressed last"] == "Biometrics":
    data = osc_server.get_biometrics_dataframes().get_latest_frames(TIMESTEPS_WINDOW)
    graphs, deltaTime = get_graphs_and_deltaTime_from_slice(data)

    groups = [[ 'HeartBeatsPerMinute', 'BreathsPerMinute']]
    graphgroups = split_by_identifierGroups(graphs, groups, exclude="Pos")
    plot_all_groups_dark(graphgroups, deltaTime, title="Biometrics", yMin=0, yMax=120)

    st.session_state["Button Pressed last"] = "Biometrics"



count = st_autorefresh (interval=MILLISECONDS_REFRESH)
if "time" not in st.session_state:
    st.session_state["time"] = time.time()
deltasecounds = time.time() - st.session_state["time"]
st.session_state["time"] = time.time()
st.write(f"Time since last refresh: {deltasecounds} seconds.")
st.write(f"Page refreshed {count} times.")

