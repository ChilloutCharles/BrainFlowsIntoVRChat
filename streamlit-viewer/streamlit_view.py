import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import threading

import osc_server  
from pyplot_graphs import get_graphs_from_slice, split_by_identifierGroups, plot_all_groups_dark

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
        st.session_state["server_thread"] = thread
        st.write("OSC Server started on port 9000.")

start_server_once()

st.title("Streamlit Viewer")

if st.button("NeuroFB"):
    slice = osc_server.get_neurofb_dataframes().get_latest_frames(1024)

    groups = [['FocusLeft', 'FocusRight', 'FocusAvg'], ['RelaxLeft', 'RelaxRight', 'RelaxAvg']]

    graphs, deltaTime = get_graphs_from_slice(slice)
    graphgroups = split_by_identifierGroups(graphs, groups, exclude="Pos")

    plot_all_groups_dark(graphgroups, deltaTime)


if st.button("PowerBands"):
    slice = osc_server.get_pwrbands_dataframes().get_latest_frames(1024)
    graphs, deltaTime = get_graphs_from_slice(slice)

    groups = [['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']]
    graphgroups = split_by_identifierGroups(graphs, groups)
    plot_all_groups_dark(graphgroups, deltaTime)

if st.button("Biometrics"):
    data = osc_server.get_biometrics_dataframes().get_latest_frames(1024)
    graphs, deltaTime = get_graphs_from_slice(data)

    groups = [[ 'HeartBeatsPerMinute ', 'BreathsPerMinute ']]
    graphs = get_graphs_from_slice(slice)
    plot_all_groups_dark(graphgroups, deltaTime)



