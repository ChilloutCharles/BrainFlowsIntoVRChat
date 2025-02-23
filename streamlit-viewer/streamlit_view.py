import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import time
import threading

import osc_server  
from pyplot_graphs import get_graphs_and_deltaTime_from_slice, split_by_identifierGroups, plot_all_groups_dark

MILLISECONDS_REFRESH = 500
TIMESTEPS_WINDOW = 256

start_time = time.time()

def write_elapsed_time_till_start(timedesc = ""):
    timeElapsed = time.time() - start_time
    st.write( timedesc + ": Time elapsed in [s]: ", timeElapsed)

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
    write_elapsed_time_till_start("Get NeuroFB slice")

    groups = [['FocusLeft', 'FocusRight', 'FocusAvg'], ['RelaxLeft', 'RelaxRight', 'RelaxAvg']]

    graphs, deltaTime = get_graphs_and_deltaTime_from_slice(slice)
    graphgroups = split_by_identifierGroups(graphs, groups, exclude="Pos")

    write_elapsed_time_till_start("Prep NeuroFB Data")

    plot_all_groups_dark(graphgroups, deltaTime)
    write_elapsed_time_till_start("Plot NeuroFB Data")
    st.session_state["Button Pressed last"] = "NeuroFB"


if st.button("PowerBands") or st.session_state["Button Pressed last"] == "PowerBands":
    sliceLeft = osc_server.get_pwrbands_dataframes_left().get_latest_frames(TIMESTEPS_WINDOW)
    sliceRight = osc_server.get_pwrbands_dataframes_right().get_latest_frames(TIMESTEPS_WINDOW)
    sliceAvg = osc_server.get_pwrbands_dataframes_avg().get_latest_frames(TIMESTEPS_WINDOW)

    write_elapsed_time_till_start("Get PowerBands slices")

    groups = [['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']]
    
    for slice in [sliceLeft, sliceRight, sliceAvg]:
        graphs, deltaTime = get_graphs_and_deltaTime_from_slice(slice)
        graphgroups = split_by_identifierGroups(graphs, groups, exclude="Pos")
        plot_all_groups_dark(graphgroups, deltaTime, title="Power Bands", yMin=-0.1, yMax=1.1)
    
    st.session_state["Button Pressed last"] = "PowerBands"
    write_elapsed_time_till_start("Plot PowerBands Data")

if st.button("Biometrics") or st.session_state["Button Pressed last"] == "Biometrics":
    data = osc_server.get_biometrics_dataframes().get_latest_frames(TIMESTEPS_WINDOW)
    write_elapsed_time_till_start("Get Biometrics slice")
    graphs, deltaTime = get_graphs_and_deltaTime_from_slice(data)

    groups = [[ 'HeartBeatsPerMinute', 'BreathsPerMinute']]
    graphgroups = split_by_identifierGroups(graphs, groups, exclude="Pos")
    write_elapsed_time_till_start("Prep Biometrics Data")
    plot_all_groups_dark(graphgroups, deltaTime, title="Biometrics", yMin=0, yMax=120)
    write_elapsed_time_till_start("Plot Biometrics Data")

    st.session_state["Button Pressed last"] = "Biometrics"


if "waitTimeMultiplier" not in st.session_state:
    st.session_state["waitTimeMultiplier"] = 1.0  

timeElapsed = time.time() - start_time
timeElapsedMilliseconds = timeElapsed * 1000
waitTime_mul = st.session_state["waitTimeMultiplier"]
milliseconds_refresh_multiplied = MILLISECONDS_REFRESH * waitTime_mul


if timeElapsedMilliseconds < milliseconds_refresh_multiplied:
    sleepTime = (milliseconds_refresh_multiplied - timeElapsedMilliseconds) / 1000
    st.write("Sleeping for: ", sleepTime)
    time.sleep(sleepTime)
else:
    st.warning("Time elapsed is greater than refresh rate. Increased refresh rate by " + str( waitTime_mul))
    st.session_state["waitTimeMultiplier"] = waitTime_mul + 0.2  

    write_elapsed_time_till_start("Time elapsed total")
        
st.rerun()

