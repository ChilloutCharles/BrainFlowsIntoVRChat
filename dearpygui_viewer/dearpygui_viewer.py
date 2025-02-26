import threading

import dearpygui.dearpygui as dpg
from math import sin, cos
import numpy as np

import osc_server
from osc_dataframes import OSCDataFrame, OSCFrameDeque, OSCFrameCollector 
from data_util import get_graphs_and_deltaTime_from_slice, split_by_identifierGroups
from performance_util import write_elapsed_time_till_start

import time

TIMESTEPS_WINDOW = 1024
TIMESTEPS_SHOW_OFFSET = 200
PLOT_WIDTH = 500

#state         
server_started = False

# --------------------------------------------

def run_osc_server(ip, port):
    osc_server.run_server(ip, port)  

def start_server_once():
    global server_started
    if not server_started:
        thread = threading.Thread(
            target=run_osc_server,
            args=("127.0.0.1", 9000),
            daemon=True
        )
        thread.start()
        print("Server started")
        server_started = True

start_server_once()

def plot_neurofb():

    slice = osc_server.get_neurofb_dataframes().get_latest_frames(TIMESTEPS_WINDOW)

    write_elapsed_time_till_start("Get PowerBands slices")

    groups = [['FocusLeft', 'FocusRight', 'FocusAvg'], ['RelaxLeft', 'RelaxRight', 'RelaxAvg']]

    graphs, deltaTime = get_graphs_and_deltaTime_from_slice(slice)
    graphgroups = split_by_identifierGroups(graphs, groups, exclude="Pos")

    #plot 

def powerbands_from_osc():
    frames_left = osc_server.get_pwrbands_dataframes_left().get_frames()
    frames_right = osc_server.get_pwrbands_dataframes_right().get_frames()
    frames_avg = osc_server.get_pwrbands_dataframes_avg().get_frames()

    groups = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

    for frames in [frames_left, frames_right, frames_avg]:
        graphArrayPerGroup = split_by_identifierGroups(frames, groups, exclude="Pos")
    
    return groups, graphArrayPerGroup

def get_metaData_powerbands():
    return {
        "y_range": [0.0, 1.0],
    }

def plot_biometrics():
    data = osc_server.get_biometrics_dataframes().get_latest_frames(TIMESTEPS_WINDOW)
    write_elapsed_time_till_start("Get Biometrics slice")
    graphs, deltaTime = get_graphs_and_deltaTime_from_slice(data)

    groups = [[ 'HeartBeatsPerMinute', 'BreathsPerMinute']]
    graphgroups = split_by_identifierGroups(graphs, groups, exclude="Pos")
    write_elapsed_time_till_start("Prep Biometrics Data")

    #plot

    write_elapsed_time_till_start("Plot Biometrics Data")


def save_callback():
    print("Save Clicked")

dpg.create_context()
dpg.create_viewport( title="Dearpygui Viewer", width=800*2, height=800*2)
dpg.setup_dearpygui()

with dpg.window(label="Example dynamic plot", autosize=True):
    

    def fetch_powerbands():
        return powerbands_from_osc()

    dpg.add_button(label="Fetch Powerbands", callback=fetch_powerbands)

    with dpg.tree_node(label="Digital Plots", default_open=True):
        
        t_plot = 0.0
        plot_data_x = []
        plot_data_y = []
        plot_show = []
        plot_lastFrameIndex = -1


        osc_meta_data = get_metaData_powerbands()

        osc_frame_labels, osc_frame_data = fetch_powerbands()      

        plot_data_x =  [[] for _ in range(len(osc_frame_labels))]
        plot_data_y =  [[] for _ in range(len(osc_frame_labels))]
        plot_show = [True for _ in range(len(osc_frame_labels))]

        def _update_plot_data(new_data_frame = []):

            if len(plot_data_x) == 0:
                return
            if frame_lastFrameIndex == -1:
                return

            for _, frame_data_array in enumerate(new_data_frame):
                if frame_data_array[0].frameIdx > frame_lastFrameIndex:
                    plot_data_y.append(frame_data_array)
                if frame_data_array[0].frameIdx < frame_lastFrameIndex:
                    # make warning on gui
                    print("Warning: frame data out of order")
                    dpg.add_text("Warning: frame data out of order") 

                delta = new_data_frame[len(new_data_frame-1)].frameIdx - frame_lastFrameIndex
                if delta > 1:
                    # make warning on gui
                    print("Warning: frame data missing")
                    # this is formatted string
                    dpg.add_text(f"Warning: frame data missing {delta} frames")

                frame_lastFrameIndex = frame_data_array[-1].frameIdx

        _update_plot_data(framedata=osc_frame_data, frame_lastFrameIndex=plot_lastFrameIndex)

        def change_val(arr, ind, val):
            arr[ind] = val

        # ToDo initialize data_frames
        with dpg.group(horizontal=True):
            for label in osc_frame_labels :
                dpg.add_checkbox(label=label, callback=lambda s, a: change_val(plot_show, 0, a),
                                                 default_value=True)

            with dpg.plot(tag="_powerbands_digital_plot", width=PLOT_WIDTH):
                # X axis
                dpg.add_plot_axis(dpg.mvXAxis, label="x", tag="x_axis_time")
                dpg.set_axis_limits(dpg.last_item(), 0, plot_lastFrameIndex)
                with dpg.plot_axis(dpg.mvYAxis, label="y"):
                    dpg.set_axis_limits(dpg.last_item(), osc_meta_data["y_range"][0], osc_meta_data["y_range"][1])
                    for x_label in osc_frame_labels:
                        dpg.add_line_series([], [], label=x_label, tag=x_label)


                    def _update_plot():
                            
                            frame_digital_labels, frame_data_array = fetch_powerbands()

                            t_plot = frame_data_array[0][0].frameIdx
                            dpg.set_axis_limits('x_axis_digital', t_plot - TIMESTEPS_WINDOW - TIMESTEPS_SHOW_OFFSET, t_plot)
                            #print(frame_data_array[0][0])
                            
                            _update_plot_data()
                            for i, data_analog in enumerate(frame_data_array):
                                if plot_show[i]:
                                    # set value from slice tplot - TIMESTEPS_WINDOW to tplot?
                                    dpg.set_value(frame_digital_labels[i], [*zip(*data_analog[i])])


                    with dpg.item_handler_registry(tag="_powerbands_digital_plot_ref"):
                        dpg.add_item_visible_handler(callback=_update_plot)
                    dpg.bind_item_handler_registry("_powerbands_digital_plot", dpg.last_container())
                    




    dpg.add_text(t_plot, label="Time:")


dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()