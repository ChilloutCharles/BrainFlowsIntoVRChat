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
PLOT_WIDTH = 500

t_digital_plot = 0.0
server_started = False

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
    deltaTime = 0.0
    sliceLeft = osc_server.get_pwrbands_dataframes_left().get_latest_frames(TIMESTEPS_WINDOW)
    sliceRight = osc_server.get_pwrbands_dataframes_right().get_latest_frames(TIMESTEPS_WINDOW)
    sliceAvg = osc_server.get_pwrbands_dataframes_avg().get_latest_frames(TIMESTEPS_WINDOW)

    write_elapsed_time_till_start("Get PowerBands slices")

    groups = [['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']]
    groupsx = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

    for slice in [sliceLeft, sliceRight, sliceAvg]:
        graphs, deltaTime = get_graphs_and_deltaTime_from_slice(slice)
        graphArrayPerGroup = split_by_identifierGroups(graphs, groups, exclude="Pos")
    
    write_elapsed_time_till_start("Plot PowerBands Data")

    return  groupsx, graphArrayPerGroup, 0.0, 1.0, deltaTime

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

    with dpg.tree_node(label="Digital Plots"):
        dpg.add_text(default_value="Digital plots do not respond to Y drag and zoom, so that",
                                     bullet=True)
        dpg.add_text(default_value="you can drag analog plots over the rising/falling digital edge.",
                                     indent=20)
        
        def set_paused(new_bool, paused):
            paused = new_bool
            return paused

        frame_data_dict = {}

        paused = False
        frame_digital_data = []
        frame_show = []
        
        frame_min_y = 0.0
        frame_max_y = 0.0
        frame_deltaTime = 0.0

        # TODO: Improve handing so that sleep is not needed
        frame_digital_labels, frame_data_dict, frame_min_y, frame_max_y,frame_deltaTime = fetch_powerbands()
        if len(frame_digital_labels) == 0:
            time.sleep(1)
            frame_digital_labels, frame_data_dict, frame_min_y, frame_max_y,frame_deltaTime = fetch_powerbands()

        data_analog =  [[], []]
        frame_show = [True for _ in range(len(frame_digital_labels))] 

        dpg.add_button(label="Pause", callback=lambda: set_paused(not paused, paused))
        
        def change_val(arr, ind, val):
            arr[ind] = val

        # ToDo initialize data_frames
        with dpg.group(horizontal=True):
            for label in frame_digital_labels :
                dpg.add_checkbox(label=label, callback=lambda s, a: change_val(frame_show, 0, a),
                                                 default_value=True)

            with dpg.plot(tag="_demo_digital_plot", width=PLOT_WIDTH):
            # TODO: better handling of show/hide (more consistency between checkboxes and legend)
                dpg.add_plot_axis(dpg.mvXAxis, label="x", tag="x_axis_digital")
                dpg.set_axis_limits(dpg.last_item(), -TIMESTEPS_WINDOW, 0)
                with dpg.plot_axis(dpg.mvYAxis, label="y"):
                    dpg.set_axis_limits(dpg.last_item(), frame_min_y, frame_max_y)
                    for x_label in frame_digital_labels:
                        dpg.add_line_series([], [], label=x_label, tag=x_label)


                    def _update_plot():
                        global t_digital_plot
                        if not paused:
                            t_digital_plot += dpg.get_delta_time()
                            
                            frame_digital_labels, frame_data_array, frame_min_y, frame_max_y,frame_deltaTime = fetch_powerbands()

                            dpg.set_axis_limits('x_axis_digital', t_digital_plot - 10, t_digital_plot)
                                #dpg.set_value(str(frame_digital_labels[i]), [*zip(*data[i])])
                            print(frame_data_array[0][0])
                            data_analog[0].append( [t_digital_plot, frame_data_array[0][0][0]]) # =??
                            dpg.set_value(frame_digital_labels[0], [*zip(*data_analog[0])])


                    with dpg.item_handler_registry(tag="__demo_digital_plot_ref"):
                        dpg.add_item_visible_handler(callback=_update_plot)
                    dpg.bind_item_handler_registry("_demo_digital_plot", dpg.last_container())
                    




    dpg.add_text(t_digital_plot, label="Time:")
    dpg.add_text(paused, label="Paused:")


dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()