import dearpygui.dearpygui as dpg

import osc_server
from osc_dataframes import get_graphs_and_deltaTime_from_slice, split_by_identifierGroups
from performance_util import write_elapsed_time_till_start

TIMESTEPS_WINDOW = 1024



def plot_neurofb():

    slice = osc_server.get_neurofb_dataframes().get_latest_frames(TIMESTEPS_WINDOW)

    write_elapsed_time_till_start("Get PowerBands slices")

    groups = [['FocusLeft', 'FocusRight', 'FocusAvg'], ['RelaxLeft', 'RelaxRight', 'RelaxAvg']]

    graphs, deltaTime = get_graphs_and_deltaTime_from_slice(slice)
    graphgroups = split_by_identifierGroups(graphs, groups, exclude="Pos")

    #plot 



def plot_powerbands():
    sliceLeft = osc_server.get_pwrbands_dataframes_left().get_latest_frames(TIMESTEPS_WINDOW)
    sliceRight = osc_server.get_pwrbands_dataframes_right().get_latest_frames(TIMESTEPS_WINDOW)
    sliceAvg = osc_server.get_pwrbands_dataframes_avg().get_latest_frames(TIMESTEPS_WINDOW)

    write_elapsed_time_till_start("Get PowerBands slices")

    groups = [['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']]
    
    for slice in [sliceLeft, sliceRight, sliceAvg]:
        graphs, deltaTime = get_graphs_and_deltaTime_from_slice(slice)
        graphgroups = split_by_identifierGroups(graphs, groups, exclude="Pos")
        # plot
    
    write_elapsed_time_till_start("Plot PowerBands Data")

def plot_biometrics():
    data = osc_server.get_biometrics_dataframes().get_latest_frames(TIMESTEPS_WINDOW)
    write_elapsed_time_till_start("Get Biometrics slice")
    graphs, deltaTime = get_graphs_and_deltaTime_from_slice(data)

    groups = [[ 'HeartBeatsPerMinute', 'BreathsPerMinute']]
    graphgroups = split_by_identifierGroups(graphs, groups, exclude="Pos")
    write_elapsed_time_till_start("Prep Biometrics Data")

    #plot

    write_elapsed_time_till_start("Plot Biometrics Data")

    st.session_state["Button Pressed last"] = "Biometrics"


def save_callback():
    print("Save Clicked")

dpg.create_context()
dpg.create_viewport()
dpg.setup_dearpygui()

with dpg.window(label="Example Window"):
    dpg.add_text("Hello world")
    dpg.add_button(label="Save", callback=save_callback)
    dpg.add_input_text(label="string")
    dpg.add_slider_float(label="float")

dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()