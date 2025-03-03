import threading
import dearpygui.dearpygui as dpg
import numpy as np
import time

import osc_server

PLOT_WIDTH = 500

#state         
server_started = False

t_digital_plot = 0
last_processed_counter = 0

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


def _fetch_complete_data_from_server(osc_keys, last_processed_counter):

    osc_and_counter_data = [ osc_server.read_from_osc_buffer(key) for key in osc_keys]
    delta_time_data, delta_time_counter = osc_server.read_from_osc_buffer_elapsedTime()

    buffer_counters = [ osc_and_counter_data[i][1] for i in range(len(osc_and_counter_data))]
    buffer_data = [ osc_and_counter_data[i][0] for i in range(len(osc_and_counter_data))]

    complete_counter_data = min(buffer_counters)
    complete_frame_counter = min(complete_counter_data, delta_time_counter)

    index_delta = complete_frame_counter - last_processed_counter

    next_data = []
    deltaTime = 0

    if index_delta > 0 and len(osc_and_counter_data) > 0:

        for idx_label in range(len(osc_keys)):
            assert( index_delta > buffer_counters[idx_label],
             f"Index delta is smaller than buffer counter {len(buffer_data[idx_label])}")
            data_per_key = buffer_data[idx_label][0::index_delta]
            next_data.append( data_per_key)

        deltaTime = delta_time_data[index_delta] if index_delta < len(delta_time_data) else 0

    # ToDo get delta times as array for X axis
    return next_data, deltaTime, complete_frame_counter

def get_labels_from_osc(key_subset=None):

    if key_subset is None:
        osc_powerband_avg_labels = [ key for key in osc_server.OSC_PATHS_TO_KEY.values()]
    else:
        osc_powerband_avg_labels = [ key for key in key_subset
            if key in osc_server.OSC_PATHS_TO_KEY.values()]
    return osc_powerband_avg_labels


def osc_labels_data_and_deltaTime(key_subset=None):
    global last_processed_counter
    labels = get_labels_from_osc(key_subset)

    data_per_key, deltaTime, complete_frame_counter = _fetch_complete_data_from_server(labels, last_processed_counter)
    last_processed_counter = complete_frame_counter

    return labels, data_per_key, deltaTime

    data_per_key = osc_server.get_biometrics_dataframes().get_latest_frames(TIMESTEPS_WINDOW)
    write_elapsed_time_till_start("Get Biometrics slice")
    graphs, deltaTime = get_graphs_and_deltaTime_from_slice(data_per_key)

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

def make_plot_range_subset(osc_labels):
    osc_limits = [osc_server.OSC_LIMITS[ osc_labels[idx]] for idx in range(len(osc_labels))]
        # group keys by different set to plot multiple plots per limit group
    osc_map_limitset_indices_to_osc_key = { idx : [].append(key) for idx, key in enumerate(osc_labels) if
            osc_server.OSC_LIMITS[osc_labels[idx]] == osc_limits[idy] for idy in range(len(osc_limits))}
        
    return osc_limits,osc_map_limitset_indices_to_osc_key

with dpg.window(label="Example dynamic plot", autosize=True):
    
    with dpg.tree_node(label="Digital Plots", default_open=True):
        
        plot_show = []

        time.sleep(1)
        osc_labels, osc_frame_dict, delta_time = osc_labels_data_and_deltaTime()      

        plot_show = { osc_label : True for osc_label in osc_labels}
        data_digital = { idx : [] for idx in range(len(osc_labels))}

        osc_limits, osc_map_limitset_index_to_osc_key = make_plot_range_subset(osc_labels)


        def change_val_in_index(dict, ind, val):
            key = list(dict.keys())[ind]
            dict[key] = val


        # ToDo initialize data_frames
        with dpg.group(horizontal=True):
            with dpg.group(horizontal=False):
                for label in osc_labels :
                    dpg.add_checkbox(label=label, callback=lambda s, a: change_val_in_index(plot_show, 0, a),
                                                 default_value=True)

            def setup_plot_with_limits_and_message_subset(osc_limit_index):

                osc_plot_limits = osc_limits[osc_limit_index]
                osc_subset_labels = osc_map_limitset_index_to_osc_key[osc_limit_index]

                tag_plot = f"_bmi_plot_{osc_limit_index}"
                tag_x_axis = f"_bmi_plot_x_time_{osc_limit_index}"
                tag_y_axis = f"_bmi_plot_y_axis_{osc_limit_index}"

                with dpg.plot(tag=tag_plot, width=PLOT_WIDTH):
                        # X axis
                    dpg.add_plot_axis(dpg.mvXAxis, label=tag_x_axis, tag=tag_x_axis)
                    dpg.set_axis_limits(dpg.last_item(), -5, 0)
                    with dpg.plot_axis(dpg.mvYAxis, label=tag_y_axis):
                        dpg.set_axis_limits(dpg.last_item(), osc_plot_limits[0], osc_plot_limits[1])
                        for x_label in osc_subset_labels:
                            pass
                            dpg.add_line_series([], [], label=x_label, tag=x_label)


                        def _update_plot(key_subset: list[str]):
                            global t_digital_plot
                            t_digital_plot += dpg.get_delta_time()
                            dpg.set_axis_limits(tag_x_axis, t_digital_plot - 5, t_digital_plot)
                            osc_new_labels, osc_new_data, newDeltaTime = osc_labels_data_and_deltaTime(key_subset=key_subset)
                            #osc_limits, osc_map_limitset_index_to_osc_key = make_plot_range_subset(osc_new_labels)

                            assert(len(osc_new_labels) == len(osc_new_data))

                            if len(osc_new_labels) > 0:
                                for idx, sub_label in enumerate(osc_new_labels):
                                    assert( sub_label in data_digital.keys() and sub_label in plot_show.keys())
                                    if plot_show[sub_label]:
                                        x = np.linspace(t_digital_plot - newDeltaTime, t_digital_plot, len(osc_new_data[idx]))[-1]
                                        y = osc_new_data[idx][-1]
                                        data_digital[sub_label].append([x,y])
                                        dpg.set_value(sub_label,  [*zip(*data_digital[idx])])

                        handler_tag_ref = f"_plot_ref_{osc_limit_index}"
                        handler_tag = f"_plot_handler_{osc_limit_index}"
                        with dpg.item_handler_registry(tag=handler_tag_ref):
                            dpg.add_item_visible_handler(callback=_update_plot(osc_subset_labels))
                        dpg.bind_item_handler_registry(handler_tag, dpg.last_container())

            for plot_subset_idx in range(len(osc_limits)):
                setup_plot_with_limits_and_message_subset(plot_subset_idx)
                    






dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()