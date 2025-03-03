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
            data_per_key = buffer_data[idx_label][0::index_delta]
            next_data.append( data_per_key)

        deltaTime = delta_time_data[index_delta-1] if index_delta-1 < len(delta_time_data) else 0

    # ToDo get delta times as array for X axis
    return next_data, deltaTime, complete_frame_counter

def get_labels_from_osc():

    osc_powerband_avg_labels = [ key for key in osc_server.OSC_PATHS_TO_KEY.values()]

    return osc_powerband_avg_labels


def osc_labels_data_and_deltaTime():
    global last_processed_counter
    labels = get_labels_from_osc()

    dataArrayOfPerKeyValues, deltaTime, complete_frame_counter = _fetch_complete_data_from_server(labels, last_processed_counter)
    last_processed_counter = complete_frame_counter

    return labels, dataArrayOfPerKeyValues, deltaTime


def save_callback():
    print("Save Clicked")

dpg.create_context()
dpg.create_viewport( title="Dearpygui Viewer")
dpg.setup_dearpygui()

def unique_range_to_sublabels(osc_labels):
    # osc_server.OSC_LIMITS[ key] is a tuple
    osc_unique_limits = set([ osc_server.OSC_LIMITS[key] for key in osc_labels])
    dict_unique_range_to_labels = { limit : [] for limit in osc_unique_limits}
    for key in osc_labels:
        osc_limits = osc_server.OSC_LIMITS[key]
        dict_unique_range_to_labels[osc_limits].append(key)
    
    return dict_unique_range_to_labels


with dpg.window(label="Example dynamic plot", autosize=True):
    
    with dpg.tree_node(label="Digital Plots", tag="Digital Plots", default_open=True):
        

        time.sleep(1)
        osc_labels, osc_frame_dict, delta_time = osc_labels_data_and_deltaTime()      

        plot_show = { osc_label : True for osc_label in osc_labels}
        data_digital = { osc_label : [] for osc_label in osc_labels}

        unique_range_to_sublabels = unique_range_to_sublabels(osc_labels)

        def change_val_in_dict(sender, key, val):
            plot_show[key] = val
            print(f"Changed {key} to {val}")

        # ToDo initialize data_frames
        with dpg.group(horizontal=False):
            
            def setup_plot_with_limits_and_message_subset(osc_limit_index, osc_plot_limits, osc_subset_labels):
                tag_plot = f"_bmi_plot_{osc_limit_index}"
                tag_x_axis = f"_bmi_plot_x_time_{osc_limit_index}"
                tag_y_axis = f"_bmi_plot_y_axis_{osc_limit_index}"

                with dpg.group(horizontal=True):
                    with dpg.plot(tag=tag_plot, width=PLOT_WIDTH):
                            # X axis
                        dpg.add_plot_axis(dpg.mvXAxis, label=tag_x_axis, tag=tag_x_axis)
                        dpg.set_axis_limits(dpg.last_item(), -5, 0)
                        with dpg.plot_axis(dpg.mvYAxis, label=tag_y_axis):
                            dpg.set_axis_limits(dpg.last_item(), osc_plot_limits[0], osc_plot_limits[1])
                            for x_label in osc_subset_labels:
                                dpg.add_line_series([], [], label=x_label, tag=x_label)

                    with dpg.group(horizontal=False):
                        print( osc_subset_labels)
                        for xlabel in osc_subset_labels:
                            # Todo get done as lamnda
                            def create_callback(current_label):
                                def callback(sender, app_data):
                                    change_val_in_dict(sender, current_label, app_data)
                                return callback
                            
                            dpg.add_checkbox(label=xlabel, 
                                             callback=create_callback(xlabel), 
                                             default_value=True)

                dpg.add_separator()

            def _update_plot():
                global t_digital_plot
                t_digital_plot += dpg.get_delta_time()
                osc_new_labels, osc_new_data, newDeltaTime = osc_labels_data_and_deltaTime()

                def _update_subplots(index, key_subset: list[str]):
                    tag_x_axis = f"_bmi_plot_x_time_{index}"
                    dpg.set_axis_limits(tag_x_axis, t_digital_plot - 5, t_digital_plot)

                    if len(osc_new_data) <= 0:
                        return

                    sub_new_labels = [ label for label in key_subset if label in osc_new_labels]
                    #ToDo Fix
                    x = [osc_new_labels.index(label) for label in sub_new_labels]
                    sub_new_data = [ osc_new_data[idx] for idx in x]

                    assert len(sub_new_labels) == len(sub_new_data)

                    if len(osc_new_labels) > 0:
                        for idx, sub_label in enumerate(sub_new_labels):
                            assert sub_label in data_digital.keys()
                            if plot_show[sub_label] and len(sub_new_data[idx]) > 0:
                                x = np.linspace(t_digital_plot - newDeltaTime, t_digital_plot, len(sub_new_data[idx]))[-1]
                                y = sub_new_data[idx][-1]
                                data_digital[sub_label].append([x,y])
                                dpg.set_value(sub_label,  [*zip(*data_digital[sub_label])])


                for plot_subset_idx, (key, value) in enumerate(unique_range_to_sublabels.items()):
                    _update_subplots(plot_subset_idx, value)

            tag_plot_first = f"_bmi_plot_{0}"

            for plot_subset_idx, (key, value) in enumerate(unique_range_to_sublabels.items()):
                setup_plot_with_limits_and_message_subset(plot_subset_idx, key, value)

            with dpg.item_handler_registry(tag="handler_tag_ref"):
                dpg.add_item_visible_handler(callback=_update_plot)
            dpg.bind_item_handler_registry(tag_plot_first, dpg.last_container())

        

                
                    






dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()