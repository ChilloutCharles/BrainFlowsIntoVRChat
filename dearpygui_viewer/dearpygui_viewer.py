import threading
import dearpygui.dearpygui as dpg
import time
import argparse
from collections import deque

import osc_server

PLOT_WIDTH = 800
DEQUEUE_SIZE = 1024*2
EPS = 0.01

#state         
server_started = False
time_viewer_started = time.time()

t_digital_plot = 0
t_count_plots = 0

# --------------------------------------------
def main(args):
    def run_osc_server(osc_ip, osc_port_listen, osc_port_forward):
        osc_server.run_server(osc_ip, osc_port_listen, osc_port_forward)  

    def run_osc_forward(osc_ip, osc_port_listen, osc_port_forward):
        osc_server.forward_messages(osc_ip, osc_port_listen, osc_port_forward)

    def start_servers_once(osc_ip, osc_port_listen, osc_port_forward):
        global server_started
        if not server_started:
            thread = threading.Thread(
                target=run_osc_server,
                args=(osc_ip, osc_port_listen, osc_port_forward),
                daemon=True
            )
            thread.start()
            print("Server started")

            if osc_port_forward is not None:
                thread_forward = threading.Thread(
                    target=run_osc_forward,
                    args=(osc_ip, osc_port_listen, osc_port_forward),
                    daemon=True
                )
                thread_forward.start()
                print("Forwarding started")

            server_started = True

    start_servers_once(args.ip, args.port_listen, args.port_forward)


    def fetch_set_label_data_and_timestep_for_keys(osc_keys):

        osc_data_per_key = [ osc_server.read_last_from_osc_buffer(key) for key in osc_keys]

        next_data = {}

        if len(osc_data_per_key) > 0:
            for idx_label, label in enumerate(osc_keys):
                data, time = osc_data_per_key[idx_label]
                next_data[label] = (data, time)

        return next_data

    def get_labels_from_osc():

        osc_powerband_avg_labels = [ key for key in osc_server.OSC_PATHS_TO_KEY.values()]

        return osc_powerband_avg_labels


    dpg.create_context()
    dpg.create_viewport( title="Dynamic BFiVRC Plot")
    dpg.setup_dearpygui()

    #todo get sublabelset from osc_server.py
    def graph_sublabels(osc_labels):
        # osc_server.OSC_LIMITS[ key] is a tuple
        osc_unique_limits = set([ osc_server.OSC_LIMITS[key] for key in osc_labels])
        dict_unique_range_to_labels = { limit : [] for limit in osc_unique_limits}
        for key in osc_labels:
            osc_limits = osc_server.OSC_LIMITS[key]
            dict_unique_range_to_labels[osc_limits].append(key)
        
        return dict_unique_range_to_labels

    window_tag = "window_tag"
    with dpg.window(label="Dynamic BFiVRC Plot", autosize=True, tag=window_tag):
        
        with dpg.tree_node(label="Live Plot", tag="Digital Plots", default_open=True):
            

            time.sleep(1)
            osc_labels = get_labels_from_osc()
            osc_data_dict = fetch_set_label_data_and_timestep_for_keys(osc_labels)      

            plot_show = { osc_label : True for osc_label in osc_labels}
            data_digital = { osc_label : deque(maxlen=DEQUEUE_SIZE) for osc_label in osc_labels}

            graph_sublabels = graph_sublabels(osc_labels)

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
                            dpg.add_plot_legend()
                                # X axis
                            dpg.add_plot_axis(dpg.mvXAxis, label=tag_x_axis, tag=tag_x_axis)
                            dpg.set_axis_limits(dpg.last_item(), -5, 0)
                            with dpg.plot_axis(dpg.mvYAxis, label=tag_y_axis):
                                dpg.set_axis_limits(dpg.last_item(), osc_plot_limits[0] - EPS, osc_plot_limits[1]+ EPS)
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
                    global t_count_plots
                    t_count_plots += 1
                    t_digital_plot = time.time() - time_viewer_started
                    osc_new_data_dict = fetch_set_label_data_and_timestep_for_keys(osc_labels) 

                    if len(osc_new_data_dict) <= 0:
                            return

                    def _update_subplots(index, key_subset: list[str]):
                        tag_x_axis = f"_bmi_plot_x_time_{index}"
                        
                        dpg.set_axis_limits(tag_x_axis, t_digital_plot - 5, t_digital_plot)

                        sub_new_labels = [ label for label in key_subset if label in osc_new_data_dict.keys()]

                        if len(sub_new_labels) > 0:
                            for idx, sub_label in enumerate(sub_new_labels):
                                assert sub_label in data_digital.keys()
                                if plot_show[sub_label] :
                                    #time relative to t_digital_plor
                                    x_relate = osc_new_data_dict[sub_label][1] - time_viewer_started
                                    y = osc_new_data_dict[sub_label][0]
                                    data_digital[sub_label].append([x_relate,y])
                                    dpg.set_value(sub_label,  [*zip(*data_digital[sub_label])])
                                    

                    for plot_subset_idx, (key, value) in enumerate(graph_sublabels.items()):
                        _update_subplots(plot_subset_idx, value)

                for plot_subset_idx, (key, value) in enumerate(graph_sublabels.items()):
                    setup_plot_with_limits_and_message_subset(plot_subset_idx, key, value)

                with dpg.item_handler_registry(tag="handler_tag_ref"):
                    dpg.add_item_visible_handler(callback=_update_plot)
                dpg.bind_item_handler_registry(window_tag, dpg.last_container())

    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP address for OSC messages")
    parser.add_argument("--port_listen", type=int, default=9010, help="The port to listen on")
    parser.add_argument("--port_forward", type=int, default=9000, help="The port to forward the data", required=False)
    args = parser.parse_args()
    main(args)