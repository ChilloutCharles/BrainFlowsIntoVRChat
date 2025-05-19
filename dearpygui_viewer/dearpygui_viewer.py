import threading
import dearpygui.dearpygui as dpg
import time
import argparse
from collections import deque

import osc_server
from ml_actions_buffer import MLActionsBuffer

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

    ml_action_buffer = None
    if args.actions > 0:
        ml_action_buffer = MLActionsBuffer(args.actions, osc_server.MAX_STORED_TIMESTEPS)

    def run_osc_server(osc_ip, osc_port_listen, osc_port_forward, ml_action_buffer):
        osc_server.run_buffer_server(osc_ip, osc_port_listen, osc_port_forward, ml_action_buffer)  

    def run_osc_forward(osc_ip, osc_port_listen, osc_port_forward):
        osc_server.forward_messages(osc_ip, osc_port_listen, osc_port_forward)

    def start_servers_once(osc_ip, osc_port_listen, osc_port_forward, ml_action_buffer=None):
        global server_started
        if not server_started:
            thread = threading.Thread(
                target=run_osc_server,
                args=(osc_ip, osc_port_listen, osc_port_forward, ml_action_buffer),
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

    start_servers_once(args.ip, args.port_listen, args.port_forward, ml_action_buffer)


    def fetch_bmi_data_and_timestep_for_keys(osc_keys):

        osc_data_per_key = [ osc_server.read_last_from_osc_buffer(key) for key in osc_keys]

        next_data = {}

        if len(osc_data_per_key) > 0:
            for idx_label, label in enumerate(osc_keys):
                data, time = osc_data_per_key[idx_label]
                next_data[label] = (data, time)


        return next_data

    def fetch_action_data_and_timestep_for_keys(ml_action_buffer=None):

        next_data = {}

        if ml_action_buffer is not None:
            for key in ml_action_buffer._action_paths_to_key.values():
                data, time = ml_action_buffer.read_from_osc_ml_action_buffer(key)
                next_data[key] = (data, time)

        return next_data

    def get_labels_from_osc():

        osc_powerband_avg_labels = [ key for key in osc_server.BMI_PATHS_TO_KEY.values()]

        return osc_powerband_avg_labels

    def make_dict_subgraphkey_to_limit_and_label(osc_labels):
        # osc_server.OSC_LIMITS[ key] is a tuple
        graph_id_to_limits = osc_server.GRAPH_ID_TO_LIMITS
        dict_range_to_labels = { subid : (limit,[]) for subid, limit in graph_id_to_limits.items()}
        for label in osc_labels:
            graph_id = osc_server.BMI_KEY_TO_GRAPH_ID[label]
            dict_range_to_labels[graph_id] = (dict_range_to_labels[graph_id][0], dict_range_to_labels[graph_id][1] + [label])
        
        return dict_range_to_labels

    def make_dict_action_subgraph(ml_action_buffer):
        if ml_action_buffer is None:
            return {}
        return { "Actions" : ( ml_action_buffer.get_action_limits(), list(ml_action_buffer._action_paths_to_key.values())) }


    dpg.create_context()
    dpg.create_viewport(title="Dynamic BFiVRC Plot")
    dpg.setup_dearpygui()

    window_tag = "window_tag"
    with dpg.window(label="Dynamic BFiVRC Plot", autosize=True, tag=window_tag):
        
        with dpg.tree_node(label="Live Plot", tag="Digital Plots", default_open=True):
            
            time.sleep(1)
            osc_labels = get_labels_from_osc()
            action_labels = list(ml_action_buffer._action_paths_to_key.values() if ml_action_buffer is not None else [])

            plot_show = { osc_label : True for osc_label in osc_labels + action_labels}
            data_digital = { osc_label : deque(maxlen=DEQUEUE_SIZE) for osc_label in osc_labels + action_labels}

            make_dict_subgraph_to_limit_and_label = make_dict_subgraphkey_to_limit_and_label(osc_labels)
            make_dict_action_subgraph_to_limit_and_label = make_dict_action_subgraph(ml_action_buffer)

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

                                    

                                if xlabel == 'Action':
                                    dpg.add_text(" Selected Action is <>",
                                    label="SelectedActionText", tag="SelectedActionText")
                                   

                    dpg.add_separator()

                def _update_plot():
                    global t_count_plots
                    t_count_plots += 1
                    t_digital_plot = time.time() - time_viewer_started

                    def _update_subplots(index, key_subset: list[str], new_data_dict):
                        tag_x_axis = f"_bmi_plot_x_time_{index}"

                        dpg.set_axis_limits(tag_x_axis, t_digital_plot - 5, t_digital_plot)

                        if len(key_subset) > 0:
                            for idx, sub_label in enumerate(key_subset):
                                assert sub_label in data_digital.keys()
                                if plot_show[sub_label] :

                                    #time relative to t_digital_plor
                                    x_relate = new_data_dict[sub_label][1] - time_viewer_started
                                    y = new_data_dict[sub_label][0]
                                    data_digital[sub_label].append([x_relate,y])
                                    dpg.set_value(sub_label,  [*zip(*data_digital[sub_label])])

                                    if sub_label == 'Action': #todo refactor none check
                                        value = y
                                        dpg.set_value("SelectedActionText", "Selected Action is " + str(value))
                                    
                    for plot_subset_idx, (key, value) in enumerate(make_dict_subgraph_to_limit_and_label.items()):
                        (limit, key_subset) = value
                        new_data_dict = fetch_bmi_data_and_timestep_for_keys(key_subset)
                        _update_subplots(plot_subset_idx, key_subset, new_data_dict)

                    for plot_subset_idx, (key, value) in enumerate(make_dict_action_subgraph_to_limit_and_label.items()):
                        (limit, key_subset) = value
                        new_data_dict = fetch_action_data_and_timestep_for_keys(ml_action_buffer)
                        _update_subplots(plot_subset_idx + len(make_dict_subgraph_to_limit_and_label.keys()), key_subset, new_data_dict)

                    time.sleep(0.001) #reduce stress caused by readers
                    # end of function _update_plot 

                for plot_subset_idx, (key, value) in enumerate(make_dict_subgraph_to_limit_and_label.items()):
                    (limit, key_subset) = value
                    setup_plot_with_limits_and_message_subset(plot_subset_idx, limit, key_subset)

                for plot_subset_idx, (key, value) in enumerate(make_dict_action_subgraph_to_limit_and_label.items()):
                    (limit, key_subset) = value
                    setup_plot_with_limits_and_message_subset(plot_subset_idx + len(make_dict_subgraph_to_limit_and_label.keys()), limit, key_subset)

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
    parser.add_argument("--actions", type=int, default=0, help="Set number of actions to be viewed (max 16)", required=False)
    args = parser.parse_args()
    main(args)