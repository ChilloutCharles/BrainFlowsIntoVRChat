from pythonosc.dispatcher import Dispatcher
from pythonosc import osc_server, udp_client
from ml_actions_buffer import MLActionsBuffer
from protected_osc_buffer import ProtectedOSCBuffer
from collections import deque
import queue
import time

OSC_BMI_KEY_BASIS = "/avatar/parameters/BFI"
MAX_STORED_TIMESTEPS = 200

BMI_PATHS_TO_KEY = { 
    "/avatar/parameters/BFI/NeuroFB/FocusLeft": "NeuroFB_FocusLeft",
    "/avatar/parameters/BFI/NeuroFB/FocusRight": "NeuroFB_FocusRight",
    "/avatar/parameters/BFI/NeuroFB/FocusAvg": "NeuroFB_FocusAvg",
    "/avatar/parameters/BFI/NeuroFB/RelaxLeft": "NeuroFB_RelaxLeft",
    "/avatar/parameters/BFI/NeuroFB/RelaxRight": "NeuroFB_RelaxRight",
    "/avatar/parameters/BFI/NeuroFB/RelaxAvg": "NeuroFB_RelaxAvg",
    "/avatar/parameters/BFI/PwrBands/Left/Delta": "PwrBands_Left_Delta",
    "/avatar/parameters/BFI/PwrBands/Left/Theta": "PwrBands_Left_Theta",
    "/avatar/parameters/BFI/PwrBands/Left/Alpha": "PwrBands_Left_Alpha",
    "/avatar/parameters/BFI/PwrBands/Left/Beta": "PwrBands_Left_Beta",
    "/avatar/parameters/BFI/PwrBands/Left/Gamma": "PwrBands_Left_Gamma",
    "/avatar/parameters/BFI/PwrBands/Right/Delta": "PwrBands_Right_Delta",
    "/avatar/parameters/BFI/PwrBands/Right/Theta": "PwrBands_Right_Theta",
    "/avatar/parameters/BFI/PwrBands/Right/Alpha": "PwrBands_Right_Alpha",
    "/avatar/parameters/BFI/PwrBands/Right/Beta": "PwrBands_Right_Beta",
    "/avatar/parameters/BFI/PwrBands/Right/Gamma": "PwrBands_Right_Gamma",
    "/avatar/parameters/BFI/PwrBands/Avg/Delta": "PwrBands_Avg_Delta",
    "/avatar/parameters/BFI/PwrBands/Avg/Theta": "PwrBands_Avg_Theta",
    "/avatar/parameters/BFI/PwrBands/Avg/Alpha": "PwrBands_Avg_Alpha",
    "/avatar/parameters/BFI/PwrBands/Avg/Beta": "PwrBands_Avg_Beta",
    "/avatar/parameters/BFI/PwrBands/Avg/Gamma": "PwrBands_Avg_Gamma",
    # Handle situation where message is not send
    "/avatar/parameters/BFI/Biometrics/HeartBeatsPerMinute": "Biometrics_HeartBeatsPerMinute",
    "/avatar/parameters/BFI/Biometrics/BreathsPerMinute": "Biometrics_BreathsPerMinute",
    "/avatar/parameters/BFI/Biometrics/OxygenPercent": "OxygenPercent"
}

BMI_KEY_TO_GRAPH_ID = {
    "NeuroFB_FocusLeft": "NeuroFB",
    "NeuroFB_FocusRight": "NeuroFB",
    "NeuroFB_FocusAvg": "NeuroFB",
    "NeuroFB_RelaxLeft": "NeuroFB",
    "NeuroFB_RelaxRight": "NeuroFB",
    "NeuroFB_RelaxAvg":  "NeuroFB",
    "PwrBands_Left_Delta": "PwrBands",
    "PwrBands_Left_Theta": "PwrBands",
    "PwrBands_Left_Alpha": "PwrBands",
    "PwrBands_Left_Beta": "PwrBands",
    "PwrBands_Left_Gamma":  "PwrBands",
    "PwrBands_Right_Delta": "PwrBands",
    "PwrBands_Right_Theta": "PwrBands",
    "PwrBands_Right_Alpha": "PwrBands",
    "PwrBands_Right_Beta": "PwrBands",
    "PwrBands_Right_Gamma": "PwrBands",
    "PwrBands_Avg_Delta": "PwrBands",
    "PwrBands_Avg_Theta": "PwrBands",
    "PwrBands_Avg_Alpha": "PwrBands",
    "PwrBands_Avg_Beta": "PwrBands",
    "PwrBands_Avg_Gamma": "PwrBands",
    "Biometrics_HeartBeatsPerMinute": "Biometrics",
    "Biometrics_BreathsPerMinute": "Biometrics",
    "OxygenPercent": "BiometricsPercent"
}

GRAPH_ID_TO_LIMITS = {
    "NeuroFB": (-1.0, 1.0),
    "PwrBands": (0.0, 1.0),
    "Biometrics": (0.0, 255.0),
    "BiometricsPercent": (0.0, 100.0)   
}

osc_buffers = { key : ProtectedOSCBuffer(MAX_STORED_TIMESTEPS) for path, key in BMI_PATHS_TO_KEY.items() }
forward_queue = queue.Queue()

def write_to_osc_buffer(path, value):
    osc_buffers[path].lock.acquire()
    osc_buffers[path].deque.append( (value, time.time()))
    osc_buffers[path].lock.release()

def read_last_from_osc_buffer(path):
    osc_buffers[path].lock.acquire()
    # reversed makes sure that the latest data is at the end of the list
    data = list(osc_buffers[path].deque)[-1] # reverse the list
    osc_buffers[path].lock.release()
    return data

def _osc_message_data_handler(path, value):
    if path in BMI_PATHS_TO_KEY:
        write_to_osc_buffer(BMI_PATHS_TO_KEY[path], value)

def _osc_message_forward_handler(path, value):
    forward_queue.put( (path, value))

osc_forward_deque = deque(maxlen=MAX_STORED_TIMESTEPS)

def forward_messages(osc_ip, osc_port_listen, osc_port_forward):
    client = udp_client.SimpleUDPClient(osc_ip, osc_port_forward)
    try:
        while True:
            address, args = forward_queue.get()
            client.send_message(address, args)
            forward_queue.task_done()
    except queue.Empty:
        time.sleep(0.001)
    except KeyboardInterrupt:
            print("Shutting down forwarder...")


def run_buffer_server(osc_ip, osc_port_listen, osc_port_forward = None, ml_actions_buffer = None):
    dispatcher = Dispatcher()
    if osc_port_forward is not None:
        dispatcher.map( "/avatar/parameters/BFI/*", _osc_message_forward_handler)
    dispatcher.map("/avatar/parameters/BFI/*", _osc_message_data_handler)

    if ml_actions_buffer is not None:
        ignored_paths = []

        def _osc_message_action_handler(path, value):
            try :
                key = ml_actions_buffer.get_action_key(path)
                ml_actions_buffer.write_to_osc_ml_action_buffer(key, value)
            except KeyError:
                if path not in ignored_paths:
                    ignored_paths.append(key)
                    print("Key not found in action buffer, ignoring... " + path + "/"+ str(value))

        dispatcher.map("/avatar/parameters/BFI/MLAction*", _osc_message_action_handler)    

    server = osc_server.BlockingOSCUDPServer(
        (osc_ip, osc_port_listen), dispatcher)
    print("Serving on {}".format(server.server_address))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, stopping server(s).")
    finally:
        server.server_close()