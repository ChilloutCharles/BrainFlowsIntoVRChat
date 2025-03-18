import threading

from pythonosc.dispatcher import Dispatcher
from pythonosc import osc_server, udp_client
from collections import deque
import queue
import time

OSC_KEY_BASIS = "/avatar/parameters/BFI"
MAX_STORED_TIMESTEPS = 200

ELAPSED_TIME_PATH = "/avatar/parameters/BFI/Info/SecondsSinceLastUpdate"

OSC_PATHS_TO_KEY = { 
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


class MLActionBuffer:

    def __init__(self, num_actions):
        self.action_buffers = {}
        self._key_actions_dict = {}
        self.num_actions = 0
        if num_actions <= 0:
            raise ValueError("num_actions must be a non-negative integer.")
        else:
            self.num_actions = num_actions
            self._make_buffers(num_actions - 1) 
        
    def _make_buffers(self, num_actions):

        pattern = "/avatar/parameters/BFI/Action"
        self._key_actions_dict = { pattern + str(i) : "Action" + str(i) for i in range(num_actions + 1)}

        for key in self._key_actions_dict.values():
            self.action_buffers[key] = ProtectedOSCBuffer(MAX_STORED_TIMESTEPS)
            self.action_buffers[key].deque.append((0.0, 0.0))

    def read_from_osc_ml_action_buffer(self, path):
        self.action_buffers[path].lock.acquire()
        data = list(self.action_buffers[path].deque)[-1]
        self.action_buffers[path].lock.release()
        return data

    def write_to_osc_ml_action_buffer(self, path, value):
        self.action_buffers[path].lock.acquire()
        self.action_buffers[path].deque.append((value, time.time()))
        self.action_buffers[path].lock.release()


OSC_LIMITS = {
    "NeuroFB_FocusLeft": (-1.0, 1.0),
    "NeuroFB_FocusRight": (-1.0, 1.0),
    "NeuroFB_FocusAvg": (-1.0, 1.0),
    "NeuroFB_RelaxLeft": (-1.0, 1.0),
    "NeuroFB_RelaxRight": (-1.0, 1.0),
    "NeuroFB_RelaxAvg": (-1.0, 1.0),
    "PwrBands_Left_Delta": (0.0, 1.0),
    "PwrBands_Left_Theta": (0.0, 1.0),
    "PwrBands_Left_Alpha": (0.0, 1.0),
    "PwrBands_Left_Beta": (0.0, 1.0),
    "PwrBands_Left_Gamma": (0.0, 1.0),
    "PwrBands_Right_Delta": (0.0, 1.0),
    "PwrBands_Right_Theta": (0.0, 1.0),
    "PwrBands_Right_Alpha": (0.0, 1.0),
    "PwrBands_Right_Beta": (0.0, 1.0),
    "PwrBands_Right_Gamma": (0.0, 1.0),
    "PwrBands_Avg_Delta": (0.0, 1.0),
    "PwrBands_Avg_Theta": (0.0, 1.0),
    "PwrBands_Avg_Alpha": (0.0, 1.0),
    "PwrBands_Avg_Beta": (0.0, 1.0),
    "PwrBands_Avg_Gamma": (0.0, 1.0),
    "Biometrics_HeartBeatsPerMinute": (0.0, 255.0),
    "Biometrics_BreathsPerMinute": (0.0, 255.0),
    "OxygenPercent": (0.0, 100.0)   
}

OSC_ACTION_LIMIT = (0.0, 1.0)

class ProtectedOSCBuffer:
    def __init__(self, max_len = 50):
        self.lock = threading.Lock()
        self.deque = deque(maxlen=max_len)

osc_buffers = { key : ProtectedOSCBuffer(MAX_STORED_TIMESTEPS) for path, key in OSC_PATHS_TO_KEY.items() }
forward_queue = queue.Queue()

for osc_buffer in osc_buffers.values():
    osc_buffer.deque.append((0.0,0.0)) # vuffer shaped (value, time)

osc_elapsed_time_buffer = ProtectedOSCBuffer(MAX_STORED_TIMESTEPS)
osc_elapsed_time_buffer.deque.append(0.0)

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
    if path in OSC_PATHS_TO_KEY:
        write_to_osc_buffer(OSC_PATHS_TO_KEY[path], value)

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


def run_server(osc_ip, osc_port_listen, osc_port_forward = None, ml_actions_num = 0):
    dispatcher = Dispatcher()
    if osc_port_forward is not None:
        dispatcher.map( "/avatar/parameters/BFI/*", _osc_message_forward_handler)
    dispatcher.map("/avatar/parameters/BFI/*", _osc_message_data_handler)    
        

    server = osc_server.BlockingOSCUDPServer(
        (osc_ip, osc_port_listen), dispatcher)
    print("Serving on {}".format(server.server_address))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, stopping server(s).")
    finally:
        server.server_close()