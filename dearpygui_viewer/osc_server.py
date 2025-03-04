import threading

from pythonosc.dispatcher import Dispatcher
from pythonosc import osc_server
from collections import deque

OSC_KEY_BASIS = "/avatar/parameters/BFI"
MAX_STORED_TIMESTEPS = 20

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

class ProtectedOSCBuffer:
    def __init__(self, max_len = 50):
        self.lock = threading.Lock()
        self.countingDataPoints = 0
        self.deque = deque(maxlen=max_len)

osc_buffers = { key : ProtectedOSCBuffer(MAX_STORED_TIMESTEPS) for path, key in OSC_PATHS_TO_KEY.items() }
osc_elapsed_time_buffer = ProtectedOSCBuffer(MAX_STORED_TIMESTEPS)

def write_to_osc_buffer(path, value):
    osc_buffers[path].lock.acquire()
    osc_buffers[path].deque.append(value)
    osc_buffers[path].countingDataPoints += 1
    osc_buffers[path].lock.release()

def _write_to_osc_elapsed_time_buffer(deltatime):
    osc_elapsed_time_buffer.lock.acquire()
    x = osc_elapsed_time_buffer.deque[-1] if len(osc_elapsed_time_buffer.deque) > 0 else 0
    osc_elapsed_time_buffer.deque.append (x + deltatime)
    osc_elapsed_time_buffer.countingDataPoints += 1
    osc_elapsed_time_buffer.lock.release()

def read_from_osc_buffer(path):
    osc_buffers[path].lock.acquire()
    # reversed makes sure that the latest data is at the end of the list
    data = list(osc_buffers[path].deque)[::-1] # reverse the list
    idxCount = osc_buffers[path].countingDataPoints
    osc_buffers[path].lock.release()
    return data, idxCount

def read_last_from_osc_buffer(path) -> float:
    osc_buffers[path].lock.acquire()
    data = list(osc_buffers[path].deque)[-1]
    idxCount = osc_buffers[path].countingDataPoints
    osc_buffers[path].lock.release()
    return data, idxCount

def read_from_osc_buffer_elapsedTime():
    osc_elapsed_time_buffer.lock.acquire()
    data = list(osc_elapsed_time_buffer.deque)[-1] # reverse the list
    idxCount = osc_elapsed_time_buffer.countingDataPoints
    osc_elapsed_time_buffer.lock.release()
    return data, idxCount

def _osc_elapsed_time_handler(path, value):
    _write_to_osc_elapsed_time_buffer(value)

def _osc_message_data_handler(path, value):
    if path in OSC_PATHS_TO_KEY:
        write_to_osc_buffer(OSC_PATHS_TO_KEY[path], value)


def run_server(ip, port):
    dispatcher = Dispatcher()
    #dispatcher.map("/avatar/parameters/BFI/*", bfi_data_handler)
    dispatcher.map( ELAPSED_TIME_PATH, _osc_elapsed_time_handler)
    dispatcher.map("/avatar/parameters/BFI/*", _osc_message_data_handler)

    server = osc_server.BlockingOSCUDPServer(
        (ip, port), dispatcher)
    print("Serving on {}".format(server.server_address))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, stopping server.")
    finally:
        server.server_close()