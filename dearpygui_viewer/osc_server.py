

from pythonosc.dispatcher import Dispatcher
from pythonosc import osc_server

from osc_dataframes import OSCFrameDeque, OSCFrameCollector

misc_collector = OSCFrameCollector()
neurofb_collector = OSCFrameCollector()
pwrbands_collector_left = OSCFrameCollector()
pwrbands_collector_right = OSCFrameCollector()
pwrbands_collector_avg = OSCFrameCollector()
biometrics_collector = OSCFrameCollector()

def bfi_data_handler_generic(path, value):
    misc_collector.process_osc_message(path, value)

def neurofb_data_handler(path, value):
    neurofb_collector.process_osc_message(path, value)

def pwrbands_data_handler_left(path, value):
    pwrbands_collector_left.process_osc_message(path, value)

def pwrbands_data_handler_right(path, value):
    pwrbands_collector_right.process_osc_message(path, value)

def pwrbands_data_handler_avg(path, value):
    pwrbands_collector_avg.process_osc_message(path, value)

def biometrics_data_handler(path, value):
    biometrics_collector.process_osc_message(path, value)

def bfi_time_handler(path, value):
    misc_collector.process_osc_deltatime(value)
    neurofb_collector.process_osc_deltatime(value)
    pwrbands_collector_left.process_osc_deltatime(value)
    pwrbands_collector_right.process_osc_deltatime(value)
    pwrbands_collector_avg.process_osc_deltatime(value)
    biometrics_collector.process_osc_deltatime(value)

def get_misc_dataframes() -> OSCFrameDeque:
    return misc_collector.get_osc_dataframes()

def get_neurofb_dataframes() -> OSCFrameDeque:
    return neurofb_collector.get_osc_dataframes()

def get_pwrbands_dataframes_left() -> OSCFrameDeque:
    return pwrbands_collector_left.get_osc_dataframes()

def get_pwrbands_dataframes_right() -> OSCFrameDeque:
    return pwrbands_collector_right.get_osc_dataframes()

def get_pwrbands_dataframes_avg() -> OSCFrameDeque:
    return pwrbands_collector_avg.get_osc_dataframes()

def get_biometrics_dataframes() -> OSCFrameDeque:
    return biometrics_collector.get_osc_dataframes()

def run_server(ip, port):
    dispatcher = Dispatcher()
    #dispatcher.map("/avatar/parameters/BFI/*", bfi_data_handler)
    dispatcher.map("/avatar/parameters/BFI/Info/SecondsSinceLastUpdate", bfi_time_handler)
    dispatcher.map("/avatar/parameters/BFI/NeuroFB/*", neurofb_data_handler)
    dispatcher.map("/avatar/parameters/BFI/PwrBands/Left/*", pwrbands_data_handler_left)
    dispatcher.map("/avatar/parameters/BFI/PwrBands/Right/*", pwrbands_data_handler_right)
    dispatcher.map("/avatar/parameters/BFI/PwrBands/Avg/*", pwrbands_data_handler_avg)
    dispatcher.map("/avatar/parameters/BFI/Biometrics/*", biometrics_data_handler)
    

    server = osc_server.BlockingOSCUDPServer(
        (ip, port), dispatcher)
    print("Serving on {}".format(server.server_address))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, stopping server.")
    finally:
        server.server_close()