

from pythonosc.dispatcher import Dispatcher
from pythonosc import osc_server

from osc_dataframes import OSCFrameDeque, OSCFrameCollector

osc_frame_collector = OSCFrameCollector()

def bfi_data_handler(path, value):
    osc_frame_collector.process_osc_message(path, value)

def bfi_time_handler(path, value):
    osc_frame_collector.process_osc_deltatime(value)

def get_dataframes() -> OSCFrameDeque:
    return osc_frame_collector.get_osc_dataframes()

def run_server(ip, port):
    dispatcher = Dispatcher()
    dispatcher.map("/avatar/parameters/BFI/*", bfi_data_handler)
    dispatcher.map("/avatar/parameters/BFI/Info/SecondsSinceLastUpdate", bfi_time_handler)
    

    server = osc_server.BlockingOSCUDPServer(
        (ip, port), dispatcher)
    print("Serving on {}".format(server.server_address))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, stopping server.")
    finally:
        server.server_close()