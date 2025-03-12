from pythonosc import udp_client
from osc_server import OSC_PATHS_TO_KEY, ELAPSED_TIME_PATH
import argparse

import time

SLEEP_TIME = 0.08 #seconds

startTime = time.time()

osc_paths = [ key for key in OSC_PATHS_TO_KEY.keys()]

def run_client(port):
    client = udp_client.SimpleUDPClient("127.0.0.1", port)

    print(osc_paths)
    assert isinstance(osc_paths[0], str)

    def _get_debuggable_times(time = 0.0):
        return 0.75 if (time % 10.0) > 7.5 else 0.5 if (time % 10.0) > 5.0 else 0.25 if (time % 10.0) > 2.5 else 0.0

    try:
        while True:
            for idx, path in enumerate(osc_paths):
                client.send_message(path, _get_debuggable_times(time.time()))

            elapsedTime = time.time() - startTime
            client.send_message(ELAPSED_TIME_PATH, elapsedTime)
            time.sleep(SLEEP_TIME)
    except KeyboardInterrupt:
        print("Shutting down testing client...")
    finally :
        print("Done")

if __name__ == "__main__":

    ## Parse argument port
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9010, help="The port to listen on")
    args = parser.parse_args()

    run_client(port=args.port)  
        