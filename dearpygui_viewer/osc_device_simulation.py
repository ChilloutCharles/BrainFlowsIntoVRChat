from pythonosc import udp_client
from osc_server import BMI_PATHS_TO_KEY
from ml_actions_buffer import MLActionsBuffer
import argparse

import time

SLEEP_TIME = 0.08 #seconds

startTime = time.time()

osc_paths = [ key for key in BMI_PATHS_TO_KEY.keys()]

def run_client(port, num_ml_actions=0):
    client = udp_client.SimpleUDPClient("127.0.0.1", port)

    print(osc_paths)
    assert isinstance(osc_paths[0], str)

    action_paths = []
    if num_ml_actions > 0:
        action_paths = MLActionsBuffer(num_ml_actions, 10).generate_action_paths(num_ml_actions)
        print (action_paths)

    def _get_debuggable_times(time = 0.0):
        return 0.75 if (time % 10.0) > 7.5 else 0.5 if (time % 10.0) > 5.0 else 0.25 if (time % 10.0) > 2.5 else 0.0

    try:
        while True:
            for idx, path in enumerate(osc_paths):
                client.send_message(path, _get_debuggable_times(time.time()))

            for idx, path in enumerate(action_paths):
                client.send_message(path, _get_debuggable_times(time.time()))

            elapsedTime = time.time() - startTime
            time.sleep(SLEEP_TIME)
    except KeyboardInterrupt:
        print("Shutting down testing client...")
    finally :
        print("Done")

if __name__ == "__main__":

    ## Parse argument port
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9010, help="The port to listen on")
    parser.add_argument("--actions", type=int, default=0, help="Number of actions of the intent model trained", required=False)
    args = parser.parse_args()

    run_client(port=args.port, num_ml_actions=args.actions)  
        