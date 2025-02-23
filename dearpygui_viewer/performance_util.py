import time

start_time = time.time()

def write_elapsed_time_till_start(timedesc = "") -> str:
    timeElapsed = time.time() - start_time
    return ( timedesc + ": Time elapsed in [s]: ", timeElapsed)