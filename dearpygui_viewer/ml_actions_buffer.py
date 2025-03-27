from protected_osc_buffer import ProtectedOSCBuffer
import time

class MLActionsBuffer:

    def __init__(self, num_actions, max_stored_timesteps):
        self.action_buffers = {}
        self._action_paths_to_key = {}
        self.num_actions = 0
        self.max_stored_timesteps = max_stored_timesteps
        if num_actions <= 0:
            raise ValueError("num_actions must be a non-negative integer.")
        if num_actions > 16: 
            num_actions = 16
            print("viewer does not support more than 16 actions, reducing to 16...")
        self.num_actions = num_actions
        self._make_buffers(num_actions)

    def generate_action_paths(self, max_actions):
        paths = []
        paths.append("/avatar/parameters/BFI/MLAction/Action")
        for i in range(max_actions):
            paths.append("/avatar/parameters/BFI/MLAction/Action" + str(i))
        return paths
        
    def _make_buffers(self, num_actions):

        paths  = self.generate_action_paths(num_actions)
        self._action_paths_to_key = { path : "Action" + str(i - 1) for i, path in enumerate(paths)}
        self._action_paths_to_key["/avatar/parameters/BFI/MLAction/Action"] = "Action" # todo refactor generation of paths

        for key in self._action_paths_to_key.values():
            self.action_buffers[key] = ProtectedOSCBuffer(self.max_stored_timesteps)
            self.action_buffers[key].deque.append((0.0, 0.0))

    def get_action_key(self, path):
        return self._action_paths_to_key[path] # current selected action  # todo review

    def read_from_osc_ml_action_buffer(self, key):
        self.action_buffers[key].lock.acquire()
        data = list(self.action_buffers[key].deque)[-1]
        self.action_buffers[key].lock.release()
        return data

    def write_to_osc_ml_action_buffer(self, key, value):
        self.action_buffers[key].lock.acquire()
        self.action_buffers[key].deque.append((value, time.time()))
        self.action_buffers[key].lock.release()

    def get_action_limits(self):
        return (0.0, 1.0)