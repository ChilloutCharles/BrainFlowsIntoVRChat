from protected_osc_buffer import ProtectedOSCBuffer
import time

class MLActionsBuffer:

    def __init__(self, num_actions, max_stored_timesteps):
        self.action_buffers = {}
        self._key_actions_dict = {}
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
        for i in range(max_actions):
            paths.append("/avatar/parameters/BFI/Action" + str(i))
        return paths
        
    def _make_buffers(self, num_actions):

        paths  = self.generate_action_paths(num_actions)
        self._key_actions_dict = { path : "Action" + str(i) for i, path in enumerate(paths)}

        for key in self._key_actions_dict.values():
            self.action_buffers[key] = ProtectedOSCBuffer(self.max_stored_timesteps)
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

    def get_action_limits(self):
        return (0.0, 1.0)