import sys
import os

viewer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add viewer directory to sys.path
sys.path.append(viewer_path)

# Now you can import your module
import ml_actions_buffer as ml_actions_buffer

import unittest

class TestMLActionBuffer(unittest.TestCase):

    maxStoredTimesteps = 10

    def test_init_with_actions(self):
        buffer = ml_actions_buffer.MLActionsBuffer(3, self.maxStoredTimesteps)
        self.assertEqual(buffer.max_stored_timesteps, self.maxStoredTimesteps)
        self.assertEqual(buffer.num_actions, 3)
        self.assertEqual(buffer._action_paths_to_key, {
                                                      "/avatar/parameters/BFI/MLAction/Action": "Action",
                                                      "/avatar/parameters/BFI/MLAction/Action0": "Action0",
                                                      "/avatar/parameters/BFI/MLAction/Action1": "Action1",
                                                      "/avatar/parameters/BFI/MLAction/Action2": "Action2"})
        self.assertEqual(len(buffer.action_buffers), 4)

    def test_init_empty_behaviour(self):
        with self.assertRaises(ValueError):
            buffer = ml_actions_buffer.MLActionsBuffer(0,  self.maxStoredTimesteps)

    def test_init_negative_num_actions(self):
        with self.assertRaises(ValueError):
            ml_actions_buffer.MLActionsBuffer(-1,  self.maxStoredTimesteps)
    
    def test_init_non_integer_num_actions(self):
        with self.assertRaises(TypeError):
            ml_actions_buffer.MLActionsBuffer(1.5,  self.maxStoredTimesteps)

    def test_init_too_many_actions(self):
        buffer = ml_actions_buffer.MLActionsBuffer(17,  self.maxStoredTimesteps)
        self.assertEqual(buffer.num_actions, 16)
        self.assertEqual(buffer._action_paths_to_key, {"/avatar/parameters/BFI/MLAction/Action": "Action",
                                                      "/avatar/parameters/BFI/MLAction/Action0": "Action0",
                                                      "/avatar/parameters/BFI/MLAction/Action1": "Action1",
                                                      "/avatar/parameters/BFI/MLAction/Action2": "Action2",
                                                      "/avatar/parameters/BFI/MLAction/Action3": "Action3",
                                                      "/avatar/parameters/BFI/MLAction/Action4": "Action4",
                                                      "/avatar/parameters/BFI/MLAction/Action5": "Action5",
                                                      "/avatar/parameters/BFI/MLAction/Action6": "Action6",
                                                      "/avatar/parameters/BFI/MLAction/Action7": "Action7",
                                                      "/avatar/parameters/BFI/MLAction/Action8": "Action8",
                                                      "/avatar/parameters/BFI/MLAction/Action9": "Action9",
                                                      "/avatar/parameters/BFI/MLAction/Action10": "Action10",
                                                      "/avatar/parameters/BFI/MLAction/Action11": "Action11",
                                                      "/avatar/parameters/BFI/MLAction/Action12": "Action12",
                                                      "/avatar/parameters/BFI/MLAction/Action13": "Action13",
                                                      "/avatar/parameters/BFI/MLAction/Action14": "Action14",
                                                      "/avatar/parameters/BFI/MLAction/Action15": "Action15"})
        self.assertEqual(len(buffer.action_buffers), 17)
    
    def test_functioning_read_and_write(self):
        buffer = ml_actions_buffer.MLActionsBuffer(3,  self.maxStoredTimesteps)
        print(buffer.action_buffers)
        self.assertEqual(buffer.read_from_osc_ml_action_buffer("Action0")[0], 0.0)
        self.assertEqual(buffer.read_from_osc_ml_action_buffer("Action1")[0], 0.0)
        self.assertEqual(buffer.read_from_osc_ml_action_buffer("Action2")[0], 0.0)
        buffer.write_to_osc_ml_action_buffer("Action0", 1.0)
        buffer.write_to_osc_ml_action_buffer("Action1", 2.0)
        buffer.write_to_osc_ml_action_buffer("Action2", 3.0)
        buffer.write_to_osc_ml_action_buffer("Action2", 3.0)
        self.assertEqual(buffer.read_from_osc_ml_action_buffer("Action0")[0], 1.0)
        self.assertEqual(buffer.read_from_osc_ml_action_buffer("Action1")[0], 2.0)
        self.assertEqual(buffer.read_from_osc_ml_action_buffer("Action2")[0], 3.0)
        
if __name__ == '__main__':
    unittest.main()