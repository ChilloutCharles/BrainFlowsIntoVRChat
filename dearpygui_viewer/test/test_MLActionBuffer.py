import sys
import os

viewer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add viewer directory to sys.path
sys.path.append(viewer_path)

# Now you can import your module
import osc_server

import unittest

class TestMLActionBuffer(unittest.TestCase):

    def test_init_with_actions(self):
        buffer = osc_server.MLActionBuffer(3)
        self.assertEqual(buffer.num_actions, 3)
        self.assertEqual(buffer._key_actions_dict, {"/avatar/parameters/BFI/Action0": "Action0",
                                                      "/avatar/parameters/BFI/Action1": "Action1",
                                                      "/avatar/parameters/BFI/Action2": "Action2"})
        self.assertEqual(len(buffer.action_buffers), 3)

    def test_init_empty_behaviour(self):
        with self.assertRaises(ValueError):
            buffer = osc_server.MLActionBuffer(0)

    def test_init_negative_num_actions(self):
        with self.assertRaises(ValueError):
            osc_server.MLActionBuffer(-1)
    
    def test_init_non_integer_num_actions(self):
        with self.assertRaises(TypeError):
            osc_server.MLActionBuffer(1.5)
    
    def test_functioning_read_and_write(self):
        buffer = osc_server.MLActionBuffer(3)
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