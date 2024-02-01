from reporters.base_reporter import Base_Reporter

from pythonosc.udp_client import SimpleUDPClient

from constants import OSC_BASE_PATH, BFI_ROOT

class OSC_Reporter(Base_Reporter):
    def __init__(self, ip, send_port):
        self.osc_client = SimpleUDPClient(ip, send_port)

    def flatten(self, data_dict, root_path=""):
        pairs = []
        for param_name in data_dict:
            param_path = '/'.join((root_path, param_name))
            param_value = data_dict[param_name]

            if type(param_value) != dict:
                pair = (param_path, param_value)
                to_extend = [pair]
            else:
                to_extend = self.flatten(param_value, param_path)
            pairs.extend(to_extend)
        
        return pairs
    
    def send(self, data_dict):
        # flatten dictionary into a list of pairs
        send_pairs = self.flatten(data_dict, OSC_BASE_PATH + BFI_ROOT)

        # send each pair
        for path, value in send_pairs:
            self.osc_client.send_message(path, value)
        
        return send_pairs

class Debug_Reporter(OSC_Reporter):
    def __init__(self, ip, send_port):
        super().__init__(ip, send_port)
    
    def send(self, data_dict):
        # flatten dictionary into a list of pairs
        send_pairs = self.flatten(data_dict)

        # send each pair
        for path, value in send_pairs:
            self.osc_client.send_message(path, value)
        
        return send_pairs