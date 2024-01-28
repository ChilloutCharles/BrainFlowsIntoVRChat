from reporters.base_reporter import Base_Reporter

from pythonosc.udp_client import SimpleUDPClient

from constants import OSC_BASE_PATH

class OSC_Reporter(Base_Reporter):
    def __init__(self, ip, send_port):
        self.osc_client = SimpleUDPClient(ip, send_port)

    def flatten(self, root_path, data_dict):
        pairs = []
        for param_name in data_dict:
            param_path = '/'.join((root_path, param_name))
            param_value = data_dict[param_name]

            if type(param_value) != dict:
                pair = (param_path, param_value)
                to_extend = [pair]
            else:
                to_extend = self.flatten(param_path, param_value)
            pairs.extend(to_extend)
        
        return pairs
    
    def send(self, data_dict):
        # flatten dictionary into a list of pairs
        send_pairs = self.flatten(OSC_BASE_PATH + "brainflow", data_dict)

        # send each pair
        for path, value in send_pairs:
            self.osc_client.send_message(path, value)
        
        return send_pairs



        
