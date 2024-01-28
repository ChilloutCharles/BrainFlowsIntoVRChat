from reporters.base_reporter import Base_Reporter

from pythonosc.udp_client import SimpleUDPClient

from constants import OSC_BASE_PATH


class Old_OSC_Reporter(Base_Reporter):
    def __init__(self, ip, send_port):
        self.osc_client = SimpleUDPClient(ip, send_port)

    def send(self, data_dict):
        # flatten dictionary into a list of pairs
        send_pairs = self.flatten(data_dict)
        send_pairs = [(OSC_BASE_PATH + k, v) for k, v in send_pairs]

        # send each pair
        for path, value in send_pairs:
            self.osc_client.send_message(path, value)
        
        return send_pairs

    def flatten(self, data_dict):
        func_dict = {
            'device' : self.flatten_telemetry,
            'respiration' : self.flatten_respiration,
            'neurofeedback' : self.flatten_neurofeedback,
            'power_ratios' : self.flatten_power_ratios,
            'addons': self.flatten_addons
        }
        list_of_pairs = [func(data_dict[k]) for k, func in func_dict.items() if k in data_dict]
        return sum(list_of_pairs, [])

    def flatten_addons(self, data_dict):
        return [("HueShift", data_dict["HueShift"])]

    def flatten_telemetry(self, data_dict):
        pairs = [ ("osc_" + k, v) for k, v in data_dict.items()]
        return pairs
    
    def flatten_respiration(self, data_dict):
        old_dict = {
            "osc_heart_bpm" : data_dict["heart_bpm"],
            "osc_heart_bps" : data_dict["heart_freq"],
            "osc_respiration_bpm" : data_dict["respiration_bpm"],
            "osc_respiration_bps" : data_dict["respiration_freq"],
            "osc_oxygen_percent" : data_dict["oxygen_percent"]
        }
        return list(old_dict.items())
    
    def flatten_neurofeedback(self, data_dict):
        pairs = []
        for location, scores_dict in data_dict.items():
            for score, value in scores_dict.items():
                param_name = "osc_{}_{}".format(score, location)
                pair = (param_name, value)
                pairs.append(pair)
        return pairs
    
    def flatten_power_ratios(self, data_dict):
        pairs = []
        for location, power_dict in data_dict.items():
            for power, value in power_dict.items():
                param_name = "osc_band_power_{}_{}".format(location, power)
                pair = (param_name, value)
                pairs.append(pair)
        return pairs