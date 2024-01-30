from reporters.base_reporter import Base_Reporter
from pythonosc.udp_client import SimpleUDPClient

from constants import OSC_BASE_PATH

from logic.telemetry import Device
from logic.power_bands import PowerBands
from logic.neuro_feedback import NeuroFeedback
from logic.ppg import HeartRate, Respiration, Ppg
from logic.addons import Addons


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
            Device.__name__ : self.flatten_telemetry,
            NeuroFeedback.__name__ : self.flatten_neurofeedback,
            PowerBands.__name__ : self.flatten_power_bands,
            Addons.__name__ : self.flatten_addons,
            HeartRate.__name__ : self.flatten_heart_rate,
            Respiration.__name__ : self.flatten_respiration
        }
        list_of_pairs = [func(data_dict[k]) for k, func in func_dict.items() if k in data_dict]
        return sum(list_of_pairs, [])

    def flatten_addons(self, data_dict):
        return [("HueShift", data_dict["HueShift"])]

    def flatten_telemetry(self, data_dict):
        telemetry_map = {
            Device.BATTERYLEVEL : "osc_battery_lvl",
            Device.CONNECTED : "osc_is_connected",
            Device.TIME_DIFF : "osc_time_diff"
        }
        keys = telemetry_map.keys() & data_dict.keys()
        pairs = [ (telemetry_map[k], data_dict[k]) for k in keys]
        return pairs
    
    def flatten_respiration(self, data_dict):
        pairs = []
        if data_dict[Ppg.SUPPORTED]:
            old_dict = {
                "osc_respiration_bpm" : data_dict[Ppg.RESP_BPM],
                "osc_respiration_bps" : data_dict[Ppg.RESP_FREQ],
                "osc_oxygen_percent" : data_dict[Ppg.OXYGEN_PERCENT]
            }
            pairs =  list(old_dict.items())
        return pairs
        
    def flatten_heart_rate(self, data_dict):
        pairs = []
        if data_dict[Ppg.SUPPORTED]:
            old_dict = {
                "osc_heart_bpm" : data_dict[Ppg.HEART_BPM],
                "osc_heart_bps" : data_dict[Ppg.HEART_FREQ]
            }
            pairs = list(old_dict.items())
        return pairs
    
    def flatten_neurofeedback(self, data_dict):
        pairs = []
        for score_name, value_dict in data_dict.items():
            signed_dict = value_dict[NeuroFeedback.SIGNED]
            for location, value in signed_dict.items():
                param_name = "osc_{}_{}".format(score_name, location).lower()
                pair = (param_name, value)
                pairs.append(pair)
        return pairs
    
    def flatten_power_bands(self, power_dict):
        pairs = []
        for location, power_dict in power_dict.items():
            for power_name, value in power_dict.items():
                param_name = "osc_band_power_{}_{}".format(location, power_name).lower()
                pair = (param_name, value)
                pairs.append(pair)
        return pairs