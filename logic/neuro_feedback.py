from logic.power_bands import PowerBands
from constants import BAND_POWERS
from utils import tanh_normalize

class NeuroFeedback(PowerBands):
    FOCUS = "Focus"
    RELAX = "Relax"
    SIGNED = "Signed"
    UNSIGNED = "Unsigned"

    def __init__(self, board, window_seconds=2, normalize_scale=1.1, ema_decay=0.025):
        super().__init__(board, window_seconds=window_seconds, ema_decay=ema_decay)
        self.normalize_scale = normalize_scale

    def get_data_dict(self):
        power_dict = super().get_data_dict()

        # create functions for getting scores per location
        get_focus = lambda location: self.calculate_ratio(
            power_dict[location][BAND_POWERS.Beta.name], 
            power_dict[location][BAND_POWERS.Theta.name])
        get_relax = lambda location: self.calculate_ratio(
            power_dict[location][BAND_POWERS.Alpha.name], 
            power_dict[location][BAND_POWERS.Theta.name])
        
        # create dictionary to return
        ret_dict = {
            NeuroFeedback.FOCUS: get_focus,
            NeuroFeedback.RELAX: get_relax
        }
        
        # apply score calculations per location
        for nfb_name, nfb_func in ret_dict.items():
            signed_dict = {location : nfb_func(location) for location in power_dict}
            unsigned_dict = {k : (v+1)/2 for k, v in signed_dict.items()}
            ret_dict[nfb_name] = {
                NeuroFeedback.SIGNED    : signed_dict,
                NeuroFeedback.UNSIGNED  : unsigned_dict
            }
        
        return ret_dict
    
    def calculate_ratio(self, numerator, denominator):
        return tanh_normalize(numerator / denominator, self.normalize_scale, -1)