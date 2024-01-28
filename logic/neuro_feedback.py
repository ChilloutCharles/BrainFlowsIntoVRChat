from logic.power_bands import PowerBands
from constants import BAND_POWERS
from utils import tanh_normalize, map2dto1d

class NeuroFeedback(PowerBands):
    def __init__(self, board, window_seconds=2, normalize_scale=1.1, ema_decay=0.025):
        super().__init__(board, window_seconds=window_seconds, ema_decay=ema_decay)
        self.bands = [BAND_POWERS.Alpha, BAND_POWERS.Beta, BAND_POWERS.Theta]
        self.band_names = list(map(lambda b: b.name.lower(), self.bands))
        self.locations = ["left", "right", "avg"]
        self.normalize_scale = normalize_scale

    def get_data_dict(self):
        power_dict = super().get_data_dict()
        ret_dict = {}

        for location in power_dict:
            loc_dict = power_dict[location]

            alpha_power = loc_dict[BAND_POWERS.Alpha.name.lower()]
            beta_power = loc_dict[BAND_POWERS.Beta.name.lower()]
            theta_power = loc_dict[BAND_POWERS.Theta.name.lower()]
            
            focus = tanh_normalize(beta_power / theta_power, self.normalize_scale, -1)
            relax = tanh_normalize(alpha_power / theta_power, self.normalize_scale, -1)

            score_dict = {
                'focus' : focus,
                'relax' : relax
            }
            
            ret_dict[location] = score_dict

        return ret_dict