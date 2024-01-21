import enum

OSC_BASE_PATH = '/avatar/parameters/'

class OSC_Path:
    RelaxAvg = OSC_BASE_PATH + 'osc_relax_avg'
    RelaxLeft = OSC_BASE_PATH + 'osc_relax_left'
    RelaxRight = OSC_BASE_PATH + 'osc_relax_right'
    FocusAvg = OSC_BASE_PATH + 'osc_focus_avg'
    FocusLeft = OSC_BASE_PATH + 'osc_focus_left'
    FocusRight = OSC_BASE_PATH + 'osc_focus_right'
    Battery = OSC_BASE_PATH + 'osc_battery_lvl'
    ConnectionStatus = OSC_BASE_PATH + 'osc_is_connected'
    HeartBps = OSC_BASE_PATH + 'osc_heart_bps'
    HeartBpm = OSC_BASE_PATH + 'osc_heart_bpm'
    OxygenPercent = OSC_BASE_PATH + 'osc_oxygen_percent'
    HueShift = OSC_BASE_PATH + 'HueShift'

class BAND_POWERS(enum.IntEnum):
    Gamma = 4
    Beta = 3
    Alpha = 2
    Theta = 1
    Delta = 0