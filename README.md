# BrainFlowsIntoVRChat

This is a BrainFlow implementation of my [bci-workshop fork](https://github.com/ChilloutCharles/bci-workshop) that sends your brain's relaxation and focus metrics to vrchat avatar paramaters via OSC.

**Why BrainFlow?**

The [BrainFlow](https://BrainFlow.org) library provides a uniform API that is device agnostic, allowing this implementation of my workshop fork to work for all [supported biosensors](https://BrainFlow.readthedocs.io/en/stable/SupportedBoards.html) (including the muse headbands). No extra software needed!

**Demos** 
- [Rantis's Brain Controlled Ears](https://twitter.com/RantiMess/status/1746704510972580061)
- [Brain Controlled Ears: Five Months Later [VRChat]](https://www.youtube.com/watch?v=kPPTT3ogEgg)
- [VRCHAT OSC MAGIC! (Last 30 seconds)](https://twitter.com/kentrl_z/status/1497020472046800897)
- [Old version of Brain Controlled Ears](https://www.youtube.com/watch?v=WjWc51xNgKg)

**Instructions**

1. Download this project to a folder and remember its folder path
2. Install [Python 3.11.5](https://www.python.org/downloads/release/python-3115/)
3. Open the command prompt by typing searching cmd at the start menu
4. Navigate to the project's path within the command prompt. 
   - example: `cd "C:\Users\<YOUR USERNAME HERE>\Documents\GitHub\BrainFlowsIntoVRChat"` 
5. Execute this command to install needed depedencies: `pip install -r requirements.txt`
6. Look up your device's id: [Board IDs Page](https://brainflow.readthedocs.io/en/stable/UserAPI.html?highlight=MUSE_2016_BOARD#brainflow-board-shim)
7. Turn on your headband
8. Run the script `main.py` with your device id. The command for running with a [Muse 2 headband](https://choosemuse.com/muse-2/) would be: `python .\main.py --board-id 38`

**OSC Avatar Parameter Schema**

Here are the various avatar parameters sent to VRChat. Neurofeedback scores range from -1 to 1 for signed floats, 0 to 1 for unsigned, with higher and lower values corresponding to higher and lower relax/focus scores. Depending on the board you're using, respiration data and battery level might be available. Power Band numbers are also sent per location as well, ranging from 0 to 1 averaging at 0.2.

```yaml
Brainflow:
  Meta:
    - VersionMajor [int] -- Determines breaking changes in the schema representation
    - VersionMinor [int] -- Any update to the schema which remains compatible with existing prefabs.
  NeuroFeedback:
    Focus:
      Signed:
        - Left [float]
        - Right [float]
        - Average [float]
      Unsigned:
        - Left [float]
        - Right [float]
        - Average [float]
    Relax:
      Signed:
        - Left [float]
        - Right [float]
        - Average [float]
      Unsigned:
        - Left [float]
        - Right [float]
        - Average [float]
  PowerBands:
    Left:
      - Alpha [float]
      - Beta [float]
      - Theta [float]
      - Delta [float]
      - Gamma [float]
    Right:
      - Alpha [float]
      - Beta [float]
      - Theta [float]
      - Delta [float]
      - Gamma [float]
    Average:
      - Alpha [float]
      - Beta [float]
      - Theta [float]
      - Delta [float]
      - Gamma [float]
  Addons:
    - Hueshift [float 0-1]
  HeartRate: # board dependent
    - Supported [bool]
    - HeartBeatsPerSecond [float]
    - HeartBeatsPerMinute [int]
  Respiration: # board dependent
    - Supported [bool]
    - OxygenPercent [float]
    - BreathsPerSecond [float]
    - BreathsPerMinute [int]
  Device:
    - SecondsSinceLastUpdate [float]
    - Connected [bool]
    Battery: # board dependent
      - Supported [bool]
      - Level [float]
```

To use parameters in within VRChat, write the parameter name as a path. For example, to get the left side alpha value, the parameter name would be:
- `Brainflow/PowerBands/Left/Alpha`

## Deprecation

In order to use the old parameter names as documented in previous versions, add the argument `--use-old-reporter`. An announcement will be made soon to sunset the old parameter names.

### Old Parameter Mappings

Need to migrate your existing prefabs? You convert your existing parameters to the new standard using this chart!

| Old Parameter | New Parameter |
| ------------- | ----------------- |
| osc_focus_left | Brainflow/NeuroFeedback/Focus/Signed/Left |
| osc_focus_right | Brainflow/NeuroFeedback/Focus/Signed/Right |
| osc_focus_avg | Brainflow/NeuroFeedback/Focus/Signed/Average |
| osc_relax_left | Brainflow/NeuroFeedback/Relax/Signed/Left |
| osc_relax_right | Brainflow/NeuroFeedback/Relax/Signed/Right |
| osc_relax_avg | Brainflow/NeuroFeedback/Relax/Signed/Average |
| osc_heart_bpm | Brainflow/HeartRate/HeartBeatsPerMinute |
| osc_heart_bps | Brainflow/HeartRate/HeartBeatsPerSecond |
| osc_oxygen_percent | Brainflow/Respiration/OxygenPercent |
| osc_respiration_bpm | Brainflow/Respiration/BreathsPerMinute |
| osc_band_power_left_alpha | Brainflow/PowerBands/Left/Alpha |
| osc_band_power_left_beta | Brainflow/PowerBands/Left/Beta |
| osc_band_power_left_theta | Brainflow/PowerBands/Left/Theta |
| osc_band_power_left_delta | Brainflow/PowerBands/Left/Delta |
| osc_band_power_left_gamma | Brainflow/PowerBands/Left/Gamma |
| osc_band_power_right_alpha | Brainflow/PowerBands/Right/Alpha |
| osc_band_power_right_beta | Brainflow/PowerBands/Right/Beta |
| osc_band_power_right_theta | Brainflow/PowerBands/Right/Theta |
| osc_band_power_right_delta | Brainflow/PowerBands/Right/Delta |
| osc_band_power_right_gamma | Brainflow/PowerBands/Right/Gamma |
| osc_band_power_overall_alpha | Brainflow/PowerBands/Average/Alpha |
| osc_band_power_overall_beta | Brainflow/PowerBands/Average/Beta |
| osc_band_power_overall_theta | Brainflow/PowerBands/Average/Theta |
| osc_band_power_overall_delta | Brainflow/PowerBands/Average/Delta |
| osc_band_power_overall_gamma | Brainflow/PowerBands/Average/Gamma |
| osc_battery_lvl | Brainflow/Device/Battery/Level |
| osc_is_connected | Brainflow/Device/Connected |
| osc_time_diff | Brainflow/Device/SecondsSinceLastUpdate |
| HueShift | Brainflow/Addon/HueShift |

## Thanks

Thanks to 
- [@Mitzi_DelverVRC](https://twitter.com/Mitzi_DelverVRC) and [AartHark](https://github.com/AartHauk) for help with PPG signal work.
- [@wordweaver1001](https://twitter.com/wordweaver1001) for intial user testing.
- [AtriusX](https://github.com/AtriusX) for helping create a parameter schema.

## Troubleshooting
- I have broken bluetooth adapter built into my pc and I would like to use a dongle instead. How can I connect my headband to that dongle?
  1. Disconnect the bluetooth dongle you want to use
  2. Search up 'device manager' on the start menu
  3. Find an entry for a bluetooth radio, right click on it and disable it
  4. Plug the new bluetooth dongle back in

- Muse Headband connects just fine but times out after a few seconds. Solution: Reset the headband
  1. Turn off the headband
  2. Press and hold the power button until it turns on. Keep pressing until the light changes.
  3. Reconnect.

- I've set up everything and made a new avatar, but its still not reacting
  - Reason: VRChat stores cached OSC parameters for your avatar that aren't updated when the avatar is updated with new parameters
  - Solution: Go to `C:\Users\<YOUR USERNAME HERE>\AppData\LocalLow\VRChat\VRChat` and delete all folders under it, then reload avatar

## License
[MIT](http://opensource.org/licenses/MIT).
