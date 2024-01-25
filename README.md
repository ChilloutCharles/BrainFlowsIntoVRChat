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

**OSC Avatar Parameters**

Avatar parameters being sent are floats that range from -1.0 to 1.0. Negative and Positive values correspond to low and high focus/relaxation. Update your avatar paramaters as needed to influnece animations and the like. Parameters are also seperated by left and right sides of the brain. Have fun!

- `osc_focus_left`
- `osc_relax_left`
- `osc_focus_right`
- `osc_relax_right`
- `osc_focus_avg`
- `osc_relax_avg`

For easier startup, I've added a hue shift parameter based on the focus and relax values. This parameter will range from 0.0 to 1.0.

- `HueShift`

Added are optional paramaters that appear based on whether or not your headband supports it.
- `osc_battery_lvl` (int [0-100])
- `osc_heart_bpm` (int[0-255])
- `osc_oxygen_percent` (float[0.0-1.0])
- `osc_respiration_bpm` (int[0-255])

I've also added the alpha, beta, theta, delta, and gamma band power ratios between left, right sides of the head as well as avg. You can access them via this path:
- `osc_band_power_(head_location)_(brainwave name)` (float [0-1]) (without the parenthesis)

For debug, a boolean paramater is sent to monitor the connection status of the headband as well as a time difference parameter to show the difference between last sample time and current time in seconds.

- `osc_is_connected` (boolean)
- `osc_time_diff` (float)

## Thanks
Thanks to [@Mitzi_DelverVRC](https://twitter.com/Mitzi_DelverVRC) and [AartHark](https://github.com/AartHauk) for help with PPG signal work

Thanks to [@wordweaver1001](https://twitter.com/wordweaver1001) for intial user testing



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

## License
[MIT](http://opensource.org/licenses/MIT).
