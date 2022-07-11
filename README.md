# BrainFlowsIntoVRChat

This is a BrainFlow implementation of my [bci-workshop fork](https://github.com/ChilloutCharles/bci-workshop) that sends your brain's relaxation and focus metrics to vrchat avatar paramaters via OSC.

**Why BrainFlow?**

The [BrainFlow](https://BrainFlow.org) library provides a uniform API that is device agnostic, allowing this implementation of my workshop fork to work for all [supported biosensors](https://BrainFlow.readthedocs.io/en/stable/SupportedBoards.html) (including the muse headbands). No extra software needed!

**Demos** 
- [VRCHAT OSC MAGIC! (Last 30 seconds)](https://twitter.com/kentrl_z/status/1497020472046800897)
- [Old version of Brain Controlled Ears](https://www.youtube.com/watch?v=WjWc51xNgKg)

**Instructions**

1. Follow the [BrainFlow Installation Instructions for Python](https://BrainFlow.readthedocs.io/en/stable/BuildBrainFlow.html#python)
2. Install [python-osc](https://pypi.org/project/python-osc/)
3. Look up your device's id: [supported biosensors](https://BrainFlow.readthedocs.io/en/stable/SupportedBoards.html)
4. Run the script `main.py` with your device id. The command for running with a [Muse 2 headband](https://choosemuse.com/muse-2/) would be: `python .\main.py --board-id 38`

**OSC Avatar Parameters**

Avatar parameters being sent are floats that range from -1.0 to 1.0. Negative and Positive values correspond to low and high focus/relaxation. Update your avatar paramaters as needed to influnece animations and the like. Have fun!

- `/avatar/parameters/osc_relax_avg`
- `/avatar/parameters/osc_focus_avg`

A boolean paramater is sent to monitor the connection status of the headband

- `/avatar/parameters/osc_is_connected`

Also added are optional paramaters that appear based on whether or not your headband supports it.
- `/avatar/parameters/osc_battery_lvl` (int [0-100])
- `/avatar/parameters/osc_heart_bps` (float)
- `/avatar/parameters/osc_heart_bpm` (int)

## Thanks
Thanks to [@Mitzi_DelverVRC](https://twitter.com/Mitzi_DelverVRC) for help with PPG work

## License
[MIT](http://opensource.org/licenses/MIT).