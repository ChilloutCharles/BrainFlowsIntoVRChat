# BrainFlowsIntoVRChat

This is a BrainFlow implementation of my [bci-workshop fork](https://github.com/ChilloutCharles/bci-workshop) that sends your brain's relaxation and focus metrics to vrchat avatar paramaters via OSC.

**Why BrainFlow?**

The [BrainFlow](https://BrainFlow.org) library provides a uniform API that is device agnostic, allowing this implementation of my workshop fork to work for all [supported biosensors](https://BrainFlow.readthedocs.io/en/stable/SupportedBoards.html) (including the muse headbands). No extra software needed!

**Demos** 
- [Brain Controlled Ears: Five Months Later [VRChat]](https://www.youtube.com/watch?v=kPPTT3ogEgg)
- [VRCHAT OSC MAGIC! (Last 30 seconds)](https://twitter.com/kentrl_z/status/1497020472046800897)
- [Old version of Brain Controlled Ears](https://www.youtube.com/watch?v=WjWc51xNgKg)

**Instructions**

1. Install [Python](https://www.python.org)
2. Install required libraries with this command: `pip install -r requirements.txt`
3. Look up your device's id: [supported biosensors](https://BrainFlow.readthedocs.io/en/stable/SupportedBoards.html)
4. Run the script `main.py` with your device id. The command for running with a [Muse 2 headband](https://choosemuse.com/muse-2/) would be: `python .\main.py --board-id 38`

**OSC Avatar Parameters**

Avatar parameters being sent are floats that range from -1.0 to 1.0. Negative and Positive values correspond to low and high focus/relaxation. Update your avatar paramaters as needed to influnece animations and the like. Have fun!

- `/avatar/parameters/osc_relax_avg`
- `/avatar/parameters/osc_focus_avg`

A boolean paramater is sent to monitor the connection status of the headband

- `/avatar/parameters/osc_is_connected`

Added are optional paramaters that appear based on whether or not your headband supports it.
- `/avatar/parameters/osc_battery_lvl` (int [0-100])

I've also added the alpha, beta, theta, delta, and gamma band power numbers. You can access them via this path:
- `/avatar/paramaters/osc_band_power_(brainwave name)` (float [0-1]) (without the parenthesis)

## Thanks
Thanks to [@Mitzi_DelverVRC](https://twitter.com/Mitzi_DelverVRC) for help with PPG work

## License
[MIT](http://opensource.org/licenses/MIT).