# BFiVRC : Dearpygui Desktop Viewer

This desktop appliation captures the values of OSC messages posted on 127.0.0.1:9000 and displays it in a graph.
This might be helpful in data exploration.

<image src="media/demobfivcviewer.PNG"></image>

## Usage:
The application will open a Websocket server to collect the OSC Messages as long as the application is running.
### Usage without using VRC at the same time
```
1.
 python main.py --board_id <board_id>
2.
 python dearpyviewer/dearpyviewer.py 
```
### Usage USING VRC at the same time
Using VRC at the same time, you need to specify the ports, and forward for VRC to 9000 e.g:
```
1.
 python main.py --board_id <board_id> --osc-port 9010
2.
 python dearpyviewer/dearpyviewer.py --ip-port_listen 9010 -- ip-port_forward 9000 
```

## Dearpygui
Dearpygui is a python native Desktop UI which allows realtime drawing cross-plattform and is highly performant.
For more information visit [DearPyGui](https://github.com/hoffstadt/DearPyGui)

## Testing
To test the viewer without BFI device, you can run osc_device_simulation.py


## Further Work
- The color of the message toggles' text in the viewer should match the graph color 
- Implement a pipeline to extract the osc_messages data, such as paths and limits from the main project
- Observe performance and test correctness of displayed data


