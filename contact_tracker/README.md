# contact_tracker

##### Track contacts by applying Kalman filters to incoming detect messages

### Dependencies
python2\
ROS melodic\
filterpy\
matplotlib\
numpy

### Installation
1. Install filterpy: `pip install filterpy`\
&nbsp;&nbsp; Note: if you do not intend to run the non-production node, skip the rest of these
installation instructions.
2. Install matplotlib: `sudo apt-get install python-matplotlib`
3. Install numpy: `pip install numpy`

### Options
#### tracker_debug.py

Run the non-production tracker node, and optionally produce plots.

usage: tracker_debug.py [-h] [-plot_type {xs_ys, xs_times, ellipses}] [-o O]

optional arguments:\
&nbsp;&nbsp;&nbsp;&nbsp;-h, --help &nbsp;&nbsp; show this help message and exit\
&nbsp;&nbsp;&nbsp;&nbsp;-plot_type {xs_ys, xs_times, ellipses} &nbsp;&nbsp; specify the type of plot to produce, if you want one\
&nbsp;&nbsp;&nbsp;&nbsp;-o O &nbsp;&nbsp; path to save the plot produced, default: tracker_plot, current working directory


Example run:  
`$ rosrun contact_tracker tracker_debug.py -plot_type ellipses -o ~/ellipse_plot`  


#### tracker.py

Run the production tracker node

usage: tracker.py [-h]

optional arguments:\
&nbsp;&nbsp;&nbsp;&nbsp;-h, --help &nbsp;&nbsp; show this help message and exit\

Example run:  
`$ rosrun contact_tracker tracker.py`
