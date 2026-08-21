source install/setup.bash
ros2 launch lab_5 two_frames_tracking.launch.yaml descriptor:=SIFT save_images:=true show_images:=false

ros2 launch lab_5 video_tracking.launch.yaml path_to_dataset:=/home/bach/projects/VNAV-labs/lab5/bags/30fps_424x240_2018-10-01-18-35-06-20260526T032439Z-3-001/30fps_424x240_2018-10-01-18-35-06/30fps_424x240_2018-10-01-18-35-06.db3 # Alternatively, you can set the absolute path to the bag with path_to_dataset:=<path>

ros2 bag play /home/bach/projects/VNAV-labs/lab5/bags/30fps_424x240_2018-10-01-18-35-06-20260526T032439Z-3-001/30fps_424x240_2018-10-01-18-35-06/30fps_424x240_2018-10-01-18-35-06.db3

ros2 launch lab_5 two_frames_tracking.launch.yaml descriptor:=SIFT