Here are the commands to build the workspace and run each of the 5 algorithms to collect the data and generate the plots:

### 1. Build the Workspace
To build and compile the code:
```bash
colcon build --symlink-install
```
Make sure to source the setup file in any new terminal:
```bash
source install/setup.bash
```

---

### 2. Run the Algorithms to Collect Logs
Run each command below. Since we set the play speed to 5x, each run will complete in under a minute. You can close/terminate (`Ctrl+C`) the launch terminal once the bag has finished playing to flush and write the logs to disk.

1. **5-point Algorithm (with RANSAC)**:
   ```bash
   ros2 launch lab_6 video_tracking.launch.yaml pose_estimator:=0 use_ransac:=true
   ```
2. **8-point Algorithm (with RANSAC)**:
   ```bash
   ros2 launch lab_6 video_tracking.launch.yaml pose_estimator:=1 use_ransac:=true
   ```
3. **2-point Algorithm (with RANSAC)**:
   ```bash
   ros2 launch lab_6 video_tracking.launch.yaml pose_estimator:=2 use_ransac:=true
   ```
4. **Arun 3D-3D Algorithm (with RANSAC)**:
   ```bash
   ros2 launch lab_6 video_tracking.launch.yaml pose_estimator:=3 use_ransac:=true
   ```
5. **5-point Algorithm (without RANSAC)**:
   ```bash
   ros2 launch lab_6 video_tracking.launch.yaml pose_estimator:=0 use_ransac:=false
   ```

---

### 3. Generate Error Plots
Once you have run all 5 configurations, run this Python script to generate the comparison plots:
```bash
python3 src/lab6/plot_errors.py
```
This will output `rotation_errors.png` and `translation_errors.png` in `src/lab6/`.