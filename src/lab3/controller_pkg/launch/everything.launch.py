import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource, PythonLaunchDescriptionSource

def generate_launch_description():
    # 1. Include the tesse_ros_bridge launch file
    bridge_launch_file = os.path.join(
        get_package_share_directory('tesse_ros_bridge'),
        'launch',
        'tesse_quadrotor_bridge.launch.yaml'
    )
    bridge_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(bridge_launch_file)
    )

    # 2. Include the controller_pkg launch file
    controller_launch_file = os.path.join(
        get_package_share_directory('controller_pkg'),
        'launch',
        'traj_tracking.launch.py'
    )
    controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(controller_launch_file)
    )

    return LaunchDescription([
        bridge_launch,
        controller_launch
    ])
