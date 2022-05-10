import launch
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='final_project',
            namespace='',
            executable='simulation.py',
            name='simulation',
            output="screen",
            on_exit=launch.actions.Shutdown(),
            parameters=[
                {"visualize":True},
                {"save":False},
                {"sensors":True},
                {"save_name":"path_slam"},
                {"initial_location":[-120,0,1.6]},
                {"initial_orientation":0.0},
                {"object_location_file":"/home/yisen/final/data/object_locations.csv"},
                {"random_object_count": 0},
                {"publish_state_directly":True},
                {"max_duration": 30.0},
                {"target_location":[120.0,0.0,0.0]}
            ],
        ),
        Node(
            package='final_project',
            namespace='',
            executable='slam_node_stereo',
            name='slam_node',
            #output="screen",
            on_exit=launch.actions.Shutdown(),
            parameters=[
            {"vocabulary": "/home/me468/vslam/ORB_SLAM3/Vocabulary/ORBvoc.txt"},
            {"settings":"/home/yisen/final/data/stereo.yaml"}
            ],
        ),

    ])
