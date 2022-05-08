#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

from ament_index_python.packages import get_package_share_directory

import numpy as np
import os


class PathPlannerFromFileNode(Node):
    def __init__(self):
        super().__init__('path_from_file_node')

        # update frequency of this node
        self.freq = 10.0

        # Read in share directory location
        package_share_directory = get_package_share_directory('control_stack')

        # ROS parameters passed from launch file
        self.declare_parameter('path_file', "")
        self.file = self.get_parameter(
            'path_file').get_parameter_value().string_value
        file_path = os.path.join(package_share_directory, self.file)
        self.preset_path = np.loadtxt(file_path, delimiter=',')

        self.declare_parameter('visualize', False)
        self.visualize = self.get_parameter(
            'visualize').get_parameter_value().bool_value

        self.declare_parameter('save', False)
        self.save = self.get_parameter(
            'save').get_parameter_value().bool_value

        # DDS QOS Setup - important for detemining lag and packet drop behavior
        qos_profile = QoSProfile(depth=1)
        qos_profile.history = QoSHistoryPolicy.KEEP_LAST

        # publishers
        self.pub_path = self.create_publisher(
            Path, 'path', qos_profile)
        self.timer = self.create_timer(
            1/self.freq, self.pub_callback)

        self.count = 0

    # callback to run a loop and publish data this class generates
    def pub_callback(self):

        #only send the path a couple times since it will never change
        if(self.count > 50):
            return

        self.count += 1

        # pack and publish vehicle control message
        msg = Path()

        #push all point from file into msg
        for p in self.preset_path:
            pt = PoseStamped()
            pt.pose.position.x = p[0]
            pt.pose.position.y = p[1]
            msg.poses.append(pt)

        #publish path msg
        self.pub_path.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    path_planner = PathPlannerFromFileNode()
    rclpy.spin(path_planner)
    path_planner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
