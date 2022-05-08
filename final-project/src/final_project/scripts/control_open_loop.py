#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile
from custom_msgs.msg import VehicleInput

from ament_index_python.packages import get_package_share_directory

import numpy as np
import os


class ControlOpenLoopNode(Node):
    def __init__(self):
        super().__init__('control_node')

        # update frequency of this node
        self.freq = 10.0

        # timing for finding open loop control values
        self.t_start = self.get_clock().now().nanoseconds / 1e9

        # Read in share directory location
        package_share_directory = get_package_share_directory('control_stack')

        # ROS parameters passed from launch file
        self.declare_parameter('control_file', "")
        self.file = self.get_parameter(
            'control_file').get_parameter_value().string_value
        file_path = os.path.join(package_share_directory, self.file)
        self.recorded_inputs = np.loadtxt(file_path, delimiter=',')

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
        self.pub_vehicle_cmd = self.create_publisher(
            VehicleInput, 'vehicle_cmd', qos_profile)
        self.timer = self.create_timer(
            1/self.freq, self.pub_callback)

    # callback to run a loop and publish data this class generates
    def pub_callback(self):
        # find the current ROS time
        t = self.get_clock().now().nanoseconds / 1e9 - self.t_start

        # interpolate inputs from file
        throttle = np.interp(
            t, self.recorded_inputs[:, 0], self.recorded_inputs[:, 1])
        braking = np.interp(
            t, self.recorded_inputs[:, 0], self.recorded_inputs[:, 2])
        steering = np.interp(
            t, self.recorded_inputs[:, 0], self.recorded_inputs[:, 3])

        # pack and publish vehicle control message
        msg = VehicleInput()
        msg.steering = steering
        msg.throttle = throttle
        msg.braking = braking
        self.pub_vehicle_cmd.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    control = ControlOpenLoopNode()
    rclpy.spin(control)
    control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
