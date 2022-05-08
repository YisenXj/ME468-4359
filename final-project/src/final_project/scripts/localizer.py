#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile
from custom_msgs.msg import VehicleState
from sensor_msgs.msg import NavSatFix, MagneticField

from ament_index_python.packages import get_package_share_directory

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import pychrono.sensor as sens
import pychrono as chrono

class LocalizerNode(Node):
    def __init__(self):
        super().__init__('localizer_node')

        # update frequency of this node
        self.freq = 10.0

        #tracked variables
        self.position = np.array([0.0,0.0,0.0])
        self.velocity =np.array([0.0,0.0,0.0])
        self.heading = np.array([1.0,0.0,0.0])

        self.gps_reference = chrono.ChVectorD(-89.400, 43.070, 260.0)
        self.last_pos = np.array([0.0,0.0,0.0])
        self.last_gps = -1

        # Read in share directory location
        package_share_directory = get_package_share_directory('control_stack')

        self.declare_parameter('visualize', False)
        self.visualize = self.get_parameter(
            'visualize').get_parameter_value().bool_value

        self.declare_parameter('save', False)
        self.save = self.get_parameter(
            'save').get_parameter_value().bool_value

        self.declare_parameter('save_name', "localizer_output")
        self.save_name = self.get_parameter(
            'save_name').get_parameter_value().string_value

        home_dir = os.getenv('HOME')
        save_dir = "me468_output"
        if(not os.path.exists(os.path.join(home_dir,save_dir))):
            os.mkdir(os.path.join(home_dir,save_dir))
            
        self.output_dir = os.path.join(home_dir,save_dir,self.save_name)
        if(not os.path.exists(self.output_dir)):
            os.mkdir(self.output_dir)
        self.frame_number = 0

        # DDS QOS Setup - important for detemining lag and packet drop behavior
        qos_profile = QoSProfile(depth=1)
        qos_profile.history = QoSHistoryPolicy.KEEP_LAST

        # subscribers
        self.sub_gps = self.create_subscription(
            NavSatFix, 'gps', self.gps_callback, qos_profile)
        self.sub_imu = self.create_subscription(
            MagneticField, 'magnetic', self.mag_callback, qos_profile)

        # publishers
        self.pub_state = self.create_publisher(VehicleState, 'state', qos_profile)
        self.timer = self.create_timer(1/self.freq, self.pub_callback)

        #visualization setup
        if(self.visualize):
            matplotlib.use("TKAgg")
        else:
            matplotlib.use("Agg")
            self.fig, self.ax = plt.subplots()
            plt.title("Localization")
            self.patches = []
            self.ax.set_xlabel("X [m] (Global Frame)")
            self.ax.set_ylabel("Y [m] (Global Frame)")

    # function to process data this class subscribes to
    def gps_callback(self, msg):
        pos = chrono.ChVectorD(msg.longitude,msg.latitude,msg.altitude)
        sens.GPS2Cartesian(pos,self.gps_reference)

        if(self.last_gps > 0):
            dt = self.get_clock().now().nanoseconds / 1e9 - self.last_gps
            self.velocity = (self.position - np.array([pos.x,pos.y,pos.z])) / dt

        self.position =  np.array([pos.x,pos.y,pos.z])
        self.last_gps = self.get_clock().now().nanoseconds / 1e9


    def mag_callback(self, msg):
        self.heading =  np.array([msg.magnetic_field.x,msg.magnetic_field.y,0.0])

        if(np.linalg.norm(self.heading) > 1e-3):
            self.heading = self.heading / np.linalg.norm(self.heading)

    # callback to run a loop and publish data this class generates
    def pub_callback(self):

        msg = VehicleState()
        msg.position = self.position
        msg.velocity = self.velocity
        msg.heading = self.heading
        self.pub_state.publish(msg)

        if(self.visualize or self.save):
            self.ax.set_xlim((self.position[0]-10,self.position[0]+10))
            self.ax.set_ylim((self.position[1]-10,self.position[1]+10))
            [p.remove() for p in self.patches]
            self.patches.clear()

            scale = .5*np.linalg.norm(self.velocity) + 1
            arr = patches.Arrow(self.position[0],self.position[1],scale*self.heading[0],scale*self.heading[1],width=2,color='b')
            self.ax.add_patch(arr)
            self.patches.append(arr)

            if(self.visualize):
                plt.draw()
                plt.pause(0.0001)
            elif(self.save):
                plt.savefig(os.path.join(self.output_dir,"frame_{}.png".format(self.frame_number)))
                self.frame_number += 1


def main(args=None):
    rclpy.init(args=args)
    localizer = LocalizerNode()
    rclpy.spin(localizer)
    localizer.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
