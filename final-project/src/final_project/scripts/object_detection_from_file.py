#!/usr/bin/env python3 
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from custom_msgs.msg import VehicleState, ObjectList, Object

from ament_index_python.packages import get_package_share_directory

import numpy as np
import os
import matplotlib 
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class OjbectDetectionFromFileNode(Node):
    def __init__(self):
        super().__init__('object_detection_from_file_node')

        # update frequency of this node
        self.freq = 10.0

        # Read in share directory location
        package_share_directory = get_package_share_directory('control_stack')

        # ROS parameters passed from launch file
        self.declare_parameter('object_file', "")
        self.object_file = self.get_parameter(
            'object_file').get_parameter_value().string_value
        file_path = os.path.join(package_share_directory, self.object_file)
        self.object_locations = np.loadtxt(file_path, delimiter=',')

        self.declare_parameter('goal_location', [0,400])
        self.goal_location = self.get_parameter(
            'goal_location').get_parameter_value().double_array_value

        self.declare_parameter('field_of_view', 1.0)
        self.field_of_view = self.get_parameter(
            'field_of_view').get_parameter_value().double_value

        self.declare_parameter('view_distance', 100.0)
        self.view_distance = self.get_parameter(
            'view_distance').get_parameter_value().double_value

        self.declare_parameter('visualize', False)
        self.visualize = self.get_parameter(
            'visualize').get_parameter_value().bool_value

        self.declare_parameter('save', False)
        self.save = self.get_parameter(
            'save').get_parameter_value().bool_value

        # tracked variables
        self.vehicle_state = VehicleState()

        # DDS QOS Setup - important for detemining lag and packet drop behavior
        qos_profile = QoSProfile(depth=1)
        qos_profile.history = QoSHistoryPolicy.KEEP_LAST

        #subscribers
        self.sub_state = self.create_subscription(
            VehicleState, 'state', self.state_callback, qos_profile)

        # publishers
        self.pub_objects = self.create_publisher(
            ObjectList, 'objects', qos_profile)
        self.timer = self.create_timer(
            1/self.freq, self.pub_callback)


        #setup visualization
        if(self.visualize):
            matplotlib.use("TKAgg")
            self.fig, self.ax = plt.subplots()
            plt.title("Known Objects From File")
            self.patches = []
            self.ax.set_xlabel("X [m] (Global Frame)")
            self.ax.set_ylabel("Y [m] (Global Frame)")
            self.ax.set_xlim([-500,500])
            self.ax.set_ylim([-500,500])

    def state_callback(self, msg):
        self.vehicle_state = msg

    # callback to run a loop and publish data this class generates
    def pub_callback(self):
        
        pos = [self.vehicle_state.position[0],self.vehicle_state.position[1]]
        heading = [self.vehicle_state.heading[0],self.vehicle_state.heading[1]]
        if(np.linalg.norm(heading) > 0):
            heading = heading/np.linalg.norm(heading)

        #calculate which objects are in view
        viewable_objects = []
        for l in self.object_locations:
            #needs to meet the distance requirement
            if(np.linalg.norm(l - pos) < self.view_distance):
                #needs to be within the field of view
                l_norm = (l - pos)
                if(np.linalg.norm(l_norm) > 0):
                    l_norm = l_norm/np.linalg.norm(l_norm)
                ang = np.arccos(np.dot(l_norm,heading))
                # self.get_logger().info("Object pos='%s', vehicle pos='%s', angle='%s'" % (str(l),str(pos),str(ang)))
                if(ang < self.field_of_view / 2):
                    viewable_objects.append(l)

        viewable_objects = np.asarray(viewable_objects)

        # self.get_logger().info("Viewable objects: '%s'" % (str(viewable_objects.shape)))


        if(self.visualize):
            [p.remove() for p in self.patches]
            self.patches.clear()

            #plot heading from velocity with blue arrow at vehicle position
            scale = 50
            arr = patches.Arrow(pos[0],pos[1],scale*heading[0],scale*heading[1],width=2,color='b')
            self.ax.add_patch(arr)
            self.patches.append(arr)

            #plot vehicle point in blue
            radius = 10
            circ = patches.Circle(pos[0:2],radius=radius,color='b')
            self.ax.add_patch(circ)
            self.patches.append(circ)

            #plot all object locations
            for l in self.object_locations:
                circ = patches.Circle(l,radius=radius,color='b',fill=False)
                self.ax.add_patch(circ)
                self.patches.append(circ)

            #plot the viewable object locations
            for l in viewable_objects:
                circ = patches.Circle(l,radius=radius,color='r',fill=False)
                self.ax.add_patch(circ)
                self.patches.append(circ)

            plt.draw()
            plt.pause(0.0001)

        # pack and publish vehicle control message
        msg = ObjectList()

        #push all the viewable objects into the msg
        for p in viewable_objects:
            obj = Object()
            obj.position = [p[0],p[1],0.0]
            msg.objects.append(obj)

        #publish objects msg
        self.pub_objects.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    object_detector = OjbectDetectionFromFileNode()
    rclpy.spin(object_detector)
    object_detector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
