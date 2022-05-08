#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile
from custom_msgs.msg import VehicleState, VehicleInput
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

from ament_index_python.packages import get_package_share_directory

import pychrono as chrono

import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class ControlPathFollowerNode(Node):
    def __init__(self):
        super().__init__('control_node')

        # update frequency of this node
        self.freq = 10.0

        # Read in share directory location
        package_share_directory = get_package_share_directory('control_stack')

        # ROS parameters passed from launch file
        self.declare_parameter('steering_kp', 1.0)
        self.steering_kp = self.get_parameter(
            'steering_kp').get_parameter_value().double_value

        self.declare_parameter('steering_kd', 1.0)
        self.steering_kd = self.get_parameter(
            'steering_kd').get_parameter_value().double_value

        self.declare_parameter('steering_ki', 1.0)
        self.steering_ki = self.get_parameter(
            'steering_ki').get_parameter_value().double_value

        self.declare_parameter('lookahead', 1.0)
        self.lookahead = self.get_parameter(
            'lookahead').get_parameter_value().double_value

        self.declare_parameter('speed_kp', 1.0)
        self.speed_kp = self.get_parameter(
            'speed_kp').get_parameter_value().double_value

        self.declare_parameter('speed_kd', 1.0)
        self.speed_kd = self.get_parameter(
            'speed_kd').get_parameter_value().double_value

        self.declare_parameter('speed_ki', 1.0)
        self.speed_ki = self.get_parameter(
            'speed_ki').get_parameter_value().double_value

        self.declare_parameter('target_speed', 10.0)
        self.target_speed = self.get_parameter(
            'target_speed').get_parameter_value().double_value

        self.declare_parameter('visualize', False)
        self.visualize = self.get_parameter(
            'visualize').get_parameter_value().bool_value

        self.declare_parameter('vis_with_buffer', 20.0)
        self.vis_with_buffer = self.get_parameter(
            'vis_with_buffer').get_parameter_value().double_value

        self.declare_parameter('save', False)
        self.save = self.get_parameter(
            'save').get_parameter_value().bool_value

        self.declare_parameter('save_name', "path_follower_output")
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

        # tracked class variables
        self.path = Path()
        self.vehicle_state = VehicleState()
        self.total_steering_error = 0.0
        self.previous_steering_error = 0.0
        self.total_speed_error = 0.0
        self.previous_speed_error = 0.0
        self.tracker = None
        self.tracker_changed = False

        # DDS QOS Setup - important for detemining lag and packet drop behavior
        qos_profile = QoSProfile(depth=1)
        qos_profile.history = QoSHistoryPolicy.KEEP_LAST

        # subscribers
        self.sub_path = self.create_subscription(
            Path, 'path', self.path_callback, qos_profile)
        self.sub_state = self.create_subscription(
            VehicleState, 'state', self.state_callback, qos_profile)

        # publishers
        self.pub_vehicle_cmd = self.create_publisher(
            VehicleInput, 'vehicle_cmd', qos_profile)
        self.timer = self.create_timer(
            1/self.freq, self.pub_callback)

        #visualization setup
        if(self.visualize):
            matplotlib.use("TKAgg")
        else:
            matplotlib.use("Agg")

        self.fig, self.ax = plt.subplots()
        plt.title("Controller - Path Follower")
        self.patches = []
        self.ax.set_xlabel("X [m] (Global Frame)")
        self.ax.set_ylabel("Y [m] (Global Frame)")
        self.path_points = None

    def path_callback(self, msg):
        self.path = msg

        path_pts = []
        scattering_pts = []
        for p in self.path.poses:
            pt = chrono.ChVectorD(p.pose.position.x,p.pose.position.y,0)
            path_pts.append(pt)
            scattering_pts.append([p.pose.position.x,p.pose.position.y])
        if(self.visualize or self.save):
            scattering_pts = np.asarray(scattering_pts)

            if(self.path_points == None):
                self.path_points = self.ax.scatter(scattering_pts[:,0],scattering_pts[:,1])
            else:
                self.path_points.set_offsets(scattering_pts)

            # self.ax.scatter(scattering_pts[:,0],scattering_pts[:,1])
            self.ax.set_xlim(np.min(scattering_pts[:,0])-self.vis_with_buffer,np.max(scattering_pts[:,0])+self.vis_with_buffer)
            self.ax.set_ylim(np.min(scattering_pts[:,1])-self.vis_with_buffer,np.max(scattering_pts[:,1])+self.vis_with_buffer)

        if(len(path_pts) < 3):
            return

        self.tracker = chrono.ChBezierCurveTracker(chrono.ChBezierCurve(path_pts),False)
        # self.tracker.m_maxNumIters = 20000
        # self.tracker_changed = True

    def state_callback(self, msg):
        self.vehicle_state = msg

    # callback to run a loop and publish data this class generates
    def pub_callback(self):
        if(self.tracker == None):
            return
        # find the location of the sentinel point from state and lookahead
        pos = np.array([self.vehicle_state.position[0],self.vehicle_state.position[1],0])
        heading = np.array([self.vehicle_state.heading[0],self.vehicle_state.heading[1],0])
        if(np.linalg.norm(heading) > 0):
            heading = heading/np.linalg.norm(heading)
        sentinel = pos + heading * self.lookahead
        ch_pos = chrono.ChVectorD(pos[0],pos[1],pos[2])
        ch_sentinel = chrono.ChVectorD(sentinel[0],sentinel[1],sentinel[2])
        ch_heading = chrono.ChVectorD(heading[0],heading[1],heading[2])

        # find closest point between sentinel and path
        target = chrono.ChVectorD(0,0,0)
        res = self.tracker.calcClosestPoint(ch_sentinel, target)

        # self.get_logger().info("Curve track return: '%s'" % (str(res)))

        if(self.visualize or self.save):
            [p.remove() for p in self.patches]
            self.patches.clear()

            #plot heading with blue arrow at vehicle position
            scale = 5
            arr = patches.Arrow(pos[0],pos[1],scale*heading[0],scale*heading[1],width=2,color='b')
            self.ax.add_patch(arr)
            self.patches.append(arr)

            #plot vehicle point in blue
            circ = patches.Circle(pos[0:2],radius=1,color='b')
            self.ax.add_patch(circ)
            self.patches.append(circ)

            #plot sentinel point in red
            circ = patches.Circle(sentinel[0:2],radius=1,color='r')
            self.ax.add_patch(circ)
            self.patches.append(circ)

            #plot target point in green
            circ = patches.Circle([target.x,target.y],radius=1,color='g')
            self.ax.add_patch(circ)
            self.patches.append(circ)

            if(self.visualize):
                plt.draw()
                plt.pause(0.0001)
            elif(self.save):
                plt.savefig(os.path.join(self.output_dir,"frame_{}.png".format(self.frame_number)))
                self.frame_number += 1

        # self.get_logger().info("Sentinel: '%s', Target: '%s'" % (str(sentinel),str(target)))
        error_vector = ch_sentinel - target
        steering_error = error_vector.Length() if error_vector.Cross(ch_heading).z > 0 else -error_vector.Length()
        self.total_steering_error += .5 * (steering_error + self.previous_steering_error) / self.freq
        steering = self.steering_kp * steering_error + self.steering_ki*self.total_steering_error + \
            self.steering_kd * ((steering_error - self.previous_steering_error) * self.freq)
        self.previous_steering_error = steering_error

        # self.get_logger().info("Steering error: '%s', steering: '%s'" % (str(steering_error),str(steering)))

        # calculate throttle input from current speed and target
        speed_error = self.target_speed - \
            np.linalg.norm(self.vehicle_state.velocity)
        self.total_speed_error += .5 * (speed_error + self.previous_speed_error) / self.freq
        throttle = self.speed_kp * speed_error + self.speed_ki*self.total_speed_error + self.speed_kd*(speed_error - self.previous_speed_error) * self.freq
        self.previous_speed_error = speed_error

        # self.get_logger().info("Speed error: '%s', throttle: '%s'" % (str(speed_error),str(throttle)))

        # pack and publish vehicle control message
        msg = VehicleInput()
        msg.steering = np.clip(steering,-1.0,1.0)
        msg.throttle = np.clip(throttle,0.0,1.0)
        msg.braking = 0.0
        self.pub_vehicle_cmd.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    control = ControlPathFollowerNode()
    rclpy.spin(control)
    control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
