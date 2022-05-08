#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile
from custom_msgs.msg import VehicleState, ObjectList, Object
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

from ament_index_python.packages import get_package_share_directory

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy.integrate import odeint


class PathPlanningNode(Node):
    def __init__(self):
        super().__init__('path_planning_node')

        # update frequency of this node
        self.freq = 10.0

        #tracked variables
        self.vehicle_state = VehicleState()
        self.objects = ObjectList()
        self.obstacles = []
        self.position = np.zeros(2)
        self.heading = np.array([1,0])

        # Read in share directory location
        package_share_directory = get_package_share_directory('control_stack')

        self.declare_parameter('visualize', False)
        self.visualize = self.get_parameter(
            'visualize').get_parameter_value().bool_value

        self.declare_parameter('path_resolution', 10.0)
        self.path_resolution = self.get_parameter(
            'path_resolution').get_parameter_value().double_value

        #distance in meters between grid samples
        self.declare_parameter('grid_resolution', 5.0)
        self.grid_resolution = self.get_parameter(
            'grid_resolution').get_parameter_value().double_value

        # number of samples along each side of the grid
        self.declare_parameter('grid_samples', 21)
        self.grid_samples = self.get_parameter(
            'grid_samples').get_parameter_value().integer_value

        #for extending the field behind vehicle in case pointing away from target
        self.declare_parameter('grid_buffer', 10.0)
        self.grid_buffer = self.get_parameter(
            'grid_buffer').get_parameter_value().double_value

        #magnitude of target potential in field
        self.declare_parameter('max_goal_weight', 10.0)
        self.max_goal_weight = self.get_parameter(
            'max_goal_weight').get_parameter_value().double_value

        # magnitude of obstacle potential in field
        self.declare_parameter('obstacle_weight', 10.0)
        self.obstacle_weight = self.get_parameter(
            'obstacle_weight').get_parameter_value().double_value

        #radius of max obstacle penalty
        self.declare_parameter('obstacle_radius', 2.0)
        self.obstacle_radius = self.get_parameter(
            'obstacle_radius').get_parameter_value().double_value

        # distance where field is effected by obstacle
        # this needs to be at least a few times larger than resolution
        self.declare_parameter('obstacle_effect_radius', 20.0)
        self.obstacle_effect_radius = self.get_parameter(
            'obstacle_effect_radius').get_parameter_value().double_value

        self.declare_parameter('heading_weight', 10.0)
        self.heading_weight = self.get_parameter(
            'heading_weight').get_parameter_value().double_value

        self.declare_parameter('heading_effective_radius', 20.0)
        self.heading_effective_radius = self.get_parameter(
            'heading_effective_radius').get_parameter_value().double_value

        #target location
        self.declare_parameter('target', [350.0,350.0])
        self.target = self.get_parameter(
            'target').get_parameter_value().double_array_value

        self.declare_parameter('save', False)
        self.save = self.get_parameter(
            'save').get_parameter_value().bool_value

        self.declare_parameter('save_name', "planner_output")
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
        self.sub_state = self.create_subscription(
            VehicleState, 'state', self.state_callback, qos_profile)
        self.sub_objects = self.create_subscription(
            ObjectList, 'objects', self.objects_callback, qos_profile)

        # publishers
        self.pub_path = self.create_publisher(Path, 'path', qos_profile)
        self.timer = self.create_timer(1/self.freq, self.pub_callback)

        #visualization setup
        if(self.visualize):
            self.fig, self.ax = plt.subplots()
            plt.title("Path Planner")
            self.patches = []
            self.ax.set_xlim((-500,500))
            self.ax.set_ylim((-500,500))
            self.ax.set_xlabel("X [m] (Global Frame)")
            self.ax.set_ylabel("Y [m] (Global Frame)")
            self.path_points = None


    # function to process data this class subscribes to
    def state_callback(self, msg):
        self.vehicle_state = msg

    def objects_callback(self, msg):
        self.objects = msg

    def WeightVecToTarget(self,p1):
        dist = np.linalg.norm(self.target-p1)
        direction = self.target-p1
        if(dist > 1e-3):
           direction = direction / dist
        magnitude = np.clip(dist,0.0,self.max_goal_weight)
        return magnitude * direction


    def WeightVecToObstacle(self,p1,obstacle):
        dist =  np.linalg.norm(obstacle-p1)
        direction = p1-obstacle
        if(dist > 0):
            direction = direction/dist
        magnitude = self.obstacle_weight - self.obstacle_weight / (self.obstacle_radius-self.obstacle_effect_radius) * (self.obstacle_radius - dist)
        magnitude = np.clip(magnitude,0,self.obstacle_weight)
        return magnitude * direction

    def WeightVecFromHeading(self,p1,pos,heading):
        dist =  np.linalg.norm(pos-p1)
        magnitude = self.heading_weight - (self.heading_weight / self.heading_effective_radius) * dist
        magnitude = np.clip(magnitude,0,self.heading_weight)
        direction = heading / np.linalg.norm(heading)
        return magnitude * direction

    def Grad(self,pos,t):
        gradient = self.WeightVecToTarget(pos)
        for ob in self.obstacles:
            gradient += self.WeightVecToObstacle(pos,ob.position[0:2])
        # gradient += self.WeightVecFromHeading(pos,self.position,self.heading)
        return gradient

    def plan_path(self,start,target,obstacles):
        t = np.linspace(0.1,30.0,200)
        line_segments = odeint(self.Grad, self.position, t)

        #get a point for the path every [path resolution] meters
        path = line_segments
        if(self.visualize):
            self.ax.clear()
            self.ax.set_xlim((-500,500))
            self.ax.set_ylim((-500,500))
            self.ax.set_xlabel("X [m] (Global Frame)")
            self.ax.set_ylabel("Y [m] (Global Frame)")

            self.ax.scatter(target[0],target[1],s=200,c='g')
            for ob in obstacles:
                self.ax.scatter(ob.position[0],ob.position[1],s=200,c='r')

            self.ax.scatter(path[:,0],path[:,1],color='cyan')
            self.ax.scatter(path[0,0],path[0,1],s=100,c='b')

        if(self.visualize):
            plt.draw()
            plt.pause(0.0001)
        elif(self.save):
            plt.savefig(os.path.join(self.output_dir,"frame_{}.png".format(self.frame_number)))
            self.frame_number += 1

        return path

    # callback to run a loop and publish data this class generates
    def pub_callback(self):
        #for now, create a path us and the target using a potential field
        # self.get_logger().info("vehicle position=%s" % (str(self.vehicle_state.position[0:2])))

        self.obstacles = self.objects.objects
        self.heading = self.vehicle_state.heading[0:2]
        self.position = self.vehicle_state.position[0:2]

        pts = self.plan_path(self.vehicle_state.position[0:2],self.target,self.objects.objects)

        #package and publish the path
        msg = Path()
        for p in pts:
            pt = PoseStamped()
            pt.pose.position.x = p[0]
            pt.pose.position.y = p[1]
            msg.poses.append(pt)

        self.pub_path.publish(msg)
        # self.get_logger().info('Publishing "%s"' % msg)


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlanningNode()
    rclpy.spin(planner)
    planner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
