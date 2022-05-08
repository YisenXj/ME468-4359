#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile
from sensor_msgs.msg import Image, PointCloud2
from custom_msgs.msg import VehicleState, ObjectList, Object

from ament_index_python.packages import get_package_share_directory

# import torch, torchvision, time
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn import linear_model
import scipy.cluster as cl
import os

class ObjectRecognitionNode(Node):
    def __init__(self):
        super().__init__('object_recognition_node')

        # update frequency of this node
        self.freq = 10.0

        # data that will be used by this class
        self.state = VehicleState()
        self.pointcloud = np.array([])

        # Read in share directory location
        package_share_directory = get_package_share_directory('control_stack')

        self.declare_parameter('visualize', False)
        self.visualize = self.get_parameter(
            'visualize').get_parameter_value().bool_value

        self.declare_parameter('ransac_residual', 1.0)
        self.ransac_residual = self.get_parameter(
            'ransac_residual').get_parameter_value().double_value

        self.declare_parameter('cluster_threshold', 5.0)
        self.cluster_threshold = self.get_parameter(
            'cluster_threshold').get_parameter_value().double_value

        self.declare_parameter('min_points_for_cluster', 5)
        self.min_points_for_cluster = self.get_parameter(
            'min_points_for_cluster').get_parameter_value().integer_value

        self.declare_parameter('save', False)
        self.save = self.get_parameter(
            'save').get_parameter_value().bool_value

        self.declare_parameter('save_name', "perception_output")
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
        self.sub_lidar = self.create_subscription(
            PointCloud2, 'lidar', self.lidar_callback, qos_profile)

        # publishers
        self.pub_objects = self.create_publisher(
            ObjectList, 'objects', qos_profile)
        self.timer = self.create_timer(1/self.freq, self.pub_callback)

        if(self.visualize):
            matplotlib.use("TKAgg")
        else:
            matplotlib.use("Agg")

        self.fig, self.ax = plt.subplots()
        plt.title("Object Recognition - Point Cloud Clustering")
        self.ax.set_xlim(-150,150)
        self.ax.set_ylim(-150,150)
        self.ax.set_xlabel("X [m] (Global Frame)")
        self.ax.set_ylabel("Y [m] (Global Frame)")
        self.patches = []

        self.frame = 0

    # function to process data this class subscribes to: vehicle state information
    def state_callback(self, msg):
        self.state = msg

    # function for saving the points when a point cloud msg is published to /lidar
    def lidar_callback(self, msg):
        h = msg.height
        w = msg.width

        c = 4 #4 floats per points
        data = np.frombuffer(msg.data,dtype=np.float32, count=h*w*c)

        self.pointcloud = np.reshape(data,(w,c))

    #transform points from local to global coordinate system
    def transform_points(self, points, vehicle_state):
        if(points.shape[0]<1): #only transform if clusters exist
            return points

        heading = vehicle_state.heading[0:2]
        if(np.linalg.norm(heading) < 1e-3):
            return points

        heading = heading / np.linalg.norm(heading)
        ang = np.arctan2(vehicle_state.heading[1], vehicle_state.heading[0])

        points[:,0] = np.cos(ang) * points[:,0] - np.sin(ang) * points[:,1]
        points[:,1] = np.sin(ang) * points[:,0] + np.cos(ang) * points[:,1]

        points[:,0] = points[:,0] + vehicle_state.position[0]
        points[:,1] = points[:,1] + vehicle_state.position[1]

        return points

    def ground_removal(self,pointcloud):
        ransac = linear_model.RANSACRegressor(residual_threshold=self.ransac_residual)
        ransac.fit(pointcloud[:,0:2], pointcloud[:,2])

        outlier_mask = np.logical_not(ransac.inlier_mask_)
        data = pointcloud[outlier_mask,0:3] #also removes intensity

        return data

    def cluster_objects(self,pointcloud):
        
        if(pointcloud.shape[0]<5): #only cluster if we have a reasonable number of points
            return np.array([])

        clustered_data = cl.hierarchy.fclusterdata(pointcloud,self.cluster_threshold,criterion='distance')

        #get cluster means
        clusters = []
        for i in range(1,np.max(clustered_data)+1):
            subset = pointcloud[clustered_data==i,:]
            if(len(subset) > self.min_points_for_cluster):
                clusters.append(np.mean(subset,axis=0))

        return np.asarray(clusters)

    # callback to run a loop and publish data this class generates
    def pub_callback(self):

        cluster_means = []
        if(len(self.pointcloud) > 4):
            #remove ground point from the point cloud
            points = self.ground_removal(self.pointcloud)

            #cluster the remaining objects by distance
            cluster_means = self.cluster_objects(points)

            #transform the clusters from local to global coordinates
            cluster_means = self.transform_points(cluster_means,self.state)

        msg = ObjectList()

        for c in cluster_means:
            obj = Object()
            obj.position = c.astype(np.float64)
            msg.objects.append(obj)
        self.pub_objects.publish(msg)

        if(self.visualize or self.save):
            [p.remove() for p in self.patches]
            self.patches.clear()

            radius = 5
            circ = patches.Circle(self.state.position[0:2],radius=radius,color='b')
            self.ax.add_patch(circ)
            self.patches.append(circ)

            scale = 5
            arr = patches.Arrow(self.state.position[0],self.state.position[1],scale*self.state.heading[0],scale*self.state.heading[1],width=2,color='b')
            self.ax.add_patch(arr)
            self.patches.append(arr)

            self.ax.set_xlim(self.state.position[0]-150,self.state.position[0]+150)
            self.ax.set_ylim(self.state.position[1]-150,self.state.position[1]+150)

            for l in cluster_means:
                circ = patches.Circle(l,radius=radius,color='r',fill=False)
                self.ax.add_patch(circ)
                self.patches.append(circ)

            if(self.visualize):
                plt.draw()
                plt.pause(0.0001)
            elif(self.save):
                plt.savefig(os.path.join(self.output_dir,"frame_{}.png".format(self.frame_number)))
                self.frame_number += 1

def main(args=None):
    rclpy.init(args=args)
    recognition = ObjectRecognitionNode()
    rclpy.spin(recognition)
    recognition.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
