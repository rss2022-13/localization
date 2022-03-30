#!/usr/bin/env python2

import rospy
from sensor_model import SensorModel
from motion_model import MotionModel

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray, Pose
from tf.transformations import quaternion_from_euler, euler_from_quaternion, quaternion_matrix
from visualization_msgs.msg import MarkerArray, Marker
import numpy as np
from scipy.stats import circmean
# import threading


class ParticleFilter:
    # sensor_model = None
    # particles = None

    def __init__(self):
        # Get parameters
        self.particle_filter_frame = rospy.get_param("~particle_filter_frame")

        self.deterministic = rospy.get_param("~deterministic", True)

        # Initialize publishers/subscribers
        #
        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.
        #scan_topic = rospy.get_param("~scan_topic", "/scan")
        odom_topic = rospy.get_param("~odom_topic", "/odom")
        #self.laser_sub = rospy.Subscriber(scan_topic, LaserScan,
         #                                 self.interpret_scan,
          #                                queue_size=1)
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry,
                                         self.interpret_odometry,
                                         queue_size=1)

        # self._lock = threading.Lock()
        # self.odom_thread = None
        # self.laser_thread = None

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.
        self.pose_sub = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped,
                                         self.initialize_particles,
                                         queue_size=1)

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.
        self.odom_pub = rospy.Publisher("/pf/pose/odom", Odometry, queue_size=1)
        self.odom_estimate = Odometry()
        self.frame_id = "/map"

        # For visualizing the points
        self.vis_pub = rospy.Publisher("/particle_visualization", MarkerArray, queue_size=1)

        # Initialize the models
        self.motion_model = MotionModel()
        #self.sensor_model = SensorModel()
        self.probs = None
        # Implement the MCL algorithm
        # using the sensor model and the motion model
        #
        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        # Publish a transformation frame between the map
        # and the particle_filter_frame.
        self.particles = None
        self.particles_set = False
        self.num_particles = rospy.get_param("~num_particles", 200)
        self.markers = None

    def update_pose(self):
        ''' 
            For averaging particles:
                Probably fine to use arithmetic mean for x,y for all particles  
                For angles, could use Circular mean. Requires summing over every angle and converting all to polar 
                with atan2(sum(sin(theta)), sum(cos(theta)))  
                    Not that time intensive if we use np functions
                Could also use a MSE regression which essentially finds modes of the distribution
                    Would be definitely more time intensive unless I find a way to use fast libraries
        '''
        x = np.mean(self.particles[:, 0])
        y = np.mean(self.particles[:, 1])

        theta = circmean(self.particles[:, 2])

        n = self.particles.shape[0]
 
        if not self.markers:
            self.markers = [0]*n
        for i in range(n):
            pose = self.markers[i]
            if not self.markers[i]:
                pose = Marker()
            pose.header.frame_id = self.frame_id
            pose.type = Marker.CYLINDER
            pose.action = Marker.ADD
            pose.scale.x = .1
            pose.scale.y = .1
            pose.scale.z = .1
            pose.color.a = 1.0
            pose.color.r = 1.0
            pose.color.g = .5
            pose.pose.position.x = self.particles[i,0]
            pose.pose.position.y = self.particles[i,1]
            #print(pose.pose.position.x, pose.pose.position.y)
            pose.pose.position.z = 0
            pose.header.stamp = rospy.Time.now()
            
            ori = quaternion_from_euler(0.0,0.0,self.particles[i,2])

            pose.pose.orientation.x = ori[0]
            pose.pose.orientation.y = ori[1]
            pose.pose.orientation.z = ori[2]
            pose.pose.orientation.w = ori[3]

            self.markers[i] = pose
        
        visualization = MarkerArray()
        visualization.markers = self.markers

        
        id = 0
        for m in visualization.markers:
            m.id = id
            id += 1
        
        
        quaternion = quaternion_from_euler(0.0, 0.0, theta)


        self.odom_estimate.header.frame_id = self.frame_id
        self.odom_estimate.header.stamp = rospy.Time.now()
        self.odom_estimate.pose.pose.orientation.x = quaternion[0]
        self.odom_estimate.pose.pose.orientation.y = quaternion[1]
        self.odom_estimate.pose.pose.orientation.z = quaternion[2]
        self.odom_estimate.pose.pose.orientation.w = quaternion[3]
        self.odom_estimate.pose.pose.position.x = x
        self.odom_estimate.pose.pose.position.y = y
        self.odom_estimate.pose.pose.position.z = 0.0

        self.odom_pub.publish(self.odom_estimate)
        self.vis_pub.publish(visualization)

    def initialize_particles(self, pose):
        # sample regions of the map that allow us to place points, then uniformly distribute with random thetas
        x = np.random.choice(np.linspace(
            pose.pose.pose.position.x - 5, pose.pose.pose.position.x + 5, 30), self.num_particles)
        x = x.reshape((self.num_particles, 1))
        y = np.random.choice(np.linspace(
            pose.pose.pose.position.y - 5, pose.pose.pose.position.y + 5, 30), self.num_particles)
        y = y.reshape((self.num_particles, 1))

        thetas = np.random.choice(np.linspace(0, 2*np.pi, 300), 200)
        thetas = thetas.reshape((self.num_particles, 1))

        self.particles = np.hstack((x, y, thetas))
        #print(self.particles)
        self.particles_set = True

    # def scan_callback(self,scan):
        # if not self.particles_set:
        #     return None
        # self.laser_thread = threading.Thread(target=ParticleFilter.interpret_scan, name="Laser-Thread", args=(self,scan))
        # self.laser_thread.start()

    def interpret_scan(self, scan):
        # with self._lock:
        self.probs = self.sensor_model.evaluate(self.particles, np.array(scan.ranges))
        # normalizing here
        self.probs = np.divide(self.probs, np.sum(self.probs))

        # resampling based on computed probabilities

        self.particles = self.particles[np.random.choice(len(self.particles), len(self.particles), p=self.probs)]
        self.update_pose()

    # def motion_callback(self,odom):
        # if not self.particles_set:
        #     return None
        # self.odom_thread = threading.Thread(target=ParticleFilter.interpret_odometry, name="Odom-Thread", args=(self,odom))
        # self.odom_thread.start()

    def interpret_odometry(self, odom):
        # with self._lock:
        #quat = [odom.pose.pose.orientation.x,odom.pose.pose.orientation.y,odom.pose.pose.orientation.z, odom.pose.pose.orientation.w]
        vector = [odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.angular.z]
        self.particles = self.motion_model.evaluate(self.particles, vector)
        self.update_pose()

if __name__ == "__main__":
    rospy.init_node("particle_filter")
    pf = ParticleFilter()
    #print(pf.particles)
    rospy.spin()
