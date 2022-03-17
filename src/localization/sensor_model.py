import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scan_simulator_2d import PyScanSimulator2D

import rospy
import tf
from nav_msgs.msg import OccupancyGrid
from tf.transformations import quaternion_from_euler

class SensorModel:


    def __init__(self):
        # Fetch parameters
        self.map_topic = rospy.get_param("~map_topic")
        self.num_beams_per_particle = rospy.get_param("~num_beams_per_particle")
        self.scan_theta_discretization = rospy.get_param("~scan_theta_discretization")
        self.scan_field_of_view = rospy.get_param("~scan_field_of_view")

        ####################################
        # TODO
        # Adjust these parameters
        self.alpha_hit = 0.74
        self.alpha_short = 0.07
        self.alpha_max = 0.07
        self.alpha_rand = 0.12
        self.sigma_hit = 8.0
        self.z_max = 200

        # Your sensor table will be a `table_width` x `table_width` np array:
        self.table_width = 201
        ####################################

        # Precompute the sensor model table
        self.sensor_model_table = np.zeros((self.table_width, self.table_width))
        self.precompute_sensor_model()

        # Create a simulated laser scan
        self.scan_sim = PyScanSimulator2D(
                self.num_beams_per_particle,
                self.scan_field_of_view,
                0, # This is not the simulator, don't add noise
                0.01, # This is used as an epsilon
                self.scan_theta_discretization) 

        # Subscribe to the map
        self.map = None
        self.map_set = False
        rospy.Subscriber(
                self.map_topic,
                OccupancyGrid,
                self.map_callback,
                queue_size=1)
    
    def calc_prob(self, z, d):
        phit = 0.0
        pshort = 0.0
        pmax = 0.0
        pmax = 0.0

        # MAKE SURE TO CHANGE z_max to the real value
        if 0 <= z <= self.z_max:
            phit = 1.0/(np.sqrt(2*np.pi*self.sigma_hit**2)) * np.exp(-(z-d)**2/(2*self.sigma_hit**2))
            prand = 1.0/self.z_max

        if 0 <= z <= d and d != 0:
            pshort = 2.0/d*(1-z/d)

        if z == self.z_max:
            pmax = 1.0

        
        return self.alpha_short * pshort + self.alpha_max * pmax + self.alpha_rand * prand, phit

    def precompute_sensor_model(self):
        """
        Generate and store a table which represents the sensor model.
        
        For each discrete computed range value, this provides the probability of 
        measuring any (discrete) range. This table is indexed by the sensor model
        at runtime by discretizing the measurements and computed ranges from
        RangeLibc.
        This table must be implemented as a numpy 2D array.

        Compute the table based on class parameters alpha_hit, alpha_short,
        alpha_max, alpha_rand, sigma_hit, and table_width.

        args:
            N/A
        
        returns:
            No return type. Directly modify `self.sensor_model_table`.
        """
        p_hits = np.zeros((self.table_width,))
        
        for d in range(self.z_max+1):
            for z in range(self.z_max+1):
                #calculate probability initially
                self.sensor_model_table[d,z], p_hits[z] = self.calc_prob(z,d)

            #normalize p_hit values if not normal
            if np.sum(p_hits) != 1:
                p_hits = np.divide(p_hits,np.sum(p_hits))

            #use normalized vals to compute probabilities
            p_hits = p_hits * self.alpha_hit

            #add these probabilities to the current values
            self.sensor_model_table[d:d+1,:] = np.add(self.sensor_model_table[d:d+1,:],p_hits)
        
        #once all probabilities filled, need to normalize each row
        for d in range(self.z_max+1):
            if np.sum(self.sensor_model_table[d:d+1,:]) != 1:
                self.sensor_model_table[d:d+1,:] = np.divide(self.sensor_model_table[d:d+1,:],np.sum(self.sensor_model_table[d:d+1,:]))

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # Make data.
        X = np.arange(0, 201, 1)
        Y = np.arange(0, 201, 1)
        X, Y = np.meshgrid(X, Y)

        # Plot the surface.
        surf = ax.plot_surface(X, Y, self.sensor_model_table, linewidth=0, antialiased=False)

        plt.show()
                


    def evaluate(self, particles, observation):
        """
        Evaluate how likely each particle is given
        the observed scan.

        args:
            particles: An Nx3 matrix of the form:
            
                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            observation: A vector of lidar data measured
                from the actual lidar.

        returns:
           probabilities: A vector of length N representing
               the probability of each particle existing
               given the observation and the map.
        """

        if not self.map_set:
            return

        ####################################
        # TODO
        # Evaluate the sensor model here!
        #
        # You will probably want to use this function
        # to perform ray tracing from all the particles.
        # This produces a matrix of size N x num_beams_per_particle 

        scans = self.scan_sim.scan(particles)

        ####################################

    def map_callback(self, map_msg):
        # Convert the map to a numpy array
        self.map = np.array(map_msg.data, np.double)/100.
        self.map = np.clip(self.map, 0, 1)

        # Convert the origin to a tuple
        origin_p = map_msg.info.origin.position
        origin_o = map_msg.info.origin.orientation
        origin_o = tf.transformations.euler_from_quaternion((
                origin_o.x,
                origin_o.y,
                origin_o.z,
                origin_o.w))
        origin = (origin_p.x, origin_p.y, origin_o[2])

        # Initialize a map with the laser scan
        self.scan_sim.set_map(
                self.map,
                map_msg.info.height,
                map_msg.info.width,
                map_msg.info.resolution,
                origin,
                0.5) # Consider anything < 0.5 to be free

        # Make the map set
        self.map_set = True

        print("Map initialized")
