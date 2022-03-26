import numpy as np
import rospy
import math

class MotionModel:

    def __init__(self):

        ####################################
        # Pre-computation
        self.deterministic = rospy.get_param("~deterministic")

        ####################################

    def deterministic_evaluate(self, particles, odometry):
        dx_car = odometry[0]
        dy_car = odometry[1]
        dtheta_car = odometry[2]
        
        x = particles[:,0]
        y = particles[:,1]
        theta = particles[:,2]

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        dx = dx_car * cos_theta - dy_car * sin_theta
        dy = dx_car * sin_theta + dy_car * cos_theta

        particles[:,0] = x + dx
        particles[:,1] = y + dy
        particles[:,2] = theta + dtheta_car
        
        return particles

    def noisy_evaluate(self, particles, odometry):
        pass

    def evaluate(self, particles, odometry):
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles : An Nx3 matrix of the form:
            
                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            odometry : A 3-vector [dx dy dtheta]

        returns :
            particles: An updated matrix of the same size
        """
        
        ####################################        
        if self.deterministic:
            return self.deterministic_evaluate(particles, odometry)
        else:
            return self.noisy_evaluate(particles, odometry)

        ####################################
