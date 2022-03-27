import numpy as np
import rospy
import math

class MotionModel:

    def __init__(self):

        ####################################
        # Pre-computation
        self.deterministic = rospy.get_param("~deterministic")
        self.params = [1.0, 1.0, 0.5]

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
        shape = (particles.shape[0], 1)
        noise_std = abs(odometry) * np.array(self.params)
        odometry_noisy = np.concatenate([np.random.normal(odometry[0], noise_std[0], shape),
                                        np.random.normal(odometry[1], noise_std[1], shape),
                                        np.random.normal(odometry[2], noise_std[2], shape)], axis = 1)
        
        x = particles[:,0]
        y = particles[:,1]
        theta = particles[:,2]
        
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        dx = odometry_noisy[:,0] * cos_theta - odometry_noisy[:,1] * sin_theta
        dy = odometry_noisy[:,0] * sin_theta + odometry_noisy[:,1] * cos_theta
        dtheta = odometry_noisy[:,2]

        particles[:,0] = x + dx
        particles[:,1] = y + dy
        particles[:,2] = theta + dtheta

        return particles

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
