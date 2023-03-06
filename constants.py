import numpy as np

<<<<<<< HEAD
# Choose whether or not the asteroids split
do_asteroids_split = False
# Choose whether or not the player loses points if they hit the edge (if not set to 0)
points_lost = 3

=======
>>>>>>> cd3ffa7b731c5a8f51e16289c3220cae4671ad1d
# Screen dimensions
SCREEN_HEIGHT = 800
SCREEN_WIDTH = 800

# Factor multiplying the standard 60 FPS
FPS_FACTOR = 0.1

# Frames per second
FPS = 60 * FPS_FACTOR

# Add one to the score after each second survived
SCORE_RATE = FPS

# Asteroid variables
<<<<<<< HEAD
SECONDS_BETWEEN_ASTEROIDS = 15
=======
SECONDS_BETWEEN_ASTEROIDS = 4
>>>>>>> cd3ffa7b731c5a8f51e16289c3220cae4671ad1d
# Number of frames after which another asteroid spawns
FRAMES_BETWEEN_ASTEROIDS = SECONDS_BETWEEN_ASTEROIDS * FPS
# Initial minimum, average and variance of the asteroid speeds
MIN_ASTEROID_SPEED_INITIAL = 0.1 / FPS_FACTOR
AVG_ASTEROID_SPEED_INITIAL = 0.2 / FPS_FACTOR
VAR_ASTEROID_SPEED_INITIAL = 0.2 / FPS_FACTOR
# The minimum, average and variance of the asteroid speed increases by the following each second
MIN_ASTEROID_GRADIENT = (1 / 500)
AVG_ASTEROID_GRADIENT = (1 / 300)
VAR_ASTEROID_GRADIENT = np.radians(1 / 200)
# Variance in the angle of the spawned asteroids
<<<<<<< HEAD
#asteroid_angle_var = np.radians(20)
asteroid_angle_var = np.radians(25)
ASTEROID_SIZES = (96, 64, 32)
ASTEROID_MASSES = (150, 60, 10)
# Points scored for destroying an asteroid
ASTEROID_SCORES = (20, 30, 15)
#ASTEROID_SCORES = (5, 10, 15)
=======
asteroid_angle_var = np.radians(20)
ASTEROID_SIZES = (96, 64, 32)
ASTEROID_MASSES = (150, 60, 10)
# Points scored for destroying an asteroid
ASTEROID_SCORES = (5, 10, 15)
>>>>>>> cd3ffa7b731c5a8f51e16289c3220cae4671ad1d
# The mean and variance of the deviation in angle of the two new asteroids from the original bigger asteroid (after
# it is destroyed)
COLLISION_ANGLE_DEV_MEAN = np.radians(20)
COLLISION_ANGLE_DEV_VAR = np.radians(5)
# Upper bound on the probability of spawning an asteroid of maximum size
MAX_SPAWN_PROB = 0.2
# Frame at which the maximum probability of a large asteroid spawning is reached
FRAME_MAX_PROB_IS_REACHED = 90 * FPS
# Gradient of the linear mapping between frame number and probability of spawning a large asteroid
SPAWN_GRADIENT = MAX_SPAWN_PROB / FRAME_MAX_PROB_IS_REACHED

# Laser variables
LASER_SIZE = 6
LASER_SPEED = 6 / FPS_FACTOR
LASER_MASS = 1
# Number of frames after which you can reload the laser gun
# FRAMES_BETWEEN_RELOADS = 50 * FPS_FACTOR
FRAMES_BETWEEN_RELOADS = 500 * FPS_FACTOR

# Parameters controlling how the spaceship moves
SHIPSIZE = (30, 60)
ACCELERATION_FORWARDS = 0.05 / FPS_FACTOR
ACCELERATION_BACKWARDS = 0.03 / FPS_FACTOR
SPEED_DAMPING = 0.99 ** (1 / FPS_FACTOR)
ANGULAR_DAMPING = 0.95 ** (1 / FPS_FACTOR)
ANGLE_CHANGE = np.radians(0.1) / FPS_FACTOR
EDGE_DAMPING = 0.995 ** (1 / FPS_FACTOR)

# Factor by which the explosion is bigger than the original exploding object
EXPLOSION_DILATION = 1.5
EXPLOSION_DURATION = 5

# Text
TEXT_COLOUR_1 = (0, 100, 255)
TEXT_COLOUR_2 = (255, 100, 100)