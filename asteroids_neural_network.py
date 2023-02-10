# Neural network learns to play asteroids using a genetic algorithm
# Remove all graphics because not needed during the training of the NN


# Things left to do:
# Modify the rect function in the Player class
# Create the forward propagation function
# Create the inputs translator

# Asteroids game
import numpy as np
import pygad as pygad
from numpy.linalg import norm
import pygame as g
from pygame.locals import (K_UP, K_DOWN, K_LEFT, K_RIGHT, K_ESCAPE, K_SPACE)
import random
import math
import pygad.gann
import pygad.nn
import tensorflow.keras
import pygad.kerasga
import pygad
import time

# Initialize
#g.init()

# Screen dimensions
SCREEN_HEIGHT = 800
SCREEN_WIDTH = 800

# Factor multiplying the standard 60 FPS
FPS_FACTOR = 0.1

# Frames per second
FPS = 60 * FPS_FACTOR

# Add one to score after each second survived
SCORE_RATE = FPS

# Asteroid variables
SECONDS_BETWEEN_ASTEROIDS = 2
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
asteroid_angle_var = np.radians(20)
ASTEROID_SIZES = (96, 64, 32)
ASTEROID_MASSES = (150, 60, 10)
# Points scored for destroying an asteroid
ASTEROID_SCORES = (1, 3, 10)
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
FRAMES_BETWEEN_RELOADS = 50 * FPS_FACTOR

# Parameters controlling how the spaceship moves
SHIPSIZE = (30, 60)
ACCELERATION_FORWARDS = 0.05 / FPS_FACTOR
ACCELERATION_BACKWARDS = 0.03 / FPS_FACTOR
SPEED_DAMPING = 0.99**(1/FPS_FACTOR)
ANGULAR_DAMPING = 0.95**(1/FPS_FACTOR)
ANGLE_CHANGE = np.radians(0.1) / FPS_FACTOR
EDGE_DAMPING = 0.995**(1/FPS_FACTOR)

def off_screen(rect, distance):
    # Function takes in a rect and returns True if the sprite is 'distance' away from the screen and False otherwise
    return rect.left < - distance \
           or rect.right > SCREEN_WIDTH + distance \
           or rect.top < - distance \
           or rect.bottom > SCREEN_HEIGHT + distance


def random_edge_and_angle(distance):
    # Generate a random point a given distance away from the screen frame
    # Also generates an angle such that the asteroid moves towards the centre with some normally distributed deviation
    edge = random.randint(0, 3)
    if edge == 0:
        x, y = (-distance, random.randint(-distance, SCREEN_HEIGHT + distance))
    elif edge == 1:
        x, y = (SCREEN_WIDTH + distance, random.randint(-distance, SCREEN_HEIGHT + distance))
    elif edge == 2:
        x, y = (random.randint(-distance, SCREEN_WIDTH + distance), -distance)
    elif edge == 3:
        x, y = (random.randint(-distance, SCREEN_WIDTH + distance), SCREEN_HEIGHT + distance)
    angle = math.atan2(y - SCREEN_HEIGHT / 2, SCREEN_WIDTH / 2 - x) + random.normalvariate(0, asteroid_angle_var)
    return (x, y), angle


def unit_vector(angle):
    # Takes an angle in radians and returns a unit vector in that direction
    return np.array((np.cos(angle), -np.sin(angle)))


def find_angle(vector):
    # returns the angle of the vector in radians
    return np.arctan2(-vector[1], vector[0])


def range_pi_to_pi(angle):
    # Takes an angle in radians in the range (0, 2pi) and puts it in the range (-pi, pi) while keeping 0 fixed
    return ((angle + np.pi) % (2 * np.pi)) - np.pi


def asteroid_size_choice(framenumber):
    threshold = min(MAX_SPAWN_PROB, framenumber * SPAWN_GRADIENT)
    return ASTEROID_SIZES[int(random.random() > threshold)]


def asteroid_split(asteroid_level, asteroid_velocity, laser_angle):
    # Calculates the trajectory of two smaller asteroids after a collision of a laser and larger asteroid
    # Momentum is conserved
    # Added explosive kinetic energy is generated randomly
    total_mom = (asteroid_velocity * ASTEROID_MASSES[asteroid_level] + LASER_SPEED * unit_vector(
        laser_angle) * LASER_MASS)
    total_angle = np.arctan2(-total_mom[1], total_mom[0])
    parallel_velocity = total_mom / (2 * ASTEROID_MASSES[asteroid_level + 1])
    # perpendicular_deviation = 0*random.uniform(avg_asteroid_speed*0.9, avg_asteroid_speed*1.3)*direction(np.degrees(total_angle - np.pi/2))
    # velocity1 = parallel_velocity - perpendicular_deviation
    # velocity2 = parallel_velocity + perpendicular_deviation
    angle_deviation = random.normalvariate(COLLISION_ANGLE_DEV_MEAN, COLLISION_ANGLE_DEV_VAR)
    return np.linalg.norm(parallel_velocity), total_angle + angle_deviation, total_angle - angle_deviation


class Player(g.sprite.Sprite):
    def __init__(self):
        super(Player, self).__init__()

        self.width, self.height = SHIPSIZE

        # Initial position in the centre of the screen
        self.position = np.array([SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2])
        # Initial speed of zero
        self.velocity = np.array([0, 0], dtype=float)

        # Initial angle pointing upwards
        self.angle = np.pi / 2
        # Initial angular frequency of zero
        self.frequency = 0

    @property
    #### DESPERATELY NEEDS REWRITING!!!!!!!!!!!! (does not rotate with the player!)
    def rect(self):
        return g.Rect(self.position[0] - self.width / 2, self.position[1] - self.height / 2, self.width, self.height)

    # Update the players position based on the keys pressed
    def update(self, pressed_keys):
        # speed decays by damping factor
        self.velocity = self.velocity * SPEED_DAMPING

        # Move forwards (key 1) or backwards (key 2)
        if pressed_keys[1]:
            self.velocity += unit_vector(self.angle) * ACCELERATION_FORWARDS
        if pressed_keys[2]:
            self.velocity += -unit_vector(self.angle) * ACCELERATION_BACKWARDS
        self.position += self.velocity

        # speed decays by damping factor
        self.frequency *= ANGULAR_DAMPING

        # Rotate using left and right keys (anticlockwise key 3, clockwise key 4)
        if pressed_keys[3]:
            self.frequency += ANGLE_CHANGE
        if pressed_keys[4]:
            self.frequency -= ANGLE_CHANGE
        self.angle = (self.angle + self.frequency) % (2 * np.pi)

        # Player not allowed off screen and loses all speed in corresponding direction
        if self.rect.left < 0:
            self.rect.left = 0
            self.velocity[0] *= -EDGE_DAMPING
            self.angle = np.pi - self.angle
        if self.rect.right > SCREEN_WIDTH:
            self.rect.right = SCREEN_WIDTH
            self.velocity[0] *= -EDGE_DAMPING
            self.angle = np.pi - self.angle
        if self.rect.top < 0:
            self.rect.top = 0
            self.velocity[1] *= -EDGE_DAMPING
            self.angle = np.pi + self.angle
        if self.rect.bottom > SCREEN_HEIGHT:
            self.rect.bottom = SCREEN_HEIGHT
            self.velocity[1] *= -EDGE_DAMPING
            self.angle = np.pi + self.angle


class Asteroid(g.sprite.Sprite):
    def __init__(self, size, centre_position, speed, angle):
        super(Asteroid, self).__init__()
        self.size = size
        self.position = np.array(centre_position)
        # Work out the constant velocity
        self.velocity = speed * unit_vector(angle)

        # Angle of rotation of the image
        self.sprite_angle = random.random()*2 * np.pi

    @property
    def rect(self):
        return g.Rect(self.position[0] - self.size / 2, self.position[1] - self.size / 2, self.size, self.size)

    def update(self):
        # Move the asteroid according to its speed
        self.position += self.velocity

        # Asteroid destroyed if off screen
        if off_screen(self.rect, self.size):
            self.kill()


class Laser(g.sprite.Sprite):
    def __init__(self, centre_position, angle):
        super(Laser, self).__init__()

        self.position = np.array(centre_position)
        self.velocity = LASER_SPEED * unit_vector(angle)
        self.angle = angle

    @property
    def rect(self):
        return g.Rect(self.position[0] - LASER_SIZE / 2, self.position[1] - LASER_SIZE / 2, LASER_SIZE, LASER_SIZE)

    def update(self):
        # Move according to speed
        self.position += self.velocity

        # Laser destroyed if off screen
        if off_screen(self.rect, LASER_SIZE):
            self.kill()


def angle_player_object(player, object_position):
    # Calculates the angle in radians of the asteroid in the frame of reference of the player
    # (where the players angle is set to 0 radians and the range is between -pi and pi)
    displacement = object_position - player.position
    relative_angle = find_angle(displacement) - player.angle
    return range_pi_to_pi(relative_angle)


def nearest_edge(player):
    # Find the nearest point on the edge of the screen
    edge_point = np.zeros(2)
    if min(SCREEN_WIDTH - player.position[0], player.position[0]) > min(SCREEN_HEIGHT - player.position[1],
                                                                        player.position[1]):
        edge_point[0] = player.position[0]
        if SCREEN_HEIGHT - player.position[1] < player.position[1]:
            edge_point[1] = SCREEN_HEIGHT
    else:
        edge_point[1] = player.position[1]
        if SCREEN_WIDTH - player.position[0] < player.position[0]:
            edge_point[0] = SCREEN_WIDTH
    return edge_point


def soonest_to_hit(player, asteroids, NUMBER_OF_SECTORS):
    # Finds the soonest to hit asteroid (or just an edge) in each sector and returns:
    # 1. 1/(time until impact considering only relative radial speed) of the asteroid to the player
    # 2. the angle between the asteroids' relative velocity and the line joining the asteroid and the player (in the range -pi to pi)
    # 3. the angle of the asteroid in the frame of reference of the player (in the range -pi to pi)

    # Finds the asteroid with the highest 1/(time until impact) in each sector
    soonest_to_hit_asteroids = np.array([(0, np.pi, 0) for _ in range(NUMBER_OF_SECTORS)])
    for asteroid in asteroids:
        # Find the time until impact given relative radial speed of the asteroid and the player
        relative_velocity = asteroid.velocity - player.velocity
        displacement = asteroid.position - player.position
        # Positive radial speed means the asteroid is travelling away from the player
        radial_speed = np.dot(unit_vector(find_angle(displacement)), relative_velocity)
        # Find 1/(time until impact)
        inverse_time_to_impact = -radial_speed / norm(displacement)

        # Find the angle of the asteroid in the frame of reference of the player
        relative_angle = angle_player_object(player, asteroid.position)

        # Determine which sector the asteroid is in
        sector_number = 0
        # Should sector edges be inside the function or outside??
        SECTOR_EDGES = np.linspace(-np.pi, np.pi, NUMBER_OF_SECTORS + 1)
        while relative_angle > SECTOR_EDGES[sector_number + 1]:
            sector_number += 1
        # Check which 1/(time until impact) is highest in this sector
        if soonest_to_hit_asteroids[sector_number][0] < inverse_time_to_impact:
            # Find the angle between the asteroids' relative velocity and the line joining the asteroid and the player
            angle_relative_velocity = range_pi_to_pi(find_angle(relative_velocity) - find_angle(-displacement))
            soonest_to_hit_asteroids[sector_number] = (inverse_time_to_impact, angle_relative_velocity, relative_angle)
    return soonest_to_hit_asteroids


def input_translator(player, asteroids):
    global NUMBER_OF_SECTORS
    inputs = np.zeros(NUMBER_OF_SECTORS * 3 + 2)
    inputs[:-2] = soonest_to_hit(player, asteroids, NUMBER_OF_SECTORS).flatten()

    # Distance to nearest edge and the angle of the nearest point on the edge in the frame of reference of the player
    edge_point = nearest_edge(player)
    inputs[-2] = norm(edge_point - player.position)
    inputs[-1] = angle_player_object(player, edge_point)
    return np.array([inputs])


def fitness_function(NN, NN_index):
    time_at_start = time.time()
    # This function lets the neural network (NN) play the asteroids game and calculates the score it achieves
    # The function returns 1/score
    global keras_ga, model, NUMBER_OF_SECTORS

    print('started one simulation')

    score = 0
    framenumber = 0
    reload_timer = 0
    asteroid_timer = 0

    # Create a spaceship to be controlled by the player
    player = Player()

    # Create a group of asteroid sprites and laser sprites
    asteroids = g.sprite.Group()
    lasers = g.sprite.Group()

    # Set the initial distribution of asteroid speeds
    var_asteroid_speed = VAR_ASTEROID_SPEED_INITIAL
    avg_asteroid_speed = AVG_ASTEROID_SPEED_INITIAL
    min_asteroid_speed = MIN_ASTEROID_SPEED_INITIAL

    alive = True
    while alive:
        # Translate the game to the inputs to the NN
        inputs = input_translator(player, asteroids)

        # Find the keys pressed by the NN given the input
        # output = pygad.nn.predict(last_layer=GANN_instance.population_networks[solution_index],
        #                          data_inputs=inputs,
        #                          problem_type="regression")

        output = pygad.kerasga.predict(model=model, solution=NN, data=inputs)

        # Round the prediction to the nearest integer (0 or 1)
        keys_pressed = np.rint(output[0])

        # Frame number
        framenumber += 1
        if framenumber % SCORE_RATE == 0:
            score += 1

        # Change the speed distribution of the asteroids
        var_asteroid_speed += VAR_ASTEROID_GRADIENT / FPS
        avg_asteroid_speed += AVG_ASTEROID_GRADIENT / FPS
        min_asteroid_speed += MIN_ASTEROID_GRADIENT / FPS

        # Add an asteroid after FRAMES_BETWEEN_ASTEROIDS number of frames
        if asteroid_timer > 0: asteroid_timer -= 1

        if asteroid_timer == 0:
            asteroid_timer = FRAMES_BETWEEN_ASTEROIDS
            size = asteroid_size_choice(framenumber)
            # Initial centre randomly generated on the edge
            random_centre, angle = random_edge_and_angle(size / 2)
            # Initial speed and angle randomly generated
            speed = abs(abs(random.normalvariate(avg_asteroid_speed,
                                                 var_asteroid_speed)) - min_asteroid_speed) + min_asteroid_speed
            new_asteroid = Asteroid(size, random_centre, speed, angle)
            asteroids.add(new_asteroid)

        # Update the player's ship's position depending on the keys pressed
        player.update(keys_pressed)
        asteroids.update()
        lasers.update()

        if reload_timer > 0: reload_timer -= 1

        if keys_pressed[0] and reload_timer == 0:
            new_laser = Laser(player.position, player.angle)
            lasers.add(new_laser)
            # Reset the reload time
            reload_timer = FRAMES_BETWEEN_RELOADS

        # Check for collisions between player and asteroid
        if g.sprite.spritecollide(player, asteroids, True):
            alive = False
            break

        # Deal with collisions between lasers and asteroids
        collide_dict = g.sprite.groupcollide(asteroids, lasers, True, True)
        for asteroid in collide_dict:
            if asteroid.size == ASTEROID_SIZES[0]:
                score += ASTEROID_SCORES[0]
                speed, angle1, angle2 = asteroid_split(0, asteroid.velocity, collide_dict[asteroid][0].angle)
                new_asteroid1 = Asteroid(ASTEROID_SIZES[1], asteroid.position + ASTEROID_SIZES[1] * (
                        unit_vector(angle1) - unit_vector((angle1 + angle2) / 2)), speed, angle1)
                new_asteroid2 = Asteroid(ASTEROID_SIZES[1], asteroid.position + ASTEROID_SIZES[1] * (
                        unit_vector(angle2) - unit_vector((angle1 + angle2) / 2)), speed, angle2)
                asteroids.add(new_asteroid1)
                asteroids.add(new_asteroid2)
            elif asteroid.size == ASTEROID_SIZES[1]:
                score += ASTEROID_SCORES[1]
                speed, angle1, angle2 = asteroid_split(1, asteroid.velocity, collide_dict[asteroid][0].angle)
                new_asteroid1 = Asteroid(ASTEROID_SIZES[2], asteroid.position + ASTEROID_SIZES[2] * (
                        unit_vector(angle1) - unit_vector((angle1 + angle2) / 2)), speed, angle1)
                new_asteroid2 = Asteroid(ASTEROID_SIZES[2], asteroid.position + ASTEROID_SIZES[2] * (
                        unit_vector(angle2) - unit_vector((angle1 + angle2) / 2)), speed, angle2)
                asteroids.add(new_asteroid1)
                asteroids.add(new_asteroid2)
            else:
                score += ASTEROID_SCORES[2]

    # Use max(score,1) to avoid dividing by 0
    print(f'actual time taken = {time_at_start - time.time()}')
    print(f'estimated running time = {framenumber*0.04}')
    print(f'score = {score}')
    return score


# fitness_func = fitness_function

# Variables relating to the NN
NUMBER_OF_SECTORS = 1
num_neurons_input = NUMBER_OF_SECTORS * 3 + 2
num_neurons_hidden_layers = 5
num_solutions = 4


def callback_generation(ga_instance):
    # global GANN_instance

    # population_matrices = pygad.gann.population_as_matrices(population_networks=GANN_instance.population_networks, population_vectors=ga_instance.population)
    # GANN_instance.update_population_trained_weights(population_trained_weights=population_matrices)

    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("best score = {fitness}".format(fitness=ga_instance.best_solution()[1]))

callback_generation = callback_generation

# Create the structure of the neural network
input_layer = tensorflow.keras.layers.Input(num_neurons_input)
dense_layer1 = tensorflow.keras.layers.Dense(num_neurons_hidden_layers, activation="relu")
output_layer = tensorflow.keras.layers.Dense(5, activation="softmax")

model = tensorflow.keras.Sequential()
model.add(input_layer)
model.add(dense_layer1)
model.add(output_layer)

keras_ga = pygad.kerasga.KerasGA(model=model, num_solutions=num_solutions)

# Variables relating to the genetic algorithm
num_generations = 3
num_parents_mating = 2
mutation_percent_genes = 5
initial_population = keras_ga.population_weights

# GANN_instance = pygad.gann.GANN(num_solutions=num_solutions,
# num_neurons_input=num_neurons_input,
# num_neurons_hidden_layers=num_neurons_hidden_layers,
# num_neurons_output=5,
# hidden_activations=["relu"],
# output_activation="softmax")

# population_vectors = pygad.gann.population_as_vectors(population_networks=GANN_instance.population_networks)

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       initial_population=initial_population,
                       fitness_func=fitness_function,
                       mutation_percent_genes=mutation_percent_genes,
                       on_generation=callback_generation)

ga_instance.run()

#ga_instance.plot_fitness(title="Iteration vs Fitness")

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"best solution: {solution}")
print(f"best solution fitness = {solution_fitness}")
print(f"best solution index = {solution_idx}")
