# Asteroids game
import numpy as np
from numpy.linalg import norm
import pygame
import random
import math
from constants import *


# Load images
<<<<<<< HEAD
asteroid_image = pygame.image.load('Images/asteroid.png')
explosion_image = pygame.image.load('Images/explosion.png')
spaceship_image = pygame.image.load('Images/spaceship.png')
=======
asteroid_image = pygame.image.load('asteroid.png')
explosion_image = pygame.image.load('explosion.png')
spaceship_image = pygame.image.load('spaceship.png')

# Choose whether or not the asteroids split
do_asteroids_split = False
>>>>>>> cd3ffa7b731c5a8f51e16289c3220cae4671ad1d


def rotate(image, angle):
    # Rotate image to the correct angle
    image = pygame.transform.rotate(image, angle - 90)
    return image


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


def show_text(surface, message, position, font, colour):
    # Generates a message on the surface given the position, font and colour
    text = font.render(message, True, colour)
    surface.blit(text, position)


def draw_score(surface, score, SCORE_FONT):
    # Score drawn in top left corner
    show_text(surface, f"Score = {score}", (10, 10), SCORE_FONT, TEXT_COLOUR_1)


class Player(pygame.sprite.Sprite):
    def __init__(self):
        super(Player, self).__init__()

        self.width, self.height = SHIPSIZE
        self.image = pygame.transform.scale(spaceship_image, SHIPSIZE)
        # Initial position in the centre of the screen
        self.position = np.array([SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2])
        # Initial speed of zero
        self.velocity = np.array([0, 0], dtype=float)

        # Initial angle pointing upwards
        self.angle = np.pi / 2
        # Initial angular frequency of zero
        self.frequency = 0

    @property
<<<<<<< HEAD
=======
    #### DESPERATELY NEEDS REWRITING!!!!!!!!!!!! (does not rotate with the player!)
>>>>>>> cd3ffa7b731c5a8f51e16289c3220cae4671ad1d
    def rect(self):
        return pygame.Rect(self.position[0] - self.width / 2, self.position[1] - self.height / 2, self.width, self.height)

    # Update the players position based on the keys pressed
    def update(self, pressed_keys):
        # speed decays by damping factor
        self.velocity = self.velocity * SPEED_DAMPING
<<<<<<< HEAD
        # Records whether the player hits the edge or not
        did_hit = 0
=======
>>>>>>> cd3ffa7b731c5a8f51e16289c3220cae4671ad1d

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
        if self.rect.left <= 0:
            self.rect.left = 0
            self.velocity[0] = max(abs(EDGE_DAMPING*self.velocity[0]), ACCELERATION_FORWARDS)
            self.angle = np.pi - self.angle
<<<<<<< HEAD
            did_hit = 1
=======
>>>>>>> cd3ffa7b731c5a8f51e16289c3220cae4671ad1d
        if self.rect.right >= SCREEN_WIDTH:
            self.rect.right = SCREEN_WIDTH
            self.velocity[0] = -max(abs(EDGE_DAMPING*self.velocity[0]), ACCELERATION_FORWARDS)
            self.angle = np.pi - self.angle
<<<<<<< HEAD
            did_hit = 1
=======
>>>>>>> cd3ffa7b731c5a8f51e16289c3220cae4671ad1d
        if self.rect.top <= 0:
            self.rect.top = 0
            self.velocity[1] = max(abs(EDGE_DAMPING*self.velocity[1]), ACCELERATION_FORWARDS)
            self.angle = np.pi + self.angle
<<<<<<< HEAD
            did_hit = 1
=======
>>>>>>> cd3ffa7b731c5a8f51e16289c3220cae4671ad1d
        if self.rect.bottom >= SCREEN_HEIGHT:
            self.rect.bottom = SCREEN_HEIGHT
            self.velocity[1] = -max(abs(EDGE_DAMPING*self.velocity[1]), ACCELERATION_FORWARDS)
            self.angle = np.pi + self.angle
<<<<<<< HEAD
            did_hit = 1
        return did_hit
=======
>>>>>>> cd3ffa7b731c5a8f51e16289c3220cae4671ad1d

    def draw(self, surface):
        # Draw the spaceship to the given surface
        image = pygame.transform.rotate(self.image, np.degrees(self.angle - np.pi / 2))
        rect = image.get_rect()
        rect.center = self.rect.center
        surface.blit(image, rect)

<<<<<<< HEAD
    def angle_player_object(self, object_position):
        # Calculates the angle in radians of the asteroid in the frame of reference of the player
        # (where the players angle is set to 0 radians and the range is between -pi and pi)
        displacement = object_position - self.position
        relative_angle = find_angle(displacement) - self.angle
        return range_pi_to_pi(relative_angle)

    def inverse_time_to_edge(self):
        # Find the nearest point on the edge of the screen to the player
        # Find the time until the player will hit this nearest edge and return 1/time
=======
    def nearest_edge(self):
        # Find the nearest point on the edge of the screen to the player
>>>>>>> cd3ffa7b731c5a8f51e16289c3220cae4671ad1d
        edge_point = np.zeros(2)
        if min(SCREEN_WIDTH - self.position[0], self.position[0]) > min(SCREEN_HEIGHT - self.position[1],
                                                                        self.position[1]):
            edge_point[0] = self.position[0]
            if SCREEN_HEIGHT - self.position[1] < self.position[1]:
                edge_point[1] = SCREEN_HEIGHT
<<<<<<< HEAD
            one_over_time = self.velocity[1] / (edge_point[1] - self.position[1])
=======
>>>>>>> cd3ffa7b731c5a8f51e16289c3220cae4671ad1d
        else:
            edge_point[1] = self.position[1]
            if SCREEN_WIDTH - self.position[0] < self.position[0]:
                edge_point[0] = SCREEN_WIDTH
<<<<<<< HEAD
            one_over_time = self.velocity[0] / (edge_point[0] - self.position[0])

        # Also find and return the angle of this point on the edge in the reference frame of the spaceship
        angle = self.angle_player_object(edge_point)

        return one_over_time, angle
=======
        return edge_point

    def angle_player_object(self, object_position):
        # Calculates the angle in radians of the asteroid in the frame of reference of the player
        # (where the players angle is set to 0 radians and the range is between -pi and pi)
        displacement = object_position - self.position
        relative_angle = find_angle(displacement) - self.angle
        return range_pi_to_pi(relative_angle)
>>>>>>> cd3ffa7b731c5a8f51e16289c3220cae4671ad1d


class Asteroid(pygame.sprite.Sprite):
    def __init__(self, size, centre_position, speed, angle):
        super(Asteroid, self).__init__()
        self.size = size
        self.position = np.array(centre_position)
        # Work out the constant velocity
        self.velocity = speed * unit_vector(angle)

        # Angle of rotation of the image
        self.sprite_angle = random.random() * 2 * np.pi

        # Create asteroid image of the right size
        self.image = rotate(pygame.transform.scale(asteroid_image, (size, size)), self.sprite_angle)

    @property
    def rect(self):
        return pygame.Rect(self.position[0] - self.size / 2, self.position[1] - self.size / 2, self.size, self.size)

    def update(self):
        # Move the asteroid according to its speed
        self.position += self.velocity

        # Asteroid destroyed if off screen
        if off_screen(self.rect, self.size):
            self.kill()

    def draw(self, surface):
        # Draw the spaceship to the given surface
        surface.blit(self.image, self.rect)


class Laser(pygame.sprite.Sprite):
    def __init__(self, centre_position, angle):
        super(Laser, self).__init__()

        self.position = np.array(centre_position)
        self.velocity = LASER_SPEED * unit_vector(angle)
        self.angle = angle

        # Draw the laser
        self.image = pygame.Surface((LASER_SIZE, LASER_SIZE))
        self.image.fill((10, 0, 30))
        pygame.draw.circle(self.image, (255, 0, 0), (LASER_SIZE / 2, LASER_SIZE / 2), LASER_SIZE)

    @property
    def rect(self):
        return pygame.Rect(self.position[0] - LASER_SIZE / 2, self.position[1] - LASER_SIZE / 2, LASER_SIZE, LASER_SIZE)

    def update(self):
        # Move according to speed
        self.position += self.velocity

        # Laser destroyed if off screen
        if off_screen(self.rect, LASER_SIZE):
            self.kill()

    def draw(self, surface):
        # Draw the spaceship to the given surface
        surface.blit(self.image, self.rect)


class Explosion(pygame.sprite.Sprite):
    def __init__(self, centre_position, object_size):
        super(Explosion, self).__init__()

        # Draw the explosion
        self.size = EXPLOSION_DILATION * object_size
        self.image = rotate(pygame.transform.scale(explosion_image, (self.size, self.size)), random.random() * np.pi)
        self.position = np.array(centre_position)
        self.age = 0

    @property
    def rect(self):
        return pygame.Rect(self.position[0] - self.size / 2, self.position[1] - self.size / 2, self.size, self.size)

    def update(self):
        self.age += 1
        if self.age >= EXPLOSION_DURATION:
            self.kill()

    def draw(self, surface):
        # Draw the spaceship to the given surface
        surface.blit(self.image, self.rect)


class Game():
    def __init__(self):
        self.score = 0
        self.framenumber = 0
        self.reload_timer = 0
        self.asteroid_timer = 0

        # Create a spaceship to be controlled by the player
        self.player = Player()

        # Create a group of asteroid sprites and laser sprites
        self.asteroids = pygame.sprite.Group()
        self.lasers = pygame.sprite.Group()
        self.explosions = pygame.sprite.Group()

        # Set the initial distribution of asteroid speeds
        self.var_asteroid_speed = VAR_ASTEROID_SPEED_INITIAL
        self.avg_asteroid_speed = AVG_ASTEROID_SPEED_INITIAL
        self.min_asteroid_speed = MIN_ASTEROID_SPEED_INITIAL

        self.alive = True

    def update(self, keys_pressed):
<<<<<<< HEAD
        # Update frame number
=======
        # Frame number
>>>>>>> cd3ffa7b731c5a8f51e16289c3220cae4671ad1d
        self.framenumber += 1
        if self.framenumber % SCORE_RATE == 0:
            self.score += 1

        # Change the speed distribution of the asteroids
        self.var_asteroid_speed += VAR_ASTEROID_GRADIENT / FPS
        self.avg_asteroid_speed += AVG_ASTEROID_GRADIENT / FPS
        self.min_asteroid_speed += MIN_ASTEROID_GRADIENT / FPS

<<<<<<< HEAD
        # Add an asteroid is created after FRAMES_BETWEEN_ASTEROIDS number of frames
=======
        # Add an asteroid after FRAMES_BETWEEN_ASTEROIDS number of frames
>>>>>>> cd3ffa7b731c5a8f51e16289c3220cae4671ad1d
        if self.asteroid_timer > 0: self.asteroid_timer -= 1

        if self.asteroid_timer == 0:
            self.asteroid_timer = FRAMES_BETWEEN_ASTEROIDS
            size = asteroid_size_choice(self.framenumber)
            # Initial centre randomly generated on the edge
            random_centre, angle = random_edge_and_angle(size / 2)
            # Initial speed and angle randomly generated
            speed = abs(abs(random.normalvariate(self.avg_asteroid_speed,
                                                 self.var_asteroid_speed)) - self.min_asteroid_speed) + self.min_asteroid_speed
            new_asteroid = Asteroid(size, random_centre, speed, angle)
            self.asteroids.add(new_asteroid)

<<<<<<< HEAD
        # Update the sprites depending on the keys pressed
        if self.player.update(keys_pressed) == 1:
            self.score -= points_lost
=======
        # Update the player's ship's position depending on the keys pressed
        self.player.update(keys_pressed)
>>>>>>> cd3ffa7b731c5a8f51e16289c3220cae4671ad1d
        self.asteroids.update()
        self.lasers.update()
        self.explosions.update()

        if self.reload_timer > 0: self.reload_timer -= 1

        if keys_pressed[0] and self.reload_timer == 0:
            new_laser = Laser(self.player.position, self.player.angle)
            self.lasers.add(new_laser)
            # Reset the reload time
            self.reload_timer = FRAMES_BETWEEN_RELOADS

        # Check for collisions between player and asteroid
        if pygame.sprite.spritecollide(self.player, self.asteroids, True):
            new_explosion = Explosion(self.player.position, self.player.height)
            self.explosions.add(new_explosion)
            self.alive = False

        # Deal with collisions between lasers and asteroids
        collide_dict = pygame.sprite.groupcollide(self.asteroids, self.lasers, True, True)
        for asteroid in collide_dict:
<<<<<<< HEAD
            if asteroid.size == ASTEROID_SIZES[0]:
                self.score += ASTEROID_SCORES[0]
                if do_asteroids_split == True:
=======
            if do_asteroids_split == True:
                if asteroid.size == ASTEROID_SIZES[0]:
                    self.score += ASTEROID_SCORES[0]
>>>>>>> cd3ffa7b731c5a8f51e16289c3220cae4671ad1d
                    speed, angle1, angle2 = asteroid_split(0, asteroid.velocity, collide_dict[asteroid][0].angle)
                    new_asteroid1 = Asteroid(ASTEROID_SIZES[1], asteroid.position + ASTEROID_SIZES[1] * (
                            unit_vector(angle1) - unit_vector((angle1 + angle2) / 2)), speed, angle1)
                    new_asteroid2 = Asteroid(ASTEROID_SIZES[1], asteroid.position + ASTEROID_SIZES[1] * (
                            unit_vector(angle2) - unit_vector((angle1 + angle2) / 2)), speed, angle2)
                    self.asteroids.add(new_asteroid1)
                    self.asteroids.add(new_asteroid2)
<<<<<<< HEAD
            elif asteroid.size == ASTEROID_SIZES[1]:
                self.score += ASTEROID_SCORES[1]
                if do_asteroids_split == True:
=======
                elif asteroid.size == ASTEROID_SIZES[1]:
                    self.score += ASTEROID_SCORES[1]
>>>>>>> cd3ffa7b731c5a8f51e16289c3220cae4671ad1d
                    speed, angle1, angle2 = asteroid_split(1, asteroid.velocity, collide_dict[asteroid][0].angle)
                    new_asteroid1 = Asteroid(ASTEROID_SIZES[2], asteroid.position + ASTEROID_SIZES[2] * (
                            unit_vector(angle1) - unit_vector((angle1 + angle2) / 2)), speed, angle1)
                    new_asteroid2 = Asteroid(ASTEROID_SIZES[2], asteroid.position + ASTEROID_SIZES[2] * (
                            unit_vector(angle2) - unit_vector((angle1 + angle2) / 2)), speed, angle2)
                    self.asteroids.add(new_asteroid1)
                    self.asteroids.add(new_asteroid2)
<<<<<<< HEAD
            else:
                self.score += ASTEROID_SCORES[2]
=======
                else:
                    self.score += ASTEROID_SCORES[2]
>>>>>>> cd3ffa7b731c5a8f51e16289c3220cae4671ad1d

            # Explosion!
            new_explosion = Explosion(asteroid.position, asteroid.size)
            self.explosions.add(new_explosion)

    def draw_game(self, screen, SCORE_FONT, inputs=None):
        # Draws the current state of a game (class Game) onto the screen
        # Black background
        screen.fill((10, 0, 30))
        # Draw the sprites
        self.player.draw(screen)
        for asteroid in self.asteroids:
            asteroid.draw(screen)
        for laser in self.lasers:
            laser.draw(screen)
        for explosion in self.explosions:
            explosion.draw(screen)
        # Draw score
        draw_score(screen, self.score, SCORE_FONT)

        # Draw lines
        if inputs is not None:
            NUMBER_OF_SECTORS = int(len(inputs)/3)
            SECTOR_EDGES = np.linspace(-np.pi, np.pi, NUMBER_OF_SECTORS + 1)
            for n in range(NUMBER_OF_SECTORS):
                pygame.draw.line(screen, (255, 255, 255), self.player.position, unit_vector(self.player.angle + SECTOR_EDGES[n]) * 1200 + self.player.position, width=1)
                pygame.draw.line(screen, (0, 255, 0), self.player.position, unit_vector(self.player.angle + inputs[3*n+2]) * 1200 + self.player.position)

        # Update content of the screen
        pygame.display.flip()
