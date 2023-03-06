# Asteroids game
# This was my first attempt at writing the game Asteroids that is human playable.
# The structure and content of this code has been improved significantly and can be found in game.py

import numpy as npgit
import pygame as g
from pygame.locals import (K_UP, K_DOWN, K_LEFT, K_RIGHT, K_ESCAPE, K_SPACE)
import random
import math
import time

# Initialize
g.init()

# Load images
asteroid_image = g.image.load('../Images/asteroid.png')
explosion_image = g.image.load('../Images/explosion.png')
spaceship_image = g.image.load('../Images/spaceship.png')

# Screen dimensions
SCREEN_HEIGHT = 800
SCREEN_WIDTH = 800

# Frames per second
FPS = 60
# Frame rate in milliseconds rounded up
FRAME_RATE = np.ceil(1000 / FPS)
# Frames after which score accumulates
SCORE_RATE = 60
time_between_asteroids = 2000
MIN_ASTEROID_SPEED_INITIAL = 0.1
AVG_ASTEROID_SPEED_INITIAL = 0.2
VAR_ASTEROID_SPEED_INITIAL = 0.2
MIN_ASTEROID_GRADIENT = 1 / 500
AVG_ASTEROID_GRADIENT = 1 / 300
VAR_ASTEROID_GRADIENT = 1 / 200
asteroid_angle_var = 20
ASTEROID_SIZES = (96, 64, 32)
ASTEROID_MASSES = (150, 60, 10)
ASTEROID_SCORES = (1, 3, 10)
# Upper bound on the probability of spawning an asteroid of maximum size
MAX_SPAWN_PROB = 0.2
# Frame at which the maximum probability of a large asteroid is reached
FRAME_MAX_PROB_IS_REACHED = 90 * FPS
# Gradient of the linear mapping between frame number and probability of spawning a large asteroid
SPAWN_GRADIENT = MAX_SPAWN_PROB / FRAME_MAX_PROB_IS_REACHED

LASER_SIZE = 6
LASER_SPEED = 6
LASER_MASS = 1

# Factor by which the explosion is bigger than the original exploding object
EXPLOSION_DILATION = 1.5
EXPLOSION_DURATION = 5

RELOAD_TIME = 50
reload_timer = 0

SHIPSIZE = (30, 60)
# Parameters controlling how the ship moves
ACCELERATION_FORWARDS = 0.05
ACCELERATION_BACKWARDS = 0.03
SPEED_DAMPING = 0.99
ANGULAR_DAMPING = 0.95
ANGLE_CHANGE = 0.1
EDGE_DAMPING = 0.995

# Text
TEXT_COLOUR_1 = (0, 100, 255)
TEXT_COLOUR_2 = (255, 100, 100)
SCORE_FONT = g.font.SysFont('phosphate', 60)
END_FONT = g.font.SysFont('phosphate', 100)
SMALL_TEXT_FONT = g.font.SysFont('phosphate', 40)


def rotate(image, angle):
    # Rotate image to the correct angle
    image = g.transform.rotate(image, angle - 90)
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
    angle = math.degrees(
        math.atan2(y - SCREEN_HEIGHT / 2, SCREEN_WIDTH / 2 - x)) \
            + random.normalvariate(0, asteroid_angle_var)
    return (x, y), angle


def direction(angle):
    # Takes an angle in degrees and returns a unit vector in that direction
    return np.array((np.cos(math.radians(angle)), -np.sin(math.radians(angle))))


def show_text(surface, message, position, font, colour):
    # Generates a message on the surface given the position, font and colour
    text = font.render(message, True, colour)
    surface.blit(text, position)


def draw_score(surface, score):
    # Score drawn in top left corner
    show_text(surface, f"Score = {score}", (10, 10), SCORE_FONT, TEXT_COLOUR_1)


def asteroid_size_choice(framenumber):
    threshold = min(MAX_SPAWN_PROB, framenumber * SPAWN_GRADIENT)
    return ASTEROID_SIZES[int(random.random() > threshold)]


def asteroid_split(asteroid_level, asteroid_velocity, laser_angle):
    # Calculates the trajectory of two smaller asteroids after a collision of a laser and larger asteroid
    # Momentum is conserved
    # Added explosive kinetic energy is generated randomly
    total_mom = (
                asteroid_velocity * ASTEROID_MASSES[asteroid_level] + LASER_SPEED * direction(laser_angle) * LASER_MASS)
    total_angle = np.degrees(np.arctan2(-total_mom[1], total_mom[0]))
    parallel_velocity = total_mom / (2 * ASTEROID_MASSES[asteroid_level + 1])
    # perpendicular_deviation = 0*random.uniform(avg_asteroid_speed*0.9, avg_asteroid_speed*1.3)*direction(np.degrees(total_angle - np.pi/2))
    # velocity1 = parallel_velocity - perpendicular_deviation
    # velocity2 = parallel_velocity + perpendicular_deviation
    angle_deviation = random.normalvariate(20, 5)
    return np.linalg.norm(parallel_velocity), total_angle + angle_deviation, total_angle - angle_deviation


# Define a Player object by extending pygame.sprite.Sprite
class Player(g.sprite.Sprite):
    def __init__(self):
        super(Player, self).__init__()

        # Create the image of the ship
        self.width, self.height = SHIPSIZE
        self.image = g.transform.scale(spaceship_image, SHIPSIZE)

        # Initial position in the centre of the screen
        self.position = np.array([SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2])
        # Initial speed of zero
        self.velocity = np.array([0, 0], dtype=float)

        # Initial angle pointing upwards
        self.angle = 90
        # Initial angular frequency of zero
        self.frequency = 0

    @property
    def rect(self):
        return g.Rect(self.position[0] - self.width / 2, self.position[1] - self.height / 2, self.width, self.height)

    # Update the players position based on the keys pressed
    def update(self, pressed_keys):
        # speed decays by damping factor
        self.velocity = self.velocity * SPEED_DAMPING

        # Move forwards or backwards
        if pressed_keys[K_UP]:
            self.velocity += direction(self.angle) * ACCELERATION_FORWARDS
        if pressed_keys[K_DOWN]:
            self.velocity += -direction(self.angle) * ACCELERATION_BACKWARDS
        self.position += self.velocity

        # speed decays by damping factor
        self.frequency *= ANGULAR_DAMPING

        # Rotate using left and right keys
        if pressed_keys[K_LEFT]:
            self.frequency += ANGLE_CHANGE
        if pressed_keys[K_RIGHT]:
            self.frequency -= ANGLE_CHANGE
        self.angle = (self.angle + self.frequency) % 360

        # Player not allowed off screen and loses all speed in corresponding direction
        if self.rect.left < 0:
            self.rect.left = 0
            self.velocity[0] *= -EDGE_DAMPING
            self.angle = 180 - self.angle
        if self.rect.right > SCREEN_WIDTH:
            self.rect.right = SCREEN_WIDTH
            self.velocity[0] *= -EDGE_DAMPING
            self.angle = 180 - self.angle
        if self.rect.top < 0:
            self.rect.top = 0
            self.velocity[1] *= -EDGE_DAMPING
            self.angle = 180 + self.angle
        if self.rect.bottom > SCREEN_HEIGHT:
            self.rect.bottom = SCREEN_HEIGHT
            self.velocity[1] *= -EDGE_DAMPING
            self.angle = 180 + self.angle

    def draw(self, surface):
        # Draw the spaceship to the given surface
        image = g.transform.rotate(self.image, self.angle - 90)
        rect = image.get_rect()
        rect.center = self.rect.center
        surface.blit(image, rect)


class Asteroid(g.sprite.Sprite):
    def __init__(self, size, centre_position, speed, angle):
        super(Asteroid, self).__init__()
        self.size = size
        self.position = np.array(centre_position)
        # Work out the constant velocity
        self.velocity = speed * direction(angle)

        # Angle of rotation of the image
        self.sprite_angle = random.randint(0, 359)

        # Create asteroid image of the right size
        self.image = g.transform.scale(asteroid_image, (size, size))

    @property
    def rect(self):
        return g.Rect(self.position[0] - self.size / 2, self.position[1] - self.size / 2, self.size, self.size)

    def update(self):
        # Move the asteroid according to its speed
        self.position += self.velocity

        # Asteroid destroyed if off screen
        if off_screen(self.rect, self.size):
            self.kill()

    def draw(self, surface):
        # Draw the spaceship to the given surface
        surface.blit(rotate(self.image, self.sprite_angle), self.rect)


class Laser(g.sprite.Sprite):
    def __init__(self, centre_position, angle):
        super(Laser, self).__init__()

        # Draw the laser
        self.image = g.Surface((LASER_SIZE, LASER_SIZE))
        self.image.fill((10, 0, 30))
        g.draw.circle(self.image, (255, 0, 0), (LASER_SIZE / 2, LASER_SIZE / 2), LASER_SIZE)
        self.position = np.array(centre_position)
        self.velocity = LASER_SPEED * direction(angle)
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

    def draw(self, surface):
        # Draw the spaceship to the given surface
        surface.blit(self.image, self.rect)


class Explosion(g.sprite.Sprite):
    def __init__(self, centre_position, object_size):
        super(Explosion, self).__init__()

        # Draw the explosion
        self.size = EXPLOSION_DILATION * object_size
        self.image = rotate(g.transform.scale(explosion_image, (self.size, self.size)), random.randint(0, 359))
        self.position = np.array(centre_position)
        self.age = 0

    @property
    def rect(self):
        return g.Rect(self.position[0] - self.size / 2, self.position[1] - self.size / 2, self.size, self.size)

    def update(self):
        self.age += 1
        if self.age >= EXPLOSION_DURATION:
            self.kill()

    def draw(self, surface):
        # Draw the spaceship to the given surface
        surface.blit(self.image, self.rect)


# Create the screen
screen = g.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])

# Set the window title
g.display.set_caption("Arianna's Asteroids")

while True:
    # Start the clock
    clock = g.time.Clock()

    # Set score and framenumber to zero
    score = 0
    framenumber = 0

    # Create a spaceship to be controlled by the player
    player = Player()

    # Create a group of asteroid sprites, laser sprites and explosion sprites
    asteroids = g.sprite.Group()
    lasers = g.sprite.Group()
    explosions = g.sprite.Group()

    # Create a events for new asteroids to be created
    AddASTEROID = g.USEREVENT + 1
    g.time.set_timer(AddASTEROID, time_between_asteroids)

    alive = True
    while alive:

        # Control the frame rate
        clock.tick(FPS)
        framenumber += 1
        if framenumber % SCORE_RATE == 0:
            score += 1

        # Warning if the framerate slows
        if clock.get_time() > 2 * FRAME_RATE:
            print(f"warning: {clock.get_time()} (> {FRAME_RATE}) milliseconds taken between frames")

        # Change the speed distribution of the asteroids
        var_asteroid_speed = VAR_ASTEROID_SPEED_INITIAL + (framenumber / FPS) * VAR_ASTEROID_GRADIENT
        avg_asteroid_speed = AVG_ASTEROID_SPEED_INITIAL + (framenumber / FPS) * AVG_ASTEROID_GRADIENT
        min_asteroid_speed = MIN_ASTEROID_SPEED_INITIAL + (framenumber / FPS) * MIN_ASTEROID_GRADIENT

        # Check inputs from the player
        for event in g.event.get():
            # Get the set of keys pressed
            pressed_keys = g.key.get_pressed()
            # Escape key stops the game

            if pressed_keys[K_ESCAPE]:
                g.quit()
                alive = False

            # Check whether the close window button has been pressed
            if event.type == g.QUIT:
                g.quit()
                alive = False

            if event.type == AddASTEROID:
                size = asteroid_size_choice(framenumber)
                # Initial centre randomly generated on the edge
                random_centre, angle = random_edge_and_angle(size / 2)
                # Initial speed and angle randomly generated
                speed = abs(abs(random.normalvariate(avg_asteroid_speed,
                                                     var_asteroid_speed)) - min_asteroid_speed) + min_asteroid_speed
                new_asteroid = Asteroid(size, random_centre, speed, angle)
                asteroids.add(new_asteroid)

        # Update the player's ship's position depending on the keys pressed
        player.update(pressed_keys)
        asteroids.update()
        lasers.update()
        explosions.update()

        if reload_timer > 0: reload_timer -= 1

        if pressed_keys[K_SPACE] and reload_timer == 0:
            new_laser = Laser(player.position, player.angle)
            lasers.add(new_laser)
            # Reset the reload time
            reload_timer = RELOAD_TIME

        # Check for collisions between player and asteroid
        if g.sprite.spritecollide(player, asteroids, True):
            # Explosion!
            new_explosion = Explosion(player.position, player.height)
            new_explosion.draw(screen)
            g.display.flip()
            time.sleep(1)

            alive = False
            break

        # Deal with collisions between lasers and asteroids
        collide_dict = g.sprite.groupcollide(asteroids, lasers, True, True)
        for asteroid in collide_dict:
            if asteroid.size == ASTEROID_SIZES[0]:
                score += ASTEROID_SCORES[0]
                speed, angle1, angle2 = asteroid_split(0, asteroid.velocity, collide_dict[asteroid][0].angle)
                new_asteroid1 = Asteroid(ASTEROID_SIZES[1], asteroid.position + ASTEROID_SIZES[1] * (
                            direction(angle1) - direction((angle1 + angle2) / 2)), speed, angle1)
                new_asteroid2 = Asteroid(ASTEROID_SIZES[1], asteroid.position + ASTEROID_SIZES[1] * (
                            direction(angle2) - direction((angle1 + angle2) / 2)), speed, angle2)
                asteroids.add(new_asteroid1)
                asteroids.add(new_asteroid2)
            elif asteroid.size == ASTEROID_SIZES[1]:
                score += ASTEROID_SCORES[1]
                speed, angle1, angle2 = asteroid_split(1, asteroid.velocity, collide_dict[asteroid][0].angle)
                new_asteroid1 = Asteroid(ASTEROID_SIZES[2], asteroid.position + ASTEROID_SIZES[2] * (
                            direction(angle1) - direction((angle1 + angle2) / 2)), speed, angle1)
                new_asteroid2 = Asteroid(ASTEROID_SIZES[2], asteroid.position + ASTEROID_SIZES[2] * (
                            direction(angle2) - direction((angle1 + angle2) / 2)), speed, angle2)
                asteroids.add(new_asteroid1)
                asteroids.add(new_asteroid2)
            else:
                score += ASTEROID_SCORES[2]
            # Explosion!
            new_explosion = Explosion(asteroid.position, asteroid.size)
            explosions.add(new_explosion)

        # Black background
        screen.fill((10, 0, 30))
        # Draw spaceship
        player.draw(screen)
        # Draw asteroids
        for asteroid in asteroids:
            asteroid.draw(screen)
        # Draw lasers
        for laser in lasers:
            laser.draw(screen)
        # Draw explosions
        for explosion in explosions:
            explosion.draw(screen)
        # Show score
        draw_score(screen, score)

        # Update content of the screen
        g.display.flip()

    start_again = False
    while start_again == False:
        # Black background
        screen.fill((10, 0, 30))
        # End game message
        show_text(screen, "GAME OVER", (140, SCREEN_HEIGHT / 2 - 200), END_FONT, TEXT_COLOUR_1)
        show_text(screen, f"SCORE = {score}", (SCREEN_WIDTH / 2 - 140, SCREEN_HEIGHT / 2 - 50), SCORE_FONT,
                  TEXT_COLOUR_1)
        show_text(screen, "Press the spacebar to play again", (SCREEN_WIDTH / 2 - 300, SCREEN_HEIGHT - 100),
                  SMALL_TEXT_FONT, TEXT_COLOUR_2)
        # Display changes
        g.display.flip()

        for event in g.event.get():
            # Get the set of keys pressed
            pressed_keys = g.key.get_pressed()

            # Escape key stops the game
            if pressed_keys[K_ESCAPE]:
                g.quit()
                break
            if event.type == g.QUIT:
                g.quit()
                break
            if pressed_keys[K_SPACE]:
                start_again = True

    time.sleep(0.5)
    continue
