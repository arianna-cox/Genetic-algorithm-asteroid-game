import pygame
import math

pygame.init()
# Set the window size
window_size = (800, 600)

# Create the window
screen = pygame.display.set_mode(window_size)

# Set the window title
pygame.display.set_caption('Asteroids')

# Create a game clock to control the frame rate
clock = pygame.time.Clock()


class Spaceship(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.image.load('spaceship.jpeg')
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.speed = 5
        self.angle = 0

    def update(self):
        # Rotate the spaceship based on the arrow keys
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.angle -= 5
        if keys[pygame.K_RIGHT]:
            self.angle += 5
        self.image = pygame.transform.rotate(self.image, self.angle)

        # Move the spaceship based on the arrow keys
        self.rect.x += self.speed * math.sin(math.radians(self.angle))
        self.rect.y -= self.speed * math.cos(math.radians(self.angle))

    def draw(self, surface):
        # Draw the spaceship to the given surface
        surface.blit(self.image, self.rect)

spaceship = Spaceship(400, 400)

running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        # elif event.type == pygame.KEYDOWN:
        #     if event.key == pygame.K_SPACE:
        #         # Fire a bullet
        #         bullet = Bullet(spaceship.rect.x, spaceship.rect.y, spaceship.angle)
        #         bullets.add(bullet)

    # Update game objects
    spaceship.update()
    # asteroids.update()
    # bullets.update()

    # Check for collisions
    pygame.sprite.groupcollide(asteroids, bullets, True, True)

    # Draw the game objects
    screen.fill((0, 0, 0))
    spaceship.draw(screen)
    # asteroids.draw(screen)
    # bullets.draw(screen)
    pygame.display.flip()

    # Limit the frame rate to 60 FPS
    clock.tick(60)
