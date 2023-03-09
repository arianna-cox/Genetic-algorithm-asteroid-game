from game import Game
from constants import *
from input_translator import input_translator
import pygame
import time
import numpy as np
import pygad
import pygad.kerasga
import pygame
import time
from random import randint


def fitness_function(NN, solution_index, prediction_function, NUMBER_OF_SECTORS, threshold, do_draw=False, include_edges=False):
    # Create an instance of the game
    game = Game()

    if do_draw:
        # Initialize pygame
        pygame.init()

        # Initialize the font
        SCORE_FONT = pygame.font.SysFont('phosphate', 60)

        # Create the screen
        screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
        game.draw_game(screen, SCORE_FONT)

    while game.alive:
        # Calculates the inputs for the neural network based on the current state of the game
        inputs = input_translator(game.player, game.asteroids, NUMBER_OF_SECTORS, include_edges=include_edges)

        # Applies the prediction function
        output = prediction_function(solution=NN, data=inputs)

        # Key is pressed if the prediction is greater than the threshold
        keys_pressed = output > threshold

        game.update(keys_pressed)

        if do_draw:
            game.draw_game(screen, SCORE_FONT, inputs=inputs[0])
            time.sleep(1 / 20)
    #print(f'score = {game.score}')
    print(solution_index)
    return max(0, game.score)
    return game.score
