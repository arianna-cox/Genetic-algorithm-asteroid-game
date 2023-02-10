from training_pytorch import threshold, NUMBER_OF_SECTORS, prediction_function
from fitness_function import fitness_function
import numpy as np
import pygame as g
import pygad.torchga

print(np.load("best_solution.npy"))
best_solution = np.load("best_solution.npy")
# Show the best solution playing the game
fitness_function(best_solution, 1, prediction_function, NUMBER_OF_SECTORS, threshold, do_draw=True)