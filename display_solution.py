import pygame

from fitness_function import fitness_function
import numpy as np
import torch.nn
import pygad.torchga

# Variables relating to the NN
NUMBER_OF_SECTORS = 1
include_edges = False
num_neurons_input = NUMBER_OF_SECTORS * 3 + 2 * include_edges
num_neurons_hidden_layer_1 = 4

# Threshold for the outputs of the NN above which a key is considered pressed
threshold = 0.1

# Create the structure of the neural network with five outputs (representing the four arrow keys and spacebar)
input_layer = torch.nn.Linear(num_neurons_input, num_neurons_hidden_layer_1)
relu_layer1 = torch.nn.ReLU()
output_layer = torch.nn.Linear(num_neurons_hidden_layer_1, 5)
softmax_layer = torch.nn.Sigmoid()

model = torch.nn.Sequential(input_layer,
                            relu_layer1,
                            output_layer,
                            softmax_layer)


def prediction_function(solution, data):
    model_weights_dictionary = pygad.torchga.model_weights_as_dict(model=model, weights_vector=solution)
    model.load_state_dict(model_weights_dictionary)
    output = model(torch.tensor(data[0]).float()).detach().numpy()
    return output

loaded_filename = 'ga_instance_s1_2'
ga_instance = pygad.load(filename=loaded_filename)
best_solution, best_solution_fitness, best_solution_idx = ga_instance.best_solution()

# Show the best solution playing the game
# best_solution = np.load("best_solution.npy")
for i in range(4):
    fitness_function(best_solution, 0, prediction_function, NUMBER_OF_SECTORS, threshold, do_draw=1)

