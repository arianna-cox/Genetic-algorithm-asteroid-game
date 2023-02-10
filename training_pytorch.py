from fitness_function import fitness_function
import numpy as np
import torch.nn
import pygame as g
import pygad.torchga

# Decide whether or not to run the graphics
do_draw = False

# Variables relating to the NN
NUMBER_OF_SECTORS = 1
num_neurons_input = NUMBER_OF_SECTORS * 3 + 2
num_neurons_hidden_layer_1 = 5
num_solutions = 10
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

torch_ga = pygad.torchga.TorchGA(model=model, num_solutions=num_solutions)

# Variables relating to the genetic algorithm
num_generations = 2
num_parents_mating = 5
mutation_percent_genes = 10
initial_population = torch_ga.population_weights


def prediction_function(solution, data):
    model_weights_dictionary = pygad.torchga.model_weights_as_dict(model=model, weights_vector=solution)
    model.load_state_dict(model_weights_dictionary)
    output = model(torch.tensor(data[0]).float()).detach().numpy()
    return output


fitness_func = lambda NN, solution_index: fitness_function(NN, solution_index, prediction_function,
                                                           NUMBER_OF_SECTORS, threshold, do_draw=do_draw)


def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))


ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       mutation_percent_genes=mutation_percent_genes,
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       on_generation=callback_generation)

if __name__ == '__main__':
    ga_instance.run()

    ga_instance.plot_fitness(title="PyGAD & PyTorch - Iteration vs. Fitness")

    best_solution, best_solution_fitness, best_solution_idx = ga_instance.best_solution()
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=best_solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=best_solution_idx))

    best_solution_weights = pygad.torchga.model_weights_as_dict(model=model, weights_vector=best_solution)

    np.save("best_solution", best_solution)
    np.save("best_solution_weights", best_solution_weights)

    best_solution = np.load("best_solution.npy")

    fitness_function(best_solution, 0, prediction_function, NUMBER_OF_SECTORS, threshold, do_draw=True)
