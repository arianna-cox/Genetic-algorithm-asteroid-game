from fitness_function import fitness_function
import numpy as np
import torch.nn
import pygame as g
import pygad.torchga
import time

# Variables relating to the NN
NUMBER_OF_SECTORS = 3
include_edges = False
num_neurons_input = NUMBER_OF_SECTORS * 3 + 2 * include_edges
num_neurons_hidden_layer_1 = 7

# Threshold for the outputs of the NN above which a key is considered pressed
threshold = 0.5

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


if __name__ == '__main__':
    # Decide whether or not to run the graphics
    do_draw_training = False

    # Variables relating to the genetic algorithm
    num_solutions = 100
    num_generations = 3
    num_parents_mating = num_solutions
    mutation_percent_genes = (15, 5)

    # If you would like to continue training with a previous instance of the GA give the filename here
    # Else, set filename to None and a new instance of the GA will be created
    loaded_filename = 'ga_instance_s3_1'

    # Choose the filename of the trained GA instance
    new_filename = 'ga_instance_s3_1'
    # Save a dictionary of the relevant variables corresponding to the instance
    dictionary = {'NUMBER_OF_SECTORS': NUMBER_OF_SECTORS, 'include_edges': include_edges,
                  'num_neurons_hidden_layer_1': num_neurons_hidden_layer_1, 'threshold': threshold,
                  'num_solutions': num_solutions, 'mutation_percent_genes': mutation_percent_genes,
                  'num_generations': num_generations}
    np.save(new_filename + '_dictionary', dictionary)
    # Choose how frequently (after how many generations) the ga_instance saves to the new_filename
    saves_after_generations = 1

    def fitness_func(NN, solution_index):
        return fitness_function(NN, solution_index, prediction_function,
                                NUMBER_OF_SECTORS, threshold, do_draw=do_draw_training)

    def callback_generation(Ga_instance):
        generation = Ga_instance.generations_completed
        print(f"Generation = {generation}")
        if generation % saves_after_generations == 0:
            Ga_instance.save(filename=new_filename)
            print("saved!")
        # Recalculates the fitness of each solution and returns the best value
        # print("Best fitness = {fitness}".format(fitness=ga_instance.best_solution()[1]))

    def on_start(ga_instance):
        print("on_start()")

    def on_fitness(ga_instance, population_fitness):
        print("on_fitness()")

    def on_parents(ga_instance, selected_parents):
        print("on_parents()")

    def on_crossover(ga_instance, offspring_crossover):
        print("on_crossover()")

    def on_mutation(ga_instance, offspring_mutation):
        print("on_mutation()")

    def on_stop(ga_instance, last_population_fitness):
        print("on_stop()")

    if loaded_filename is None:
        # Create initial population
        torch_ga = pygad.torchga.TorchGA(model=model, num_solutions=num_solutions)
        initial_population = torch_ga.population_weights

        # Create a new instance of the genetic algorithm
        ga_instance = pygad.GA(num_generations=num_generations,
                               num_parents_mating=num_parents_mating,
                               parent_selection_type="rws",
                               keep_parents=0,
                               keep_elitism=0,
                               mutation_percent_genes=mutation_percent_genes,
                               mutation_type="adaptive",
                               initial_population=initial_population,
                               fitness_func=fitness_func,
                               save_best_solutions=True,
                               save_solutions=True,
                               on_generation=callback_generation,
                               on_start=on_start,
                               on_fitness=on_fitness,
                               on_parents=on_parents,
                               on_crossover=on_crossover,
                               on_mutation=on_mutation,
                               on_stop=on_stop
                               )
    else:
        ga_instance = pygad.load(filename=loaded_filename)

    for i in range(3):
        ga_instance.run()

    ga_instance.save(filename=new_filename)
    print("saved!")

    ga_instance.plot_fitness(title="Best fitness vs generation")

    best_solution, best_solution_fitness, best_solution_idx = ga_instance.best_solution()
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=best_solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=best_solution_idx))

    #best_solution_weights = pygad.torchga.model_weights_as_dict(model=model, weights_vector=best_solution)

    #np.save("best_solution", best_solution)
    #np.save("best_solution_weights", best_solution_weights)

    # best_solution = np.load("best_solution.npy")
    for i in range(100):
        fitness_function(best_solution, 0, prediction_function, NUMBER_OF_SECTORS, threshold, do_draw=True)
