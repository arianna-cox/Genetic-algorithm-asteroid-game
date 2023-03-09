from pytorch_create_and_load_ga_instances import Genetic_algorithm_instance, display_best_solution, ga_training
import torch.nn

# Variables relating to the NN
NUMBER_OF_SECTORS = 3
include_edges = True
num_neurons_input = NUMBER_OF_SECTORS * 3 + 2 * include_edges
num_neurons_hidden_layer_1 = 9

# Create the structure of the neural network with five outputs (representing the four arrow keys and spacebar)
input_layer = torch.nn.Linear(num_neurons_input, num_neurons_hidden_layer_1)
leaky_relu_layer1 = torch.nn.LeakyReLU()
output_layer = torch.nn.Linear(num_neurons_hidden_layer_1, 5)
softmax_layer = torch.nn.Sigmoid()

model = torch.nn.Sequential(input_layer,
                            leaky_relu_layer1,
                            output_layer,
                            softmax_layer)

filename = 'ga_instances/se3_0'

# Create a new ga_instance
ga_instance = Genetic_algorithm_instance(filename=filename, model=model, NUMBER_OF_SECTORS=NUMBER_OF_SECTORS,
                                         keep_elitism=10, num_parents_mating=80, mutation_percent_genes=(15, 5))

# Training the genetic algorithm instance
ga_training(filename, filename, 10)

display_best_solution(filename)