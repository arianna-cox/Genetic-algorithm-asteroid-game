from fitness_function import *
import tensorflow.keras

do_draw = True

# Variables relating to the NN
NUMBER_OF_SECTORS = 1
num_neurons_input = NUMBER_OF_SECTORS * 3 + 2
num_neurons_hidden_layers = 5
num_solutions = 4
# Threshold for the outputs of the NN above which a key is considered pressed
threshold = 0.3

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
num_generations = 2
num_parents_mating = 2
mutation_percent_genes = 10
initial_population = keras_ga.population_weights


def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("best score = {fitness}".format(fitness=ga_instance.best_solution()[1]))


def prediction_function(NN, data):
    output = pygad.kerasga.predict(model=model, solution=NN, data=data)
    return output[0]


fitness_func = lambda NN, solution_index: fitness_function(NN, solution_index, prediction_function,
                                                           NUMBER_OF_SECTORS, threshold, do_draw=do_draw)

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       mutation_percent_genes=mutation_percent_genes,
                       on_generation=callback_generation)

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"best solution: {solution}")
print(f"best solution fitness = {solution_fitness}")
print(f"best solution index = {solution_idx}")
