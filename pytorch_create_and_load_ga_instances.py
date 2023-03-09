from fitness_function import fitness_function
import numpy as np
import torch.nn
import pygad.torchga

class Genetic_algorithm_instance(pygad.GA):
    def __init__(self, filename=None, model=None, NUMBER_OF_SECTORS=3, include_edges=True, threshold=0.5,
                 do_draw_training=False, num_solutions=100, num_generations=1, keep_elitism=5,
                 keep_parents=0, num_parents_mating=60, mutation_percent_genes=(15, 5)):

        self.model = model
        self.NUMBER_OF_SECTORS = NUMBER_OF_SECTORS
        self.include_edges = include_edges
        self.threshold = threshold
        self.do_draw_training = do_draw_training
        self.num_solutions = num_solutions
        self.num_generations = num_generations
        self.keep_elitism = keep_elitism,
        self.keep_parents = keep_parents
        self.num_parents_mating = num_parents_mating
        self.mutation_percent_genes = mutation_percent_genes

        def fitness_func(NN, solution_index):
            return fitness_function(NN, solution_index, self.prediction_function, self.NUMBER_OF_SECTORS,
                                    self.threshold, do_draw=self.do_draw_training, include_edges=self.include_edges)

        def callback_generation(ga_instance):
            print("on_callback()")

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

        # Create initial population
        torch_ga = pygad.torchga.TorchGA(model=model, num_solutions=num_solutions)
        initial_population = torch_ga.population_weights

        super().__init__(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           parent_selection_type="rws",
                           keep_parents=keep_parents,
                           keep_elitism=keep_elitism,
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

        self.save(filename=filename)

    def prediction_function(self, solution, data):
        model_weights_dictionary = pygad.torchga.model_weights_as_dict(model=self.model, weights_vector=solution)
        self.model.load_state_dict(model_weights_dictionary)
        output = self.model(torch.tensor(data[0]).float()).detach().numpy()
        return output

def ga_training(load_filename, save_filename, number_of_generations):
    ga_instance = pygad.load(filename=load_filename)
    for i in range(number_of_generations):
        ga_instance.run()
        print(f'In training loop = {i}')
        ga_instance.save(filename=save_filename)
        print("saved!")

def display_best_solution(ga_filename, number_of_games=100):
    # Load the ga_instance
    GA = pygad.load(filename=ga_filename)

    # Explore results
    GA.plot_fitness(title="Best fitness vs generation")

    best_solution, best_solution_fitness, best_solution_idx = GA.best_solution()
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=best_solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=best_solution_idx))

    for i in range(number_of_games):
        fitness_function(best_solution, 0, GA.prediction_function, GA.NUMBER_OF_SECTORS,
                         GA.threshold, do_draw=True, include_edges=GA.include_edges)
