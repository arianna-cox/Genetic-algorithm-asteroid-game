import cProfile
import pstats
from training_pytorch import *

profiler = cProfile.Profile()
profiler.enable()


ga_instance.run()


profiler.disable()
stats = pstats.Stats(profiler).sort_stats('tottime')

# Print the stats report
stats.print_stats()