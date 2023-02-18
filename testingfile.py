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

# Things to try!
# Have an option to not use the relative angle of the velocity
# Maybe do this by creating vectors of length NUMBER_OF_SECTORS