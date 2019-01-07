import math
from gaft import GAEngine
from gaft.components import BinaryIndividual, Population
from gaft.operators import RouletteWheelSelection, UniformCrossover, FlipBitMutation

# Analysis plugin base class.
from gaft.plugin_interfaces.analysis import OnTheFlyAnalysis
from gaft.analysis.fitness_store import FitnessStore

# In[17]:
# Define population
indv_template = BinaryIndividual(ranges=[(0, 10)], eps=0.001)
population = Population(indv_template=indv_template, size=50)
population.init()  # Initialize population with individuals.

# In[18]:
# Create genetic operators
# Use built-in operators here.
selection = RouletteWheelSelection()
crossover = UniformCrossover(pc=0.8, pe=0.5)
mutation = FlipBitMutation(pm=0.1)

# In[19]:
# Create genetic algorithm engine to run optimization
engine = GAEngine(population=population, selection=selection,
                  crossover=crossover, mutation=mutation,
                  analysis=[FitnessStore])


# In[20]:
# Define and register fitness function
# 最大化
@engine.fitness_register
def fitness(indv):
    x, = indv.solution
    return x + 10 * math.sin(5 * x) + 7 * math.cos(4 * x)


# 最小化
@engine.fitness_register
@engine.minimize
def fitness(indv):
    x, = indv.solution
    return x + 10 * math.sin(5 * x) + 7 * math.cos(4 * x)


# In[21]:
# Define and register an on-the-fly analysis (optional)
@engine.analysis_register
class ConsoleOutput(OnTheFlyAnalysis):
    master_only = True
    interval = 1

    def register_step(self, g, population, engine):
        best_indv = population.best_indv(engine.fitness)
        msg = 'Generation: {}, best fitness: {:.3f}'.format(g, engine.fmax)
        engine.logger.info(msg)


# In[22]:
if '__main__' == __name__:
    engine.run(ng=100)
