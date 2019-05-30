# @Time : 2019/4/22 10:54
# @Author : YingbinQiu
# @Site :
# @File : gene.py

import math
import os
import numpy as np
from gaft import GAEngine
from gaft.components import Population, DecimalIndividual
from gaft.operators import RouletteWheelSelection, UniformCrossover, FlipBitMutation, TournamentSelection

# Analysis plugin base class.
from gaft.plugin_interfaces.analysis import OnTheFlyAnalysis
from gaft.analysis.fitness_store import FitnessStore


def frame_call():
    # Define population
    mapgene = {
        "bench_qeueu_up": (-1, 1),
        "bench_qeueu_down": (-1, 1),
        "replenish_queue_up": (-1, 1),
        "replenish_queue_down": (-1, 1),
        "cell_num_thresh": (-1, 1),
        "cold_round": (-1, 1),
        "cold_seconds": (-1, 1),
        "order_buff_low": (-1, 1),
    }
    listindi = [i1 for i1 in board.values()]
    indv_templatec = DecimalIndividual(ranges=listindi, eps=0.001)
    # print('\n'.join(['%s:%s' % item for item in indv_templatec.__dict__.items()]))
    population = Population(indv_template=indv_templatec, size=6)
    population.init()  # Initialize population with individuals.
    # Create genetic operators
    # Use built-in operators here.
    selection = RouletteWheelSelection()
    crossover = UniformCrossover(pc=0.8, pe=0.5)
    mutation = FlipBitMutation(pm=0.1)

    # Create genetic algorithm engine to run optimization
    engine = GAEngine(population=population, selection=selection, crossover=crossover, mutation=mutation,
                      analysis=[FitnessStore])

    # Define and register fitness function
    # 最小化
    @engine.fitness_register
    @engine.minimize
    def fitness(indv):
        print("fitness")
        x = indv.solution
        npnowjs = np.array(x)
        # print(x)
        nprandom = np.random.normal(npnowjs, npmetric, npnowjs.shape[0])
        for i1, i2 in zip(nprandom, npboard):
            if i1 > i2[1] and i1 < i2[0]:
                return 1.e9
        # print(nprandom)
        return 1.0
        # return x + 10 * math.sin(5 * x) + 7 * math.cos(4 * x)

    # Define and register an on-the-fly analysis (optional)
    @engine.analysis_register
    class ConsoleOutput(OnTheFlyAnalysis):
        master_only = True
        interval = 1
        # 保存的文件名
        _save_fp = "my_best_fit.py"

        @property
        def save_fp(self):
            print("prorperty")
            return self._save_fp

        @save_fp.setter
        def save_fp(self, fp):
            print("set")
            self._save_fp = fp

        def finalize(self, population, engine):
            print("finnal")
            with open(self._save_fp, 'w', encoding='utf-8') as f:
                f.write('best_f2it = [\n')
                for ng, x, y in zip(self.ngs, self.solution, self.fitness_values):
                    f.write('    ({}, {}, {}),\n'.format(ng, x, y))
                f.write(']\n\n')
            self.logger.info('Best fitness values are written to best_fit.py')
            print(population)
            print(engine)

        def setup(self, ng, engine):
            # Generation numbers.
            self.ngs = []
            # Best fitness in each generation.
            self.fitness_values = []
            # Best solution.
            self.solution = []

        def register_step(self, g, population, engine):
            print("register_step")
            # Collect data.
            best_indv = population.best_indv(engine.fitness)
            best_fit = engine.ori_fmax
            msg = 'Generation: {}, best fitness: {:.3f}'.format(g, engine.fmax)
            engine.logger.info(msg)

            self.ngs.append(g)
            self.solution.append(best_indv.solution)
            self.fitness_values.append(best_fit)
            print(self.ngs)
            print(self.solution)
            print(self.fitness_values)

    return engine


sku_weigh = {}
skumap = {}
board = {
    "bench_queue_up": (2, 50),
    "bench_queue_down": (1, 20),
    "replenish_queue_up": (2, 50),
    "replenish_queue_down": (1, 20),
    "cell_num_thresh": (0, 1),
    "cold_round": (10, 100),
    "cold_seconds": (60, 36000),
    "order_buff_low": (10, 500),
}
nowjs = {
    "bench_queue_up": 20,
    "bench_queue_down": 5,
    "replenish_queue_up": 20,
    "replenish_queue_down": 5,
    "cell_num_thresh": 0.8,
    "cold_round": 30,
    "cold_seconds": 2000,
    "order_buff_low": 50,
}
metric = {
    "bench_queue_up": 2,
    "bench_queue_down": 1,
    "replenish_queue_up": 2,
    "replenish_queue_down": 1,
    "cell_num_thresh": 0.2,
    "cold_round": 1,
    "cold_seconds": 600,
    "order_buff_low": 10,
}

npboard = np.array([i1 for i1 in board.values()])
npmetric = np.array([i1 for i1 in metric.values()])

if '__main__' == __name__:
    engine = frame_call()
    engine.run(ng=100)
    # After engine running
    print(4)
    best_indv = engine.population.best_indv(engine.fitness)
    # Get the solution
    print(best_indv.solution)
    # And the fitness value
    print(engine.fitness(best_indv))
    exit(0)
