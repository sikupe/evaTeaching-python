import random
import numpy as np
import functools

import co_functions as cf
import utils
import matplotlib.pyplot as plt

DIMENSION = 10  # dimension of the problems
POP_SIZE = 100  # population size
MAX_GEN = 500  # maximum number of generations
CX_PROB = 0.8  # crossover probability
MUT_PROB = 0.2  # mutation probability
MUT_STEP = 0.5  # size of the mutation steps
MUT_STD_DEV = 2.0  # standard deviation of the mutation
MUT_DECR = 0.997231251  # decrease of standard deviation every generation (gen 0: 2.0 -> gen 500: 0.5)
CROSS_DECR = 0.998614666  # decrease of standard deviation every generation (gen 0: 2.0 -> gen 500: 0.5)
CR = 0.9
F = 0.8
REPEATS = 10  # number of runs of algorithm (should be at least 10)
OUT_DIR = 'lamarck'  # output directory for logs
EXP_ID = 'default'  # the ID of this experiment (used to create log names)


# creates the individual
def create_ind(ind_len):
    return np.random.uniform(-5, 5, size=(ind_len,))


# creates the population using the create individual function
def create_pop(pop_size, create_individual):
    return [create_individual() for _ in range(pop_size)]


# the tournament selection (roulette wheell would not work, because we can have
# negative fitness)
def tournament_selection(pop, fits, k):
    selected = []
    for i in range(k):
        p1 = random.randrange(0, len(pop))
        p2 = random.randrange(0, len(pop))
        if fits[p1] > fits[p2]:
            selected.append(np.copy(pop[p1]))
        else:
            selected.append(np.copy(pop[p2]))

    return selected


# implements the one-point crossover of two individuals
def one_pt_cross(p1, p2):
    point = random.randrange(1, len(p1))
    o1 = np.append(p1[:point], p2[point:])
    o2 = np.append(p2[:point], p1[point:])
    return o1, o2


def gradient(ind, fitness, eps=0.0000001):
    grad = np.zeros_like(ind)

    for i in range(len(ind)):
        x_plus = np.copy(ind)
        x_minus = np.copy(ind)
        x_plus[i] += eps
        x_minus[i] -= eps
        grad[i] = (fitness(x_plus).fitness - fitness(x_minus).fitness) / (2 * eps)

    return grad


class LamarckMutation:
    def __init__(self, fitness, step_size=.01, eps=.0000001):
        self.eps = eps
        self.fitness = fitness
        self.step_size = step_size

    def __call__(self, ind):
        grad = gradient(ind, self.fitness, self.eps)
        norm_grad = grad / np.linalg.norm(grad)
        return ind - self.step_size * norm_grad


class DifferentialCrossover:
    def __init__(self, CR, decrease):
        self.CR = CR
        self.decrease = decrease

    def next_generation(self):
        self.CR *= self.decrease

    def __call__(self, i1, i2):
        swap = False
        o1 = np.zeros_like(i1)
        o2 = np.zeros_like(i2)
        for i in range(len(i1)):
            if random.uniform(0, 1) > self.CR:
                swap = True
                o1[i] = i2[i]
                o2[i] = i1[i]
            else:
                o1[i] = i1[i]
                o2[i] = i2[i]

        if not swap:
            rand_swap = random.randrange(0, len(i1))
            tmp = o1[rand_swap]
            o1[rand_swap] = o2[rand_swap]
            o2[rand_swap] = tmp

        return o1, o2


# applies a list of genetic operators (functions with 1 argument - population)
# to the population
def mate(pop, operators):
    for o in operators:
        pop = o(pop)
    return pop


# applies the cross function (implementing the crossover of two individuals)
# to the whole population (with probability cx_prob)
def crossover(pop, cross, cx_prob):
    off = []
    for p1, p2 in zip(pop[0::2], pop[1::2]):
        if random.random() < cx_prob:
            o1, o2 = cross(p1, p2)
        else:
            o1, o2 = p1[:], p2[:]
        off.append(o1)
        off.append(o2)
    return off


# applies the mutate function (implementing the mutation of a single individual)
# to the whole population with probability mut_prob)
def mutation(pop, mutate, mut_prob):
    return [mutate(p) if random.random() < mut_prob else p[:] for p in pop]


# implements the evolutionary algorithm
# arguments:
#   pop_size  - the initial population
#   max_gen   - maximum number of generation
#   fitness   - fitness function (takes individual as argument and returns 
#               FitObjPair)
#   operators - list of genetic operators (functions with one arguments - 
#               population; returning a population)
#   mate_sel  - mating selection (funtion with three arguments - population, 
#               fitness values, number of individuals to select; returning the 
#               selected population)
#   mutate_ind - reference to the class to mutate an individual - can be used to 
#               change the mutation step adaptively
#   map_fn    - function to use to map fitness evaluation over the whole 
#               population (default `map`)
#   log       - a utils.Log structure to log the evolution run
def evolutionary_algorithm(pop, max_gen, fitness, operators, mate_sel, cross_ind, *, map_fn=map, log=None):
    evals = 0
    for G in range(max_gen):
        fits_objs = list(map_fn(fitness, pop))
        evals += len(pop)
        if log:
            log.add_gen(fits_objs, evals)
        fits = [f.fitness for f in fits_objs]
        objs = [f.objective for f in fits_objs]

        mating_pool = mate_sel(pop, fits, POP_SIZE)
        offspring = mate(mating_pool, operators)
        pop = offspring[:]
        cross_ind.next_generation()

    return pop


if __name__ == '__main__':

    # use `functool.partial` to create fix some arguments of the functions 
    # and create functions with required signatures
    cr_ind = functools.partial(create_ind, ind_len=DIMENSION)
    # we will run the experiment on a number of different functions
    fit_generators = [cf.make_f01_sphere,
                      cf.make_f02_ellipsoidal,
                      cf.make_f06_attractive_sector,
                      cf.make_f08_rosenbrock,
                      cf.make_f10_rotated_ellipsoidal]
    fit_names = ['f01', 'f02', 'f06', 'f08', 'f10']

    for fit_gen, fit_name in zip(fit_generators, fit_names):
        name = f'{EXP_ID}.{fit_name}'
        fit = fit_gen(DIMENSION)
        d_cross = DifferentialCrossover(CR, CROSS_DECR)
        d_mut = LamarckMutation(fit)
        xover = functools.partial(crossover, cross=d_cross, cx_prob=CX_PROB)
        mut = functools.partial(mutation, mut_prob=MUT_PROB, mutate=d_mut)

        # run the algorithm `REPEATS` times and remember the best solutions from
        # last generations

        best_inds = []
        for run in range(REPEATS):
            # initialize the log structure
            log = utils.Log(OUT_DIR, name, run,
                            write_immediately=True, print_frequency=5)
            # create population
            pop = create_pop(POP_SIZE, cr_ind)
            # run evolution - notice we use the pool.map as the map_fn
            pop = evolutionary_algorithm(pop, MAX_GEN, fit, [xover, mut], tournament_selection,
                                         d_cross,
                                         map_fn=map,
                                         log=log)
            # remember the best individual from last generation, save it to file
            bi = max(pop, key=fit)
            best_inds.append(bi)

            # if we used write_immediately = False, we would need to save the
            # files now
            # log.write_files()

        # print an overview of the best individuals from each run
        for i, bi in enumerate(best_inds):
            print(f'Run {i}: objective = {fit(bi).objective}')

        # write summary logs for the whole experiment
        utils.summarize_experiment(OUT_DIR, name)

        # read the summary log and plot the experiment
        evals, lower, mean, upper = utils.get_plot_data(OUT_DIR, name)
        utils.plot_experiment(evals, lower, mean, upper,
                              legend_name=f'Default settings {name}')
    plt.yscale('log')
    plt.legend()
    plt.savefig(f'{OUT_DIR}/stats-combined.png')
    plt.clf()
