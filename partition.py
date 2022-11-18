import random
from typing import TypeVar, List, Callable, Tuple
import functools

import numpy as np
import matplotlib.pyplot as plt

import utils

K = 10  # number of piles
POP_SIZE = 100  # population size
MAX_GEN = 500  # maximum number of generations
CX_PROB = 0.8  # crossover probability
MUT_PROB = 0.2  # mutation probability
MUT_FLIP_PROB = 0.1  # probability of chaninging value during mutation
REPEATS = 10  # numb
# er of runs of algorithm (should be at least 10)
OUT_DIR = 'partition'  # output directory for logs
EXP_ID = 'default'  # the ID of this experiment (used to create log names)

Individual = TypeVar("Individual")
Population = List[Individual]
Operator = Callable[[Population], Population]
FitnessFunction = Callable[[Individual], utils.FitObjPair]


# reads the input set of values of objects
def read_weights(filename: str) -> List[int]:
    with open(filename) as f:
        return list(map(int, f.readlines()))


# computes the bin weights
# - bins are the indices of bins into which the object belongs
def bin_weights(weights: List[int], bins: Individual) -> List[int]:
    bw = [0] * K
    for w, b in zip(weights, bins):
        bw[b] += w
    return bw


# the fitness function
def fitness(ind: Individual, weights: List[int]) -> utils.FitObjPair:
    bw = bin_weights(weights, ind)
    return utils.FitObjPair(fitness=1 / (max(bw) - min(bw) + 1),
                            objective=max(bw) - min(bw))


def fitness_std_base(ind: Individual, weights: List[int], exponent=2) -> utils.FitObjPair:
    bw = np.array(bin_weights(weights, ind))
    mean_bin = sum(weights) / K
    var = np.sum((bw - mean_bin) ** 2)
    std = np.sqrt(var)
    fitness = 1 / std
    # Taking the negative variance, as we are searching for a maximum
    return utils.FitObjPair(fitness=fitness, objective=max(bw) - min(bw))


def fitness_std(ind: Individual, weights: List[int]) -> utils.FitObjPair:
    return fitness_std_base(ind, weights)


def fitness_std_modified(ind: Individual, weights: List[int]) -> utils.FitObjPair:
    return fitness_std_base(ind, weights, 3)


# creates the individual
def create_ind(ind_len: int) -> Individual:
    return [random.randrange(0, K) for _ in range(ind_len)]


# creates the population using the create individual function
def create_pop(pop_size: int, create_individual: Callable[[], Population]) -> Population:
    return [create_individual() for _ in range(pop_size)]


# the roulette wheel selection
def roulette_wheel_selection(pop: Population, fits: List[float], k) -> Population:
    return random.choices(pop, fits, k=k)


def sus_selection(pop: Population, fits: List[float], k) -> Population:
    total_size = sum(fits)
    cum_fits = np.cumsum(fits)
    step_size = total_size / k
    start = random.uniform(0, total_size)
    off = []
    current_ind = 0
    for i in range(k):
        step = (start + i * step_size) % total_size

        lower_bound = cum_fits[current_ind - 1] % total_size
        upper_bound = cum_fits[current_ind]

        while not (lower_bound < step <= upper_bound):
            current_ind += 1
            current_ind %= len(pop)
            lower_bound = cum_fits[current_ind - 1] % total_size
            upper_bound = cum_fits[current_ind]
        off.append(pop[current_ind])
    return off


def tournament_selection(pop: Population, fits: List[float], offspring_size) -> Population:
    k = 2

    off = []
    for i in range(offspring_size):
        random_inds = [random.randint(0, len(pop) - 1) for _ in range(k)]

        max_ind = None
        max_fit = 0
        for ind in random_inds:
            if fits[ind] > max_fit:
                max_ind = ind
                max_fit = fits[ind]

        off.append(pop[max_ind])

    return off


# implements the one-point crossover of two individuals
def one_pt_cross(p1: Individual, p2: Individual) -> (Individual, Individual):
    point = random.randrange(1, len(p1))
    o1 = p1[:point] + p2[point:]
    o2 = p2[:point] + p1[point:]
    return o1, o2


# implements the "bit-flip" mutation of one individual
def flip_mutate(p: Individual, prob: float, upper: int) -> Individual:
    return [random.randrange(0, upper) if random.random() < prob else i for i in p]


# applies a list of genetic operators (functions with 1 argument - population)
# to the population
def mate(pop: Population, operators: List[Operator]):
    for o in operators:
        pop = o(pop)
    return pop


# applies the cross function (implementing the crossover of two individuals)
# to the whole population (with probability cx_prob)
def crossover(pop: Population, cross: Callable[[Individual, Individual], Tuple[Individual, Individual]],
              cx_prob: float) -> Population:
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
def mutation(pop: Population, mutate: Callable[[Individual], Individual], mut_prob: float) -> Population:
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
#   map_fn    - function to use to map fitness evaluation over the whole 
#               population (default `map`)
#   log       - a utils.Log structure to log the evolution run
def evolutionary_algorithm(pop: Population, max_gen: int, fitness: FitnessFunction,
                           operators: List[Operator], mate_sel: Callable[[Population, List[float], int], Population], *,
                           map_fn=map, log=None) -> Population:
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

    return pop


def informed_mutate(p: Individual, prob: float) -> Individual:
    bw = bin_weights(weights, p)

    mutated = []
    for i in range(len(p)):
        mutated.append(p[i])

        if random.random() < prob:
            weight = weights[i]
            bucket = p[i]
            possible_buckets = []
            for j, bucket_weight in enumerate(bw):
                if bucket != j and (bucket_weight + weight < bw[bucket] or random.random() < prob):
                    possible_buckets.append(j)

            if len(possible_buckets) > 0:
                new_bucket = random.choice(possible_buckets)

                mutated[i] = new_bucket

    return mutated


if __name__ == '__main__':
    # read the weights from input
    weights = read_weights('inputs/partition-easy.txt')
    plt.figure(figsize=(12, 8))
    for mut_name, mutate in [(flip_mutate.__name__, functools.partial(flip_mutate, upper=K)), (informed_mutate.__name__, informed_mutate)]:
        EXP_ID = f'{mut_name.replace("_", "-")}'
        # use `functool.partial` to create fix some arguments of the functions
        # and create functions with required signatures
        cr_ind: Callable[[], Individual] = functools.partial(create_ind, ind_len=len(weights))
        fit: FitnessFunction = functools.partial(fitness_std, weights=weights)
        xover: Operator = functools.partial(crossover, cross=one_pt_cross, cx_prob=CX_PROB)
        mut: Operator = functools.partial(mutation, mut_prob=MUT_PROB,
                                          mutate=functools.partial(mutate, prob=MUT_FLIP_PROB))

        # we can use multiprocessing to evaluate fitness in parallel
        import multiprocessing

        pool = multiprocessing.Pool()

        # run the algorithm `REPEATS` times and remember the best solutions from
        # last generations
        best_inds = []
        for run in range(REPEATS):
            # initialize the log structure
            log = utils.Log(OUT_DIR, EXP_ID, run, write_immediately=True, print_frequency=5)
            # create population
            pop = create_pop(POP_SIZE, cr_ind)
            # run evolution - notice we use the pool.map as the map_fn
            pop = evolutionary_algorithm(pop, MAX_GEN, fit, [xover, mut], sus_selection, map_fn=pool.map,
                                         log=log)
            # remember the best individual from last generation, save it to file
            bi = max(pop, key=fit)
            best_inds.append(bi)

            with open(f'{OUT_DIR}/{EXP_ID}_{run}.best', 'w') as f:
                for w, b in zip(weights, bi):
                    f.write(f'{w} {b}\n')

            # if we used write_immediately = False, we would need to save the
            # files now
            # log.write_files()

        # print an overview of the best individuals from each run
        for i, bi in enumerate(best_inds):
            print(f'Run {i}: difference = {fit(bi).objective}, bin weights = {bin_weights(weights, bi)}')

        # write summary logs for the whole experiment
        utils.summarize_experiment(OUT_DIR, EXP_ID)

        # read the summary log and plot the experiment
        evals, lower, mean, upper = utils.get_plot_data(OUT_DIR, EXP_ID)
        utils.plot_experiment(evals, lower, mean, upper,
                              legend_name=f'Default settings {EXP_ID}')
    plt.legend()
    plt.savefig(f'partition/stats-combined.png')
    plt.clf()
        # plt.show()

        # you can also plot mutiple experiments at the same time using
        # utils.plot_experiments, e.g. if you have two experiments 'default' and
        # 'tuned' both in the 'partition' directory, you can call
        # utils.plot_experiments('partition', ['default', 'tuned'],
        #                        rename_dict={'default': 'Default setting'})
        # the rename_dict can be used to make reasonable entries in the legend -
        # experiments that are not in the dict use their id (in this case, the
        # legend entries would be 'Default settings' and 'tuned')
