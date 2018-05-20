import numpy as np
from sklearn.decomposition import PCA
import copy
import better_exceptions
from itertools import chain
from tqdm import tqdm
import sys

datasets = {
    'lineN100M4':{
        'N': 100,
        'M': 4,
    },
    'lineN200M3':{
        'N': 200,
        'M': 3,
    },
    'lineN300M5':{
        'N': 300,
        'M': 5,
    }
}

def get_data(filename='lineN100M4.txt'):
    x = list()
    y = list()
    with open(filename, 'r') as f:
        for line in f.readlines():
            x.append(float(line.split()[0]))
            y.append(float(line.split()[1]))
    return np.array(x), np.array(y)

class Individual:
    def __init__(self, init_genes):
        self.genes = np.array(init_genes)
        self.fitness = 0.

    def sex_with(self, another_guy):
        child_gene = list()
        for ga, gb in zip(self.genes, another_guy.genes):
            if np.random.random() < 0.5:
                child_gene.append(ga)
            else:
                child_gene.append(gb)
        # Only losers use condoms
        return Individual(child_gene)

    def __str__(self):
        return str(self.genes)

    def __repr__(self):
        return self.__str__()

class IndividualP1(Individual):
    def __init__(self, init_genes, n_lines):
        super(IndividualP1, self).__init__(init_genes)
        self.n_lines = n_lines

    @staticmethod
    def born(n_points, n_lines):
        return IndividualP1( np.random.choice(n_lines, n_points, replace=True), n_lines )

    def cal_fitness(self, x, y):
        def distance(p, line_p, line_vector):
            '''
                         p
                        ^
                       /
                va    /
                     /
                    /
            line_p -------------------------------> line_vector
            '''

            va = p - line_p
            va_length = np.sqrt(va.dot(va))
            line_vector_length = np.sqrt(line_vector.dot(line_vector))
            cos = ( va.dot(line_vector) ) / ( va_length * line_vector_length )
            sin = np.sqrt( 1. - cos*cos)

            return va_length * sin

        line_points = [ list() for _ in range(self.n_lines) ]
        for index_line, xi, yi in zip(self.genes, x, y):
            line_points[index_line].append([xi, yi])

        total_distance = 0.
        for points in line_points: # for each line
            if not len(points): continue # prevent exploding
            pca = PCA(n_components=2).fit(points)
            main_components, mean = pca.components_[0], pca.mean_
            for p in points:
                total_distance += distance(p, mean, main_components)

        self.fitness = -1. * total_distance
        return None

    def sex_with(self, another_guy):
        return IndividualP1(super(IndividualP1, self).sex_with(another_guy).genes, self.n_lines)

    def mutated(self, mutation_ratio):
        child_genes = copy.copy(self.genes)
        for index_gene, gene in enumerate(child_genes):
            if np.random.random() < mutation_ratio: # do mutate
                child_genes[index_gene] = (gene + int( np.random.random() < 0.5 ) * 2 - 1 ) % self.n_lines  # gene +- 1

        return IndividualP1(child_genes, self.n_lines)

class IndividualP2(Individual):
    def __init__(self, init_genes):
        super(IndividualP2, self).__init__(init_genes)

    @staticmethod
    def born(n_points, n_lines):
        gene = [ [np.random.random() * 2. * np.pi, np.random.random() * 10.] for _ in range(n_lines) ]
        gene = list(chain.from_iterable(gene)) # flatten
        return IndividualP2( gene )

    def cal_fitness(self, x, y):
        def distance(xi, yi, theta, rho):
            return abs( xi * np.cos(theta) + yi * np.sin(theta) - rho )

        total_distance = 0.
        for xi, yi in zip(x, y):
            distances = [ distance(xi, yi, theta, rho) for theta, rho in zip(self.genes[::2], self.genes[1::2]) ]
            total_distance += min(distances)

        self.fitness = -1. * total_distance
        return None

    def sex_with(self, another_guy):
        return IndividualP2(super(IndividualP2, self).sex_with(another_guy).genes)

    def mutated(self, mutation_ratio):
        child_genes = copy.copy(self.genes)
        for index_gene, gene in enumerate(child_genes):
            if np.random.random() < mutation_ratio: # do mutate
                child_genes[index_gene] += np.random.random() - 0.5

        return IndividualP2(child_genes)

def evolve(IndividualType=IndividualP1, filename='lineN100M4', n_population=500, n_generation=100, keep_ratio=0.2, mutation_ratio=0.1, verbose=True):
    x, y = get_data('{}.txt'.format(filename))
    n_lines = datasets[filename]['M']
    n_points = x.shape[0]
    history = {
        'fitness': list()
    }

    # init polulation
    population = np.array([ IndividualType.born(n_points, n_lines) for _ in range(n_population) ])

    # from one generation to the next
    for index_generation in range(n_generation):
        # get fitness
        for p in population: p.cal_fitness(x, y)
        population = sorted(population, key=lambda p: p.fitness)[::-1] # sort by fitness in desc order
        history['fitness'].append(population[0].fitness)

        if verbose:
            print('Generation:', index_generation, 'Maximun_fitness:', population[0].fitness, file=sys.stderr)

        if index_generation == n_generation - 1: # the last gen
            return population[0], history # fits the most

        # selection
        population = population[ :int(n_population*keep_ratio) ]

        # sex party
        population = np.array([ a.sex_with(b) for a, b in zip(
                        np.random.choice(population, n_population),
                        np.random.choice(population, n_population))
                        ])

        # mutation
        population = np.array([ p.mutated(mutation_ratio) for p in population ])

if __name__ == '__main__':
    filenames = ["lineN100M4", "lineN200M3", "lineN300M5"]

    for fn in filenames[:2]:
        # matic parameters, don't touch it
        print(evolve(IndividualP1, filename=fn,
                        n_population=500,
                        n_generation=150,
                        keep_ratio=0.009,
                        mutation_ratio=0.05,
                        verbose=True))
    print(evolve(IndividualP1, filename='lineN300M5',
                    n_population=500,
                    n_generation=200,
                    keep_ratio=0.05,
                    mutation_ratio=0.01,
                    verbose=True))
