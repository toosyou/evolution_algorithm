import numpy as np
from sklearn.decomposition import PCA
import copy
import matplotlib.pyplot as plt
from numba import jit
import better_exceptions

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

@jit
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

class Individual:
    def __init__(self, init_genes):
        self.genes = np.array(init_genes)

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

    def fitness(self, x, y):
        line_points = [ list() for _ in range(self.n_lines) ]
        for index_line, xi, yi in zip(self.genes, x, y):
            line_points[index_line].append([xi, yi])

        total_distance = 0.
        for points in line_points: # for each line
            pca = PCA(n_components=2).fit(points)
            main_components, mean = pca.components_[0], pca.mean_
            for p in points:
                total_distance += distance(p, mean, main_components)

        return -1. * total_distance

    def sex_with(self, another_guy):
        return IndividualP1(super(IndividualP1, self).sex_with(another_guy).genes, self.n_lines)

    def mutated(self, mutation_ratio):
        child_genes = copy.copy(self.genes)
        for index_gene, gene in enumerate(child_genes):
            if np.random.random() < mutation_ratio: # do mutate
                child_genes[index_gene] = (gene + int( np.random.random() < 0.5 ) * 2 - 1 ) % self.n_lines  # gene +- 1

        return IndividualP1(child_genes, self.n_lines)

def presentation_1_solver(filename='lineN100M4', n_population=100, n_generation=1000, keep_ratio=0.3, mutation_ratio=0.05):
    x, y = get_data('{}.txt'.format(filename))
    n_lines = datasets[filename]['M']

    # init polulation
    population = np.array([ IndividualP1( np.random.choice(n_lines, x.shape[0], replace=True), n_lines ) for _ in range(n_population) ])

    # from one generation to the next
    for index_generation in range(n_generation):
        # get fitness
        fitnesses = np.array([ p.fitness(x, y) for p in population ])
        population = population[ fitnesses.argsort()[::-1] ] # sort by fitness in desc order

        # selection
        population = population[ :int(n_population*keep_ratio) ]

        # sex party
        population = np.array([ a.sex_with(b) for a, b in zip(np.random.choice(population, n_population), np.random.choice(population, n_population)) ])

        # mutation
        population = np.array([ p.mutated(mutation_ratio) for p in population ])
        print('Generation:', index_generation, 'Maximun_fitness:', fitnesses.max())

if __name__ == '__main__':
    presentation_1_solver()
