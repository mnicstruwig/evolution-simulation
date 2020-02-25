import numpy as np
from copy import copy
import random
import matplotlib.pyplot as plt
import pandas as pd
from plotnine import *
from collections import namedtuple
from tqdm import tqdm


class Species:
    def __init__(self, genome, gene_length):
        self.genome = genome
        assert gene_length % 2 == 0  # TODO: Wrap in try-catch
        self.gene_length = gene_length

    def make_animal(self, genes=None, **kwargs):
        if genes is None:
            genes = np.random.choice(self.genome, self.gene_length)

        return Animal(
            genome=self.genome,
            genes=genes,
            **kwargs
        )
    def breed(self, a, b):
        gene_pair_a = a.get_gene_pair()
        gene_pair_b = b.get_gene_pair()
        return self.make_animal(genes = [*gene_pair_a, *gene_pair_b])


class Animal:
    def __init__(self, genome, genes):
        self.genome = genome
        self.genes = genes
        self.gene_counts = self._get_gene_counts()
        self.gene_effects = {
            'food': [('a', 0.1)]
        }  # TODO: Use this to implement gene effects
        self.age = 0
        self.food = 1
        self.reproduction_params = self._get_reproduction_proba()

    def __repr__(self):
        return f'Animal(genome={self.genome}, genes={self.genes})'

    def does_die(self):
        if self.food <= 0:  # Die from hunger
            return True
        x = np.random.binomial(**self._get_death_params())
        if x == 1:
            return True
        return False

    def does_reproduce(self):
        x = np.random.binomial(**self._get_reproduction_params())
        if x == 1:
            return True
        return False

    def does_mutate(self):
        x = np.random.binomial(**self._get_mutation_params())
        if x == 1:
            return True
        return False

    def _get_gene_counts(self):
        gene_counts = {g:0 for g in self.genome}
        genes, counts = np.unique(self.genes, return_counts=True)
        x = {g:v for g,v in zip(genes, counts)}
        gene_counts.update(x)
        return gene_counts

    @staticmethod
    def _fast_sample(p):
        if random.random() <= p:
            return True
        return False

    def does_find_food(self):
        p = self._get_food_proba()
        return self._fast_sample(p)

    def eat(self):
        self.food -= 1

    def grow_older(self):
        self.age += 1

    def breed_with(self, other_animal):
        offspring_genes_self = self._get_gene_pair()
        offspring_genes_other = other_animal._get_gene_pair()
        offspring_genes = np.array([*offspring_genes_self, *offspring_genes_other])

        offspring = Animal(offspring_genes)

        if self.does_mutate():
            offspring._mutate()

        return Animal(offspring_genes)

    def get_gene_pair(self):
        half_gene_length = int(len(self.genes)/2)  # Choose upper or lower gene
        choose_first_pair = int(2*random.random())
        if choose_first_pair:
            return self.genes[:half_gene_length]
        return self.genes[half_gene_length:]

    def _mutate(self):
        mutation = int(len(self.genes)*random.random())
        self.genes[mutation] = ['a', 'b', 'c','d'][mutation]

    def _get_reproduction_proba(self):
        p = 0.5
        return {'n': 1, 'p': self._clip_p(p)}

    def _get_mutation_proba(self):
        return {'n':1, 'p': 0.01}

    def _get_food_proba(self):
        gene_effects = self.gene_effects['food']
        modifiers = self._calc_modifiers(gene_effects)
        p = 0.6 + modifiers
        return self._clip_p(p)

    def _get_death_proba(self):
        number_a_genes = len(np.where(self.genes == 'a')[0])
        modifier = (
            + 0.15 * self.age * number_a_genes
            + 0.05 * self.age
        )
        p = 0.05 + modifier
        return {'n': 1, 'p': self._clip_p(p)}

    def _calc_modifiers(self, gene_effects_list):
        modifiers = [self.gene_counts[g]*v for g,v in gene_effects_list]
        return np.sum(modifiers)


    @staticmethod
    def _generate_genes(genome, gene_length):
        return np.random.choice(genome, gene_length)

    def _clip_p(self, p):
        if p >= 1:
            return 1
        if p<= 0:
            return 0
        return p


State = namedtuple('State', ['step', 'pop_count', 'pop_mean_age', 'pop_genes', 'n_food'])

class Environment:
    def __init__(self, n_individuals, n_food=30, food_rate=5):
        self.animals = [Animal() for i in np.arange(n_individuals)]
        self.n_food = n_food
        self.food_rate = food_rate
        self.pop_count = n_individuals
        self.pop_genes = self.get_population_genetics()
        self.history = []
        self.steps = 0

    def age_population(self):
        for animal in self.animals:
            animal.grow_older()

    def simulate_deaths(self):
        survived = []
        for animal in self.animals:
            if not animal.does_die():
                survived.append(animal)
        self.animals = survived

    def simulate_eating(self):
        indexes = np.arange(0, len(self.animals))
        np.random.shuffle(indexes) # Animals eat in random order
        for index in indexes:
            if self.n_food > 0:  # If there's food in the environment ...
              if self.animals[index].does_find_food():  # ... and we find food
                  self.animals[index].food += 1  # ... we get the food
                  self.n_food -= 1  # ... but it's removed from the environment.
            self.animals[index].eat()  # But we always have to eat.

    def simulate_offspring(self):
        if len(self.animals) > 1:
          for animal in self.animals:
              if animal.does_reproduce():
                  other_idx = np.random.randint(0, len(self.animals))
                  other_animal = self.animals[other_idx]
                  new_animal = animal.breed_with(other_animal)
                  self.animals.append(new_animal)

    def get_pop_age(self):
        total_age = np.sum([animal.age for animal in self.animals])
        return total_age/len(self.animals)

    def get_population_genetics(self):  # TODO: Find a way of doing this faster
        gene_counts = {}
        try:
            genes = np.concatenate([x.genes for x in self.animals])
        except ValueError:  # Only one individual left
            try:
                genes = self.animals[0].genes
            except IndexError:  # No individuals left
                return {'a':0, 'b':0, 'c':0, 'd':0}
        gene, counts = np.unique(genes, return_counts=True)
        for g in ['a', 'b', 'c', 'd']:
            try:
                idx = np.where(gene == g)
                gene_counts[g] = counts[idx][0]
            except IndexError:
                gene_counts[g] = 0

        total_genes = np.sum(list(gene_counts.values()))
        return {k:v/total_genes for k,v in gene_counts.items()}

    def advance(self):
        self.n_food += self.food_rate
        self.simulate_offspring()
        self.simulate_eating()
        self.simulate_deaths()
        self.age_population()

        self.pop_count = len(self.animals)
        self.pop_mean_age = self.get_pop_age()
        self.pop_genes = self.get_population_genetics()


        self.steps += 1
        return State(self.steps, self.pop_count, self.pop_mean_age, self.pop_genes, self.n_food)

    def run(self, n_iters=100):
        for i in range(n_iters):
            new_state = self.advance()
            self.history.append(new_state)
            if new_state.pop_count <= 0:
                return self.history
            if new_state.pop_count >= 10000:
                return self.history
        return self.history


def process_history(history):
    return pd.DataFrame({
        'step': [s.step for s in history],
        'pop_count': [s.pop_count for s in history],
        'pop_mean_age': [s.pop_mean_age for s in history],
        'n_food': [s.n_food for s in history],
        'gene_a': [s.pop_genes['a'] for s in history],
        'gene_b': [s.pop_genes['b'] for s in history],
        'gene_c': [s.pop_genes['c'] for s in history],
        'gene_d': [s.pop_genes['d'] for s in history],
    })

def run_series(n_runs=3, n_individuals=20, n_food=200, food_rate=50, n_iters=100):
    results = []
    for i in tqdm(range(n_runs)):
        env = Environment(n_individuals, n_food, food_rate)
        history = env.run(n_iters=n_iters)
        df_history = process_history(history)
        df_history['run_no'] = i
        results.append(df_history)
    return pd.concat(results)


# df = run_series(n_runs=3, n_individuals=1000, n_food=1000, food_rate=300, n_iters=500)

# df_melt = df.melt(id_vars=['step', 'run_no'], value_vars=['gene_a', 'gene_b', 'gene_c', 'gene_d'])

# p = ggplot(aes(x='step', y='value', color='variable'), df_melt)
# p = p + geom_point(alpha=0.1) + geom_smooth(method='loess') #+ facet_grid('run_no ~ .')
# p.__repr__()


# df_x = df.groupby('step').mean().reset_index()
# df_x_melt = df.melt(id_vars=['step'], value_vars=['gene_a', 'gene_b', 'gene_c', 'gene_d'])


# p = ggplot(aes(x='variable', y='step', fill='value'), df_x_melt)
# p = p + geom_tile()
# p.__repr__()



# df_joy = df[['gene_a', 'gene_b', 'gene_c', 'gene_d']].copy()
# fig, axes = joypy.joyplot(df_joy)
# plt.show()
