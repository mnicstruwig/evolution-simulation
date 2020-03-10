import random
import warnings
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotnine import *
from tqdm import tqdm

State = namedtuple('State', ['step', 'pop_count', 'pop_mean_age', 'pop_genes', 'n_food'])


class Species:
    def __init__(self, genome, gene_length, gene_effects={}):
        self.genome = genome
        assert gene_length % 2 == 0  # TODO: Wrap in try-catch
        self.gene_length = gene_length
        self.gene_effects = gene_effects

        self.validate()

    def validate(self):
        if len(self.gene_effects) == 0:
            warnings.warn('No gene effects have been specified!')

    def make_animal(self, genes=None, **kwargs):
        if genes is None:
            genes = np.random.choice(self.genome, self.gene_length)

        return Animal(
            genome=self.genome,
            genes=genes,
            gene_effects=self.gene_effects,
            **kwargs
        )

    def breed(self, a, b):
        gene_pair_a = a.get_gene_pair()
        gene_pair_b = b.get_gene_pair()
        new_animal = self.make_animal(genes=[*gene_pair_a, *gene_pair_b])

        if new_animal.does_mutate():  # Allow for possibility of mutation
            new_animal.mutate()
        return new_animal


class Animal:
    def __init__(self, genome, genes, gene_effects={}):
        self.genome = genome
        self.genes = genes
        self.gene_counts = self._get_gene_counts()
        self.gene_effects = gene_effects
        self.age = 0
        self.food = 1

        self.validate()

    def __repr__(self):
        return f'Animal(genome={self.genome}, genes={self.genes}, age={self.age})'

    def validate(self):
        """Validate the input arguments."""
        pass

    def does_die(self):
        if self.food <= 0:  # Die from hunger
            return True
        p = self._get_death_proba()
        return self._fast_sample(p)

    def does_find_food(self):
        p = self._get_food_proba()
        return self._fast_sample(p)

    def does_reproduce(self):
        p = self._get_reproduction_proba()
        return self._fast_sample(p)

    def does_mutate(self):
        p = self._get_mutation_proba()
        return self._fast_sample(p)

    def _get_gene_counts(self):
        gene_counts = {g:0 for g in self.genome}
        genes, counts = np.unique(self.genes, return_counts=True)
        x = {g:v for g,v in zip(genes, counts)}
        gene_counts.update(x)
        return gene_counts

    @staticmethod
    def _fast_sample(p):
        """Sample a binomial random variable, with probability `p` quickly."""
        if random.random() <= p:
            return True
        return False

    def eat(self):
        self.food -= 1

    def grow_older(self):
        self.age += 1

    def get_gene_pair(self):
        half_gene_length = int(len(self.genes)/2)  # Choose upper or lower gene
        choose_first_pair = int(2*random.random())
        if choose_first_pair:
            return self.genes[:half_gene_length]
        return self.genes[half_gene_length:]

    def mutate(self):
        mutation = int(len(self.genes)*random.random())
        self.genes[mutation] = ['a', 'b', 'c', 'd'][mutation]

    def _get_death_proba(self):
        modifier = (
            + 0.15 * self.age * self._calc_modifiers('death_multiplier')
            + 0.05 * self.age
        )
        p = 0.05 + modifier
        return self._clip_p(p)

    def _get_food_proba(self):
        modifiers = self._calc_modifiers('food')
        p = 0.6 + modifiers
        return self._clip_p(p)

    def _get_reproduction_proba(self):
        modifiers = self._calc_modifiers('reproduce')
        p = 0.5 + modifiers
        return self._clip_p(p)

    def _get_mutation_proba(self):
        modifiers = self._calc_modifiers('mutate')
        p = 0.01 + modifiers
        return self._clip_p(p)

    def _calc_modifiers(self, which_effect):
        """
         Calculate the modifiers as a result of genes for `which_effect`.

         Parameters
         ----------
         which_effect : str
             Which effect to calculate the modifiers for. Possible values are
             {'food', 'death_multiplier', 'mutate', 'reproduce'}

         Returns
         -------
         float
             The modifier to the selected effect (this should be added) to the
             base probability of the selected effect.

         """
        try:
            gene_effects_list = self.gene_effects[which_effect]
        except KeyError:
            return 0
        # Items in `gene_effects_list` are `tuples` of length 2!
        # The first value is the gene, the second is the size of its effect.
        modifiers = [self.gene_counts[gene]*effect
                     for gene, effect in gene_effects_list]

        return np.sum(modifiers)

    def _clip_p(self, p):
        if p >= 1:
            return 1
        if p <= 0:
            return 0
        return p


class Environment:
    def __init__(self, species, n_individuals, n_food=30, food_rate=5):
        self.species = species
        self.animals = [self.species.make_animal() for i in range(n_individuals)]
        self.n_food = n_food
        self.food_rate = food_rate
        self.pop_count = n_individuals
        self.pop_genes = self._get_population_genetics()
        self.history = []
        self.steps = 0

    def age_population(self):
        for animal in self.animals:
            animal.grow_older()

    def simulate_deaths(self):
        survived = []
        for animal in self.animals:
            if not animal.does_die():
                survived.append(animal)  # TODO: Consider using `del`
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
                    new_animal = self.species.breed(animal, other_animal)
                    self.animals.append(new_animal)

    def get_pop_age(self):
        total_age = np.sum([animal.age for animal in self.animals])
        return total_age/len(self.animals)

    def _get_population_genetics(self):  # TODO: Find a way of doing this faster
        gene_counts = {}
        try:
            # Get all the genes in the population
            genes = np.concatenate([x.genes for x in self.animals])
        except ValueError:  # Only one individual left
            try:
                genes = self.animals[0].genes
            except IndexError:  # No individuals left
                return {k: 0 for k in self.species.genome}

        gene, counts = np.unique(genes, return_counts=True)
        for g in self.species.genome:
            try:
                idx = np.where(gene == g)
                gene_counts[g] = counts[idx][0]
            except IndexError:
                gene_counts[g] = 0

        total_genes = np.sum(list(gene_counts.values()))
        return {k: v/total_genes for k, v in gene_counts.items()}

    def advance(self):
        self.n_food += self.food_rate
        self.simulate_offspring()
        self.simulate_eating()
        self.simulate_deaths()
        self.age_population()

        self.pop_count = len(self.animals)
        self.pop_mean_age = self.get_pop_age()
        self.pop_genes = self._get_population_genetics()

        self.steps += 1
        return State(self.steps,
                     self.pop_count,
                     self.pop_mean_age,
                     self.pop_genes,
                     self.n_food)

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


def run_series(species, n_individuals=20, n_food=200, food_rate=50,
               n_iters=100, n_runs=3):
    results = []
    for i in tqdm(range(n_runs)):
        env = Environment(species, n_individuals, n_food, food_rate)
        history = env.run(n_iters=n_iters)
        df_history = process_history(history)
        df_history['run_no'] = i
        results.append(df_history)
    return pd.concat(results)


gene_effects = {
    'food': [('a', 0.05)],
    'death_multiplier': [('a', 0.9)]
}
species = Species(genome=['a', 'b', 'c', 'd'],
                  gene_length=4,
                  gene_effects=gene_effects)
df = run_series(species, n_individuals=200, n_food=500, food_rate=200, n_iters=100, n_runs=3)
df_melt = df.melt(id_vars=['step'], value_vars=['gene_a', 'gene_b', 'gene_c', 'gene_d'])
p = ggplot(aes(x='step', y='value', color='variable'), df_melt)
p = p + geom_point(alpha=0.25) + geom_smooth(method='glm') #+ facet_grid('run_no ~ .')
p.__repr__()

# df_x = df.groupby('step').mean().reset_index()
# df_x_melt = df.melt(id_vars=['step'], value_vars=['gene_a', 'gene_b', 'gene_c', 'gene_d'])


# p = ggplot(aes(x='variable', y='step', fill='value'), df_x_melt)
# p = p + geom_tile()
# p.__repr__()



# df_joy = df[['gene_a', 'gene_b', 'gene_c', 'gene_d']].copy()
# fig, axes = joypy.joyplot(df_joy)
# plt.show()
