from ..pg import Environment, Species, Animal


class TestAnimal:

    def test_calc_modifiers(self):
        """Test the _calc_modifiers method for the case where the gene effects
         are specified."""
        # First test
        test_gene_effects = {
            'food': [('a', 0.1)]
        }

        test_animal = Animal(genome=['a', 'b', 'c', 'd'],
                             genes=['a', 'b', 'c', 'd'],
                             gene_effects=test_gene_effects)


        actual_result = test_animal._calc_modifiers('food')
        expected_result = 0.1

        assert actual_result == expected_result

        # Second test
        test_gene_effects = {
            'death': [('a', 0.3), ('b', 0.1)]
        }

        test_animal = Animal(genome=['a', 'b', 'c', 'd'],
                             genes=['a', 'b', 'c', 'd'],
                             gene_effects=test_gene_effects)

        actual_result = test_animal._calc_modifiers('death')
        expected_result = 0.4

        assert actual_result == expected_result

    def test_calc_modifiers_unspecified(self):
        """Test the _calc_modifiers method for the case where the gene effects
         are not specified. The expected result in this case is 0, and a raised
         warning."""

        test_animal = Animal(genome=['a', 'b', 'c', 'd'],
                             genes=['a', 'b', 'c', 'd'])

        actual_result = test_animal._calc_modifiers('food')
        expected_result = 0.0
        assert actual_result == expected_result

    def test_validate(self):
        pass


class TestEnvironment:

    def test_get_population_genetics_zero(self):
        """
         Test the _get_population_genetics function for the case where the
         population is zero.
         """
        test_genome = ['a', 'b', 'c', 'd']
        test_gene_length = 4
        test_species = Species(test_genome, test_gene_length)
        test_env = Environment(test_species, n_individuals=0)

        actual_result = test_env._get_population_genetics()
        expected_result = {g: 0 for g in test_genome}

        assert actual_result == expected_result

    def test_get_population_genetics(self):
        """
         Test the _get_population_genetics function for the case where
         the population is non-zero.
         """
        test_genome = ['a', 'b', 'c', 'd']
        test_gene_length = 4

        class MockSpecies:
            def __init__(self):
                self.genome = test_genome
                self.gene_length = test_gene_length

            def make_animal(self, *args, **kwargs):
                return Animal(genome=self.genome, genes=['a', 'b', 'c', 'd'])

        test_env = Environment(species=MockSpecies(), n_individuals=4)
        actual_result = test_env._get_population_genetics()
        expected_result = {'a': 0.25, 'b': 0.25, 'c': 0.25, 'd': 0.25}
        assert actual_result == expected_result
