import msprime as ms
import tsinfer as tsi
import tsdate


def msprime_simulate_variants(params):
    """
    copyrigth pierre
    Population simulation with msprime for SMC++ (msprime 1.x).
    In this case, mutations are placed at discrete, integer coordinates => Msprime 1.01 is
    therefore needed (update of Msprime and tskit).
    Parameter
    ---------
    model: function
        (constant, sudden declin, sudden growth, etc.)
    params: dictionary
        - sample_size: the number of sampled monoploid genomes
        - Ne: the effective (diploid) population size
        - ro: the rate of recombinaison per base per generation
        - mu: the rate of infinite sites mutations per unit of sequence length per generation
        - length: the length of the simulated region in bases
    tau: the lenght of time ago at which the event (decline, growth) occured
    kappa: the growth or decline force
    debug: Boolean
        1: print msprime debugger, 0: nothing
    Some notes about the simulation of ancestry with the method sim_ancestry() of Msprime 1.x
      - samples
        The number of individual instead of the number of monoploid genomes (msprime 0.x)
      - ploidy
        Sets the default number of sample nodes (i.e. monoploid genomes) per individual
        Ploidy set to 2 means time to common ancestor in a population of size N is 2N
        generations (which is the same as msprime 0.x)
      - discrete_genome
        If True that means mutations are placed at discrete, integer coordinates
        If False that means mutations are placed at continuous, float coordinates (ms 0.x)
      If samples is set to 10 and ploidy to 1 there are N=10 genomes sampled
      If samples is set to 10 and ploidy to 2 there are 2N=20 genomes sampled
    Return
    ------
    sfs: list
        Site frequency Spectrum (sfs) - allele mutation frequency
    variants: list
        List of position and genotypes for each variant with 0 the ancestral state and 1 the
        alternative one.
    """
    # Set up the population model
    demography = ms.Demography()
    # Population actuelle au temps 0
    demography.add_population(initial_size=params['Ne'], growth_rate=0.)
    # Ancestral population
    demography.add_population_parameters_change(
        time=params['Tau'], population=0, initial_size=params['Ne'] * params['Kappa'],
        growth_rate=0.)
    ts = ms.sim_ancestry(
        samples=int(params['sample_size'] / 2), demography=demography, ploidy=2,
        sequence_length=params['length'],
        discrete_genome=False,
        recombination_rate=params['ro']
    )
    mutation_model = ms.BinaryMutationModel(state_independent=False)
    # Genetic variation of the data with mutation
    ts = ms.sim_mutations(tree_sequence=ts, rate=params['mu'], discrete_genome=False,
                          model=mutation_model)
    # return ts
    return ts, ts.edges(), list(ts.breakpoints()), \
           [variant for variant in ts.variants()]


def test_tsinfer(variants, sequence_length, simplify=False):
    with tsi.SampleData(sequence_length=sequence_length, num_flush_threads=2) as sample_data:
        for var in variants:
            sample_data.add_site(var.site.position, var.genotypes, var.alleles)
    ts = tsi.infer(sample_data)  # .simplify()
    if simplify:
        ts = ts.simplify()
    # ts = tsdate.date(ts, Ne=1, mutation_rate=8e-4)
    return [ts, ts.edges(), list(ts.breakpoints()), \
            [variant for variant in ts.variants()]]
