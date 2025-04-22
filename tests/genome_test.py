from organism import Organism
import genome
import load_genes

gene_pool = load_genes.load_genes_from_file()
org_genome = genome.Genome(0.4, gene_pool)


def test_genes_loaded_from_file():
    org = Organism("ORG1", org_genome, 0, (0, 0))

    org1_genome = org.get_genome().get_genes()

    for gene in org1_genome:
        assert org1_genome[gene].get_val() == gene_pool[gene].get_val()


def test_organism_mutated_reproduction():
    org_genome = genome.Genome(1.0, gene_pool)
    org = Organism("ORG1", org_genome, 0, (0, 0))
    child = org.reproduce((1, 1))

    parent_genome = org_genome.get_genes()
    child_genome = child.get_genome().get_genes()

    mutation_detected = False

    for gene in parent_genome:
        if parent_genome[gene].get_val() != child_genome[gene].get_val():
            mutation_detected = True

        elif parent_genome[gene].get_name != child_genome[gene].get_name():
            mutation_detected = True

    assert mutation_detected
