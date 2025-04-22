import pytest
import json
from organism import Organism
from genome import *

genes_file = open("gene_settings.json")
genes_data = json.load(genes_file)
gene_pool = {}
org_genes = {}
org_genome = Genome(0.4, org_genes)

for gene in genes_data:

    if gene == "energy_prod":
        for type in genes_data[gene]:
            type_sett = genes_data[gene][type]
            gene_pool[gene] = EnergyGene(type, type_sett[1],
                                         type_sett[0], type_sett[2])
        org_genes[gene] = gene_pool[gene]

    elif gene == "move_affordance":
        for type in genes_data[gene]:
            gene_pool[gene] = MoveGene(type)
        org_genes[gene] = gene_pool[gene]

    else:
        gene_sett = genes_data[gene]
        gene_pool[gene] = Gene(type, gene_sett[1], gene_sett[0],  gene_sett[2])
        org_genes[gene] = gene_pool[gene]


genes_file.close()


def test_genes_loaded_from_file():
    org = Organism("ORG1", org_genome, 0, (0, 0))

    org1_genome = org.get_genome().get_genes()

    for gene in org1_genome:
        assert org1_genome[gene].get_val() == gene_pool[gene].get_val()

def test_organism_mutated_reproduction():
    org_genome = Genome(1.0, org_genes)
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