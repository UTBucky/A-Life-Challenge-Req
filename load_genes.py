import json
import genome


def load_genes_from_file(filename="gene_settings.json") -> dict:
    """
    Loads gene settings from a json file,
    returning a dictionary of gene objects

    :param filename: A string
    """

    genes_file = open(filename)
    genes_data = json.load(genes_file)
    gene_pool = {}

    for gene in genes_data:

        if gene == "energy_prod":
            options = genes_data[gene]["options"]
            values = genes_data[gene]["values"]
            energy_gene = genome.EnergyGene(options[0],
                                            values[1],
                                            values[0],
                                            values[2],
                                            options)
            gene_pool[gene] = energy_gene

        elif gene == "move_aff":
            options = genes_data[gene]
            gene_pool[gene] = genome.MoveGene(options[0], options)

        else:
            gene_sett = genes_data[gene]
            gene_pool[gene] = genome.Gene(type,
                                          gene_sett[1],
                                          gene_sett[0],
                                          gene_sett[2])

    genes_file.close()
    return gene_pool
