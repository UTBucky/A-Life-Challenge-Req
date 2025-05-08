import json
import pprint

def load_genes_from_file(filename="gene_settings.json") -> dict:
    """
    Loads gene ranges from a JSON file and returns a dictionary
    mapping gene names to (min, max) tuples for continuous values
    or lists of options for categorical values.

    Example Output:
    {
        'size': (0.0, 1.0),
        'camouflage': (0.0, 1.0),
        'defense': (0.0, 1.0),
        'attack': (0.0, 1.0),
        'vision': (0.0, 1.0),
        'metabolism_rate': (0.0, 1.0),
        'nutrient_efficiency': (0.0, 1.0),
        'diet_type': ['Herb', 'Omni', 'Carn', 'Photo', 'Parasite'],
        ...
    }
    """
    with open(filename, 'r') as f:
        data = json.load(f)

    gene_pool = data["gene_pool"]
    gene_dict = {}

    # Morphological genes
    if "morphological" in gene_pool:
        values = gene_pool["morphological"]["values"]
        min_vals = values["min"]
        max_vals = values["max"]
        keys = ["size", "camouflage", "defense", "attack", "vision"]
        for i, key in enumerate(keys):
            gene_dict[key] = (min_vals[i], max_vals[i])

    # Metabolic genes
    if "metabolic" in gene_pool:
        numeric = gene_pool["metabolic"]["numeric"]
        gene_dict["metabolism_rate"] = (numeric["min"][0], numeric["max"][0])
        gene_dict["nutrient_efficiency"] = (numeric["min"][1], numeric["max"][1])
        gene_dict["diet_type"] = gene_pool["metabolic"]["diet_type"]

    # Reproduction genes
    if "reproduction" in gene_pool:
        gene_dict["fertility_rate"] = (
            gene_pool["reproduction"]["fertility_rate"]["min"],
            gene_pool["reproduction"]["fertility_rate"]["max"]
        )
        gene_dict["offspring_count"] = (
            gene_pool["reproduction"]["offspring_count"]["min"],
            gene_pool["reproduction"]["offspring_count"]["max"]
        )
        gene_dict["reproduction_type"] = gene_pool["reproduction"]["reproduction_type"]

    # Behavioral genes
    if "behavioral" in gene_pool:
        gene_dict["pack_behavior"] = gene_pool["behavioral"]["pack_behavior"]
        gene_dict["symbiotic"] = gene_pool["behavioral"]["symbiotic"]

    # Locomotion genes
    if "locomotion" in gene_pool:
        gene_dict["swim"] = gene_pool["locomotion"]["swim"]
        gene_dict["walk"] = gene_pool["locomotion"]["walk"]
        gene_dict["fly"] = gene_pool["locomotion"]["fly"]
        gene_dict["speed"] = (
            gene_pool["locomotion"]["speed"]["min"],
            gene_pool["locomotion"]["speed"]["max"]
        )

    return gene_dict


if __name__ == "__main__":
    print(load_genes_from_file("gene_settings.json"))