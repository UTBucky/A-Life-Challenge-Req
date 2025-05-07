import json
import genome
import numpy as np


def load_genes_from_file(filename="gene_settings.json") -> dict:
    """
    Loads gene‐domain templates from a JSON file and
    returns a dict mapping category → Gene instance
    (initialized at the midpoint of each allowable range).
    """
    with open(filename, 'r') as f:
        data = json.load(f)

    mutation_rate = data["mutation_rate"]
    raw_pool = data["gene_pool"]
    gene_pool = {}

    # Morphological: 5 floats [size, camouflage, defense, attack, vision]
    if "morphological" in raw_pool:
        vals = raw_pool["morphological"]["values"]
        mins = np.array(vals["min"], dtype=float)
        maxs = np.array(vals["max"], dtype=float)
        midpoints = (mins + maxs) / 2.0
        gene_pool["morphological"] = genome.MorphologicalGenes(
            mutation_rate, midpoints
        )

    # Metabolic: 2 floats + 1 categorical
    if "metabolic" in raw_pool:
        num = raw_pool["metabolic"]["numeric"]
        mins = np.array(num["min"], dtype=float)
        maxs = np.array(num["max"], dtype=float)
        mid_numeric = (mins + maxs) / 2.0
        diet_opts = raw_pool["metabolic"]["diet_type"]
        gene_pool["metabolic"] = genome.MetabolicGenes(
            mutation_rate, mid_numeric, diet_opts[0]
        )

    # Reproduction: fertility_rate, offspring_count, reproduction_type
    if "reproduction" in raw_pool:
        frange = raw_pool["reproduction"]["fertility_rate"]
        orangec = raw_pool["reproduction"]["offspring_count"]
        rtypes = raw_pool["reproduction"]["reproduction_type"]
        mid_fr = (frange["min"] + frange["max"]) / 2.0
        mid_oc = int((orangec["min"] + orangec["max"]) / 2)
        gene_pool["reproduction"] = genome.ReproductionGenes(
            mutation_rate, mid_fr, mid_oc, rtypes[0]
        )

    # Behavioral: two booleans [pack_behavior, symbiotic]
    if "behavioral" in raw_pool:
        pack_opts = raw_pool["behavioral"]["pack_behavior"]
        sym_opts = raw_pool["behavioral"]["symbiotic"]
        bools_arr = np.array([pack_opts[0], sym_opts[0]], dtype=bool)
        gene_pool["behavioral"] = genome.BehavioralGenes(
            mutation_rate, bools_arr
        )

    # Locomotion: [swim, walk, fly] + speed
    if "locomotion" in raw_pool:
        swim_opts = raw_pool["locomotion"]["swim"]
        walk_opts = raw_pool["locomotion"]["walk"]
        fly_opts = raw_pool["locomotion"]["fly"]
        speed_range = raw_pool["locomotion"]["speed"]
        bools_arr = np.array(
            [swim_opts[0], walk_opts[0], fly_opts[0]],
            dtype=bool
        )
        mid_speed = (speed_range["min"] + speed_range["max"]) / 2.0
        gene_pool["locomotion"] = genome.LocomotionGenes(
            mutation_rate, bools_arr, mid_speed
        )

    return gene_pool
