from typing import Dict, List, Set
import numpy as np

class OrgNode:
    
    def __init__(self, p_id, c_id, gen, species):
        self._parent = p_id
        self._child  = c_id
        self._gen    = gen
        self._species= species

class LineageTracker:
    
    def __init__(self):
        # 1) Every organism ever seen:
        #    c_id → OrgNode
        self._nodes: Dict[int, OrgNode] = {}

        # 2) Parent → [child₁, child₂, …]
        self._children: Dict[int, List[int]] = {}

        # 3) Species → { root₁, root₂, … }
        #    each root is a c_id where a new taxon began
        self._species_roots: Dict[str, Set[int]] = {}

        # 4) Any c_id with no known parent: “founders”
        self._global_roots: Set[int] = set()

    def track_lineage(self, 
        p_array: np.ndarray, 
        c_array: np.ndarray, 
        gen: int):
        for parent_rec, child_rec in zip(p_array, c_array):
            p_id, c_id = parent_rec['c_id'], child_rec['c_id']
            p_species, c_species = parent_rec['species'], child_rec['species']
            # 1) Create child node
            child_node = OrgNode(p_id, c_id, child_rec['gen'], c_species)
            self._nodes[c_id] = child_node
            # 2) Register child under its parent
            self._children.setdefault(p_id, []).append(c_id)