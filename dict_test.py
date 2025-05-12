from typing import Dict, List, Set
import numpy as np

class OrgNode:
    
    def __init__(self, p_id, c_id, heir_gen, birth_gen, species):
        self._parent = p_id
        self._child  = c_id
        self._gen    = heir_gen
        self._b_gen  = birth_gen
        self._species= species
    
    @property
    def c_id(self):
        return self._child
    
    @property
    def p_id(self):
        return self._parent
    
    @property
    def heirarchical_generation(self):
        self._gen

    @property
    def birth_generation(self):
        self._b_gen
        
    @property
    def species(self):
        self._species

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
            p_id = int(parent_rec['c_id'])
            c_id = int(child_rec['c_id'])
            p_spec = parent_rec['species']
            c_spec = child_rec['species']
            heir_gen  = int(child_rec['gen'])
            birth_gen  = gen


            # Create & store the new node
            node = OrgNode(p_id, c_id, heir_gen, birth_gen ,p_spec)
            self._nodes[c_id] = node

            
            # Link parent → child
            self._children.setdefault(p_id, []).append(c_id)
            
            
            # If we’ve never seen this child as anyone’s child before, it might be a founder:
            if c_id not in self._nodes:
                self._global_roots.add(c_id)


            # If we've never seen this organism's parent before then the parent is a founder
            if p_id not in self._nodes and p_id not in self._global_roots:
                self._global_roots.add(p_id)


            # If the organism is not a founder then we found its parent
            self._global_roots.discard(c_id)


            # If the species change, we mark a new taxon root
            if c_spec != p_spec:
                self._species_roots.setdefault(c_spec,set()).add(c_id)
