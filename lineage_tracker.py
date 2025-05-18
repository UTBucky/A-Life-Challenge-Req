from typing import Dict, List, Set
import numpy as np

class OrgNode:
    
    def __init__(self, p_id, c_id, hier_gen, birth_gen, species):
        self._parent = p_id
        self._child  = c_id
        self._gen    = hier_gen
        self._b_gen  = birth_gen
        self._species= species
    
    @property
    def c_id(self):
        return self._child
    
    @property
    def p_id(self):
        return self._parent
    
    @property
    def hierarchical_generation(self):
        return self._gen

    @property
    def birth_generation(self):
        return self._b_gen
        
    @property
    def species(self):
        return self._species

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

    def track_lineage(
        self, 
        p_array: np.ndarray, 
        c_array: np.ndarray, 
        gen: int
    ) -> None:
        """
        Tracks lineage relationships between parent and child organisms, 
        establishing hierarchical and species-based genealogies.
        - Takes np.ndarry's of parents and cildren
        - Takes an int representing the generation we are in 
        Returns:
        --------
        None
            This method updates internal data structures of the object, 
            specifically:
            - `_nodes`: A dictionary mapping child IDs to `OrgNode` objects.
            - `_children`: A dictionary mapping parent IDs to lists of child IDs.
            - `_species_roots`: A dictionary mapping species names to their root IDs.
            - `_global_roots`: A set of global root IDs for founders.
        
        Side Effects:
        -------------
        - If a parent or species is not seen before, it is registered in the
        appropriate root structures (`_species_roots`, `_global_roots`).
        - Creates and stores new `OrgNode` objects for child organisms.
        - Links parent to child in the `_children` structure.
        """
        # Initialize a set of previously seen nodes
        seen_before = set(self._nodes)
        
        # Iterate through parent and child records simultaneously
        for parent_rec, child_rec in zip(p_array, c_array):
            p_id = int(parent_rec['c_id'])
            c_id = int(child_rec['c_id'])
            p_spec = parent_rec['species']
            c_spec = child_rec['species']
            hier_gen  = int(child_rec['generation'])
            birth_gen  = gen


            # first time we see species p_spec
            if p_id not in seen_before and p_spec not in self._species_roots:
                self._species_roots.setdefault(p_spec, set()).add(p_id)


            # If we've never seen this organism's parent before then the parent is a founder
            if p_id not in seen_before and p_id not in self._global_roots:
                self._global_roots.add(p_id)


            # Create & store the new node
            # Link parent → child
            node = OrgNode(p_id, c_id, hier_gen, birth_gen , c_spec)
            self._nodes[c_id] = node
            self._children.setdefault(p_id, []).append(c_id)


            # If the species change, we mark a new taxon root
            if c_spec != p_spec:
                self._species_roots.setdefault(c_spec,set()).add(c_id)

    def _dfs_newick(self, node_id: int) -> str:
        """
        Recursive DFS to build Newick for the subtree at node_id,
        labeling nodes by species instead of numeric ID.
        """
        children = self._children.get(node_id, [])
        label = self._nodes[node_id].species
        if not children:
            return label

        parts: List[str] = []
        for child in sorted(children, key=lambda cid: self._nodes[cid].birth_generation):
            parts.append(self._dfs_newick(child))
        return '(' + ','.join(parts) + ')' + label

    def full_forest_newick(self) -> str:
        """
        Build a full Newick string including both global founders and speciation roots,
        with labels by species.
        """
        # Combine global roots and species roots as entry points
        root_ids: Set[int] = set(self._global_roots)
        for roots in self._species_roots.values():
            root_ids.update(roots)

        # Filter to only those present in nodes
        valid_roots = [rid for rid in root_ids if rid in self._nodes]
        if not valid_roots:
            return ';'

        # Order by birth generation for deterministic output
        ordered = sorted(valid_roots, key=lambda cid: self._nodes[cid].birth_generation)
        parts: List[str] = []
        for rid in ordered:
            parts.append(self._dfs_newick(rid))

        return '(' + ','.join(parts) + ');'

