"""
ARG Layout for JSON Format
===========================
Computes x-coordinates for ARG nodes defined in the given JSON structure.
"""

import numpy as np
from collections import defaultdict
import json


def compute_arg_xpos(arg_data, n_iterations=50):
    """
    Compute x-positions for an ARG defined in JSON format.
    
    Uses a barycenter heuristic: each internal node is positioned at the 
    centroid of its leaf descendants. For recombination nodes, considers
    both children and parents.
    
    Parameters
    ----------
    arg_data : dict
        ARG data with 'Leaf', 'Coalescent', 'Recombination', and 'Lineage' keys
    n_iterations : int
        Number of refinement iterations
    
    Returns
    -------
    dict
        The input arg_data with updated 'xpos' values for all nodes
    """
    # Extract all nodes and heights
    all_nodes = []
    heights = {}
    
    for leaf in arg_data['Leaf']:
        nid = leaf['nodeid']
        all_nodes.append(nid)
        heights[nid] = leaf['height']
    
    for coal in arg_data['Coalescent']:
        nid = coal['nodeid']
        all_nodes.append(nid)
        heights[nid] = coal['height']
    
    for recomb in arg_data['Recombination']:
        nid = recomb['nodeid']
        all_nodes.append(nid)
        heights[nid] = recomb['height']
    
    # Build edges from Lineage data: (parent, child)
    edges = []
    for lineage in arg_data['Lineage']:
        down = lineage['down']
        up = lineage['up']
        if up is not None:
            edges.append((up, down))
    
    # Build adjacency lists
    children_map = defaultdict(list)
    parents_map = defaultdict(list)
    for parent, child in edges:
        children_map[parent].append(child)
        parents_map[child].append(parent)
    
    # Identify leaves
    leaves = [n for n in all_nodes if heights[n] == 0]
    n_leaves = len(leaves)
    x_min, x_max = 0.0, 1.0
    
    # Group nodes by height
    layers = defaultdict(list)
    for node in all_nodes:
        layers[heights[node]].append(node)
    sorted_heights = sorted(layers.keys())
    
    x_pos = {}
    
    # Use original leaf positions if available, otherwise space evenly
    original_leaf_xpos = {}
    for leaf in arg_data['Leaf']:
        if 'xpos' in leaf and leaf['xpos'] is not None:
            original_leaf_xpos[leaf['nodeid']] = leaf['xpos']
    
    if original_leaf_xpos and len(original_leaf_xpos) == n_leaves:
        # Use original positions
        for node in leaves:
            x_pos[node] = original_leaf_xpos[node]
    else:
        # Space evenly (sorted by node id)
        leaves_sorted = sorted(leaves)
        for i, node in enumerate(leaves_sorted):
            if n_leaves > 1:
                x_pos[node] = x_min + i * (x_max - x_min) / (n_leaves - 1)
            else:
                x_pos[node] = (x_min + x_max) / 2
    
    # Compute leaf descendants for each node (for barycenter)
    def get_leaf_descendants(node, memo={}):
        if node in memo:
            return memo[node]
        if not children_map[node]:  # Is a leaf
            memo[node] = {node}
            return memo[node]
        result = set()
        for child in children_map[node]:
            result |= get_leaf_descendants(child, memo)
        memo[node] = result
        return result
    
    # Clear memo for fresh computation
    leaf_desc_memo = {}
    for node in all_nodes:
        get_leaf_descendants(node, leaf_desc_memo)
    
    # Position each internal node at centroid of its leaf descendants
    for h in sorted_heights:
        if h == 0:
            continue
        for node in layers[h]:
            descendants = leaf_desc_memo.get(node, set())
            if descendants:
                x_pos[node] = np.mean([x_pos[d] for d in descendants])
            else:
                x_pos[node] = (x_min + x_max) / 2
    
    # Iterative refinement: consider both children and parents
    for _ in range(n_iterations):
        for h in sorted_heights:
            if h == 0:
                continue
            for node in layers[h]:
                pulls = []
                weights = []
                
                # Pull from children (stronger)
                for child in children_map[node]:
                    pulls.append(x_pos[child])
                    weights.append(2.0)
                
                # Pull from parents (weaker, for recombination nodes)
                for parent in parents_map[node]:
                    pulls.append(x_pos[parent])
                    weights.append(1.0)
                
                if pulls:
                    x_pos[node] = np.average(pulls, weights=weights)
    
    # Ensure positions are within bounds
    for node in all_nodes:
        x_pos[node] = np.clip(x_pos[node], x_min, x_max)
    
    # Update the JSON data with computed positions
    for leaf in arg_data['Leaf']:
        leaf['xpos'] = float(x_pos[leaf['nodeid']])
    
    for coal in arg_data['Coalescent']:
        coal['xpos'] = float(x_pos[coal['nodeid']])
    
    for recomb in arg_data['Recombination']:
        recomb['xpos'] = float(x_pos[recomb['nodeid']])
    
    return arg_data


def count_crossings(arg_data):
    """Count edge crossings in the layout."""
    # Extract heights and positions
    heights = {}
    x_pos = {}
    
    for leaf in arg_data['Leaf']:
        nid = leaf['nodeid']
        heights[nid] = leaf['height']
        x_pos[nid] = leaf['xpos']
    
    for coal in arg_data['Coalescent']:
        nid = coal['nodeid']
        heights[nid] = coal['height']
        x_pos[nid] = coal['xpos']
    
    for recomb in arg_data['Recombination']:
        nid = recomb['nodeid']
        heights[nid] = recomb['height']
        x_pos[nid] = recomb['xpos']
    
    # Build edges
    edges = []
    for lineage in arg_data['Lineage']:
        if lineage['up'] is not None:
            edges.append((lineage['up'], lineage['down']))
    
    # Count crossings
    crossings = 0
    for i, (p1, c1) in enumerate(edges):
        for p2, c2 in edges[i+1:]:
            if p1 == p2 or c1 == c2 or p1 == c2 or p2 == c1:
                continue
            h1_top, h1_bot = heights[p1], heights[c1]
            h2_top, h2_bot = heights[p2], heights[c2]
            if max(h1_bot, h2_bot) >= min(h1_top, h2_top):
                continue
            if (x_pos[p1] - x_pos[p2]) * (x_pos[c1] - x_pos[c2]) < 0:
                crossings += 1
    
    return crossings


# Demo with the provided JSON
if __name__ == '__main__':
    # The provided ARG data
    arg_json = {
        "Coalescent": [
            {"nodeid": 5, "height": 0.13669486930750932, "children": [2, 1], "parent": 5, "xpos": 0.125},
            {"nodeid": 7, "height": 0.3897837685725356, "children": [7, 4], "parent": 8, "xpos": 0.5125},
            {"nodeid": 8, "height": 0.5503568938072688, "children": [8, 3], "parent": 9, "xpos": 0.60625},
            {"nodeid": 10, "height": 0.7368215587091936, "children": [10, 9], "parent": 12, "xpos": 0.578125},
            {"nodeid": 11, "height": 0.9908807698242248, "children": [6, 11], "parent": 13, "xpos": 0.3125},
            {"nodeid": 12, "height": 1.0, "children": [13, 12], "parent": 14, "xpos": 0.4453125}
        ],
        "Recombination": [
            {"nodeid": 6, "height": 0.331775128015738, "child": 5, "left_parent": 6, "right_parent": 7, "recomb_point": 0.9254180219413557, "xpos": 0.125},
            {"nodeid": 9, "height": 0.592546189634543, "child": 0, "left_parent": 10, "right_parent": 11, "recomb_point": 0.1973898113750865, "xpos": 0.5}
        ],
        "Leaf": [
            {"nodeid": 0, "height": 0.0, "intervals": [[0, 1]], "parent": 0, "xpos": 0.5},
            {"nodeid": 1, "height": 0.0, "intervals": [[0, 1]], "parent": 1, "xpos": 0.0},
            {"nodeid": 2, "height": 0.0, "intervals": [[0, 1]], "parent": 2, "xpos": 0.25},
            {"nodeid": 3, "height": 0.0, "intervals": [[0, 1]], "parent": 3, "xpos": 1.0},
            {"nodeid": 4, "height": 0.0, "intervals": [[0, 1]], "parent": 4, "xpos": 0.75}
        ],
        "Lineage": [
            {"lineageid": 0, "down": 0, "up": 9, "intervals": [[0, 1]]},
            {"lineageid": 1, "down": 1, "up": 5, "intervals": [[0, 1]]},
            {"lineageid": 2, "down": 2, "up": 5, "intervals": [[0, 1]]},
            {"lineageid": 3, "down": 3, "up": 8, "intervals": [[0, 1]]},
            {"lineageid": 4, "down": 4, "up": 7, "intervals": [[0, 1]]},
            {"lineageid": 5, "down": 5, "up": 6, "intervals": [[0, 1]]},
            {"lineageid": 6, "down": 6, "up": 11, "intervals": [[0, 0.9254180219413557]]},
            {"lineageid": 7, "down": 6, "up": 7, "intervals": [[0.9254180219413557, 1]]},
            {"lineageid": 8, "down": 7, "up": 8, "intervals": [[0, 1]]},
            {"lineageid": 9, "down": 8, "up": 10, "intervals": [[0, 1]]},
            {"lineageid": 10, "down": 9, "up": 10, "intervals": [[0, 0.1973898113750865]]},
            {"lineageid": 11, "down": 9, "up": 11, "intervals": [[0, 1]]},
            {"lineageid": 12, "down": 10, "up": 12, "intervals": [[0, 1]]},
            {"lineageid": 13, "down": 11, "up": 12, "intervals": [[0, 1]]},
            {"lineageid": 14, "down": 12, "up": None, "intervals": [[0, 1]]}
        ]
    }
    
    print("Original positions:")
    print("  Leaves:", {leaf['nodeid']: leaf['xpos'] for leaf in arg_json['Leaf']})
    print("  Coalescent:", {c['nodeid']: c['xpos'] for c in arg_json['Coalescent']})
    print("  Recombination:", {r['nodeid']: r['xpos'] for r in arg_json['Recombination']})
    print(f"  Crossings: {count_crossings(arg_json)}")
    
    # Compute new layout
    import copy
    arg_updated = compute_arg_xpos(copy.deepcopy(arg_json))
    
    print("\nComputed positions:")
    print("  Leaves:", {leaf['nodeid']: round(leaf['xpos'], 3) for leaf in arg_updated['Leaf']})
    print("  Coalescent:", {c['nodeid']: round(c['xpos'], 3) for c in arg_updated['Coalescent']})
    print("  Recombination:", {r['nodeid']: round(r['xpos'], 3) for r in arg_updated['Recombination']})
    print(f"  Crossings: {count_crossings(arg_updated)}")
    
    # Print full updated JSON
    print("\nFull updated JSON:")
    print(json.dumps(arg_updated, indent=2))
