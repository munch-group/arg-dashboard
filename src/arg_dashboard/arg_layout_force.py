"""
ARG Layout - Force-Directed (Unconstrained)
============================================
Parents are NOT required to be between their children's x-coordinates.
Uses edge repulsion to spread the graph while maintaining structure.
"""

import numpy as np
from collections import defaultdict
import json


def compute_arg_xpos(arg_data, n_iterations=100, edge_repulsion=0.02, 
                     node_repulsion=0.02, child_attraction=0.15,
                     parent_attraction=0.05):
    """
    Compute x-positions for an ARG using force-directed layout.
    
    Parents can be positioned outside their children's x-range if 
    edge repulsion forces push them there.
    
    Parameters
    ----------
    arg_data : dict
        ARG data with 'Leaf', 'Coalescent', 'Recombination', and 'Lineage' keys
    n_iterations : int
        Number of force-directed iterations
    edge_repulsion : float
        Strength of edge-edge repulsion (spreads edges apart)
    node_repulsion : float
        Strength of node-node repulsion within layers
    child_attraction : float
        Strength of attraction toward children centroid
    parent_attraction : float
        Strength of attraction toward parents (for recombination nodes)
    
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
    
    # Use original leaf positions if available
    original_leaf_xpos = {}
    for leaf in arg_data['Leaf']:
        if 'xpos' in leaf and leaf['xpos'] is not None:
            original_leaf_xpos[leaf['nodeid']] = leaf['xpos']
    
    if original_leaf_xpos and len(original_leaf_xpos) == n_leaves:
        for node in leaves:
            x_pos[node] = original_leaf_xpos[node]
    else:
        leaves_sorted = sorted(leaves)
        for i, node in enumerate(leaves_sorted):
            if n_leaves > 1:
                x_pos[node] = x_min + i * (x_max - x_min) / (n_leaves - 1)
            else:
                x_pos[node] = (x_min + x_max) / 2
    
    # Initialize internal nodes at children centroid
    def get_child_centroid(node):
        if children_map[node]:
            child_xs = [x_pos[c] for c in children_map[node] if c in x_pos]
            if child_xs:
                return np.mean(child_xs)
        return (x_min + x_max) / 2
    
    for h in sorted_heights:
        if h == 0:
            continue
        for node in layers[h]:
            x_pos[node] = get_child_centroid(node)
    
    # Force-directed iterations
    edge_list = list(edges)
    
    for iteration in range(n_iterations):
        # Learning rate with decay
        lr = 0.5 * (1 - 0.5 * iteration / n_iterations)
        
        # Initialize forces for non-leaf nodes
        forces = {node: 0.0 for node in all_nodes if heights[node] > 0}
        
        # 1. Edge-edge repulsion
        for i, (p1, c1) in enumerate(edge_list):
            for j, (p2, c2) in enumerate(edge_list[i+1:], i+1):
                # Skip if edges share a node
                if p1 == p2 or c1 == c2 or p1 == c2 or p2 == c1:
                    continue
                
                # Check if edges overlap in y (height) range
                y1_min, y1_max = min(heights[p1], heights[c1]), max(heights[p1], heights[c1])
                y2_min, y2_max = min(heights[p2], heights[c2]), max(heights[p2], heights[c2])
                
                # Only apply force if edges overlap vertically
                if y1_max <= y2_min or y2_max <= y1_min:
                    continue
                
                # Compute edge midpoints
                mid1_x = (x_pos[p1] + x_pos[c1]) / 2
                mid2_x = (x_pos[p2] + x_pos[c2]) / 2
                
                # Horizontal distance
                dx = mid1_x - mid2_x
                dist = abs(dx) + 0.05
                
                # Repulsive force
                force = edge_repulsion / (dist ** 2)
                force = min(force, 0.1)  # Cap force
                
                # Direction
                sign = 1 if dx > 0 else -1
                if dx == 0:
                    sign = 1 if np.random.random() > 0.5 else -1
                
                # Apply force to endpoints (except leaves)
                if heights[p1] > 0:
                    forces[p1] += force * sign * 0.5
                if heights[c1] > 0:
                    forces[c1] += force * sign * 0.5
                if heights[p2] > 0:
                    forces[p2] -= force * sign * 0.5
                if heights[c2] > 0:
                    forces[c2] -= force * sign * 0.5
        
        # 2. Node-node repulsion within layers
        for h in sorted_heights:
            if h == 0:
                continue
            layer_nodes = layers[h]
            for i, n1 in enumerate(layer_nodes):
                for n2 in layer_nodes[i+1:]:
                    dx = x_pos[n1] - x_pos[n2]
                    dist = abs(dx) + 0.05
                    force = node_repulsion / (dist ** 2)
                    force = min(force, 0.05)
                    sign = 1 if dx > 0 else -1
                    forces[n1] += force * sign
                    forces[n2] -= force * sign
        
        # 3. Attraction to children (maintains some structure)
        for node in all_nodes:
            if heights[node] > 0 and children_map[node]:
                child_xs = [x_pos[c] for c in children_map[node]]
                centroid = np.mean(child_xs)
                forces[node] += child_attraction * (centroid - x_pos[node])
        
        # 4. Attraction to parents (for recombination nodes)
        for node in all_nodes:
            if heights[node] > 0 and parents_map[node]:
                parent_xs = [x_pos[p] for p in parents_map[node]]
                centroid = np.mean(parent_xs)
                forces[node] += parent_attraction * (centroid - x_pos[node])
        
        # Apply forces
        for node, force in forces.items():
            x_pos[node] += lr * force
            # Hard boundaries
            x_pos[node] = np.clip(x_pos[node], x_min, x_max)
    
    # Update the JSON data
    for leaf in arg_data['Leaf']:
        leaf['xpos'] = float(x_pos[leaf['nodeid']])
    
    for coal in arg_data['Coalescent']:
        coal['xpos'] = float(x_pos[coal['nodeid']])
    
    for recomb in arg_data['Recombination']:
        recomb['xpos'] = float(x_pos[recomb['nodeid']])
    
    return arg_data


def count_crossings(arg_data):
    """Count edge crossings in the layout."""
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
    
    edges = []
    for lineage in arg_data['Lineage']:
        if lineage['up'] is not None:
            edges.append((lineage['up'], lineage['down']))
    
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


def check_parents_outside_children(arg_data):
    """Find parents positioned outside their children's x-range."""
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
    
    children_map = defaultdict(list)
    for lineage in arg_data['Lineage']:
        if lineage['up'] is not None:
            children_map[lineage['up']].append(lineage['down'])
    
    outside = []
    for node, children in children_map.items():
        if children:
            child_xs = [x_pos[c] for c in children]
            lo, hi = min(child_xs), max(child_xs)
            node_x = x_pos[node]
            if node_x < lo - 0.001 or node_x > hi + 0.001:
                outside.append({
                    'node': node,
                    'x': node_x,
                    'children_range': (lo, hi),
                    'offset': min(node_x - lo, node_x - hi, key=abs)
                })
    
    return outside


# Demo
if __name__ == '__main__':
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
    
    import copy
    
    print("Original layout:")
    print(f"  Crossings: {count_crossings(arg_json)}")
    print(f"  Leaves: {[(l['nodeid'], l['xpos']) for l in arg_json['Leaf']]}")
    print(f"  Coalescent: {[(c['nodeid'], round(c['xpos'], 3)) for c in arg_json['Coalescent']]}")
    print(f"  Recombination: {[(r['nodeid'], round(r['xpos'], 3)) for r in arg_json['Recombination']]}")
    outside = check_parents_outside_children(arg_json)
    print(f"  Parents outside children: {len(outside)}")
    
    # Compute new layout
    arg_updated = compute_arg_xpos(copy.deepcopy(arg_json))
    
    print("\nForce-directed layout:")
    print(f"  Crossings: {count_crossings(arg_updated)}")
    print(f"  Leaves: {[(l['nodeid'], l['xpos']) for l in arg_updated['Leaf']]}")
    print(f"  Coalescent: {[(c['nodeid'], round(c['xpos'], 3)) for c in arg_updated['Coalescent']]}")
    print(f"  Recombination: {[(r['nodeid'], round(r['xpos'], 3)) for r in arg_updated['Recombination']]}")
    
    outside = check_parents_outside_children(arg_updated)
    print(f"  Parents outside children: {len(outside)}")
    for o in outside:
        print(f"    Node {o['node']} at {o['x']:.3f}, children in [{o['children_range'][0]:.3f}, {o['children_range'][1]:.3f}]")
