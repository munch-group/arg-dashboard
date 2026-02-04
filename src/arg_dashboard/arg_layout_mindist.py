"""
ARG Layout - Minimum Distance Constrained
==========================================
Parents stay within children's x-range UNLESS moving outside is needed
to maintain minimum Euclidean distance (default 0.5) between nodes.
"""

import numpy as np
from collections import defaultdict


def compute_arg_xpos(arg_data, min_distance=None, n_iterations=300):
    """
    Compute x-positions for an ARG with minimum distance constraint.
    
    Parents stay within children's x-range unless they need to move
    outside to maintain minimum Euclidean distance from other nodes.
    
    Parameters
    ----------
    arg_data : dict
        ARG data with 'Leaf', 'Coalescent', 'Recombination', and 'Lineage' keys
    min_distance : float, optional
        Minimum allowed Euclidean distance between any two nodes.
        Default: 1/(2*n_samples) where n_samples is the number of leaves.
    n_iterations : int
        Number of iterations for constraint satisfaction
    
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
    
    # Build edges from Lineage data
    edges = []
    for lineage in arg_data['Lineage']:
        down = lineage['down']
        up = lineage['up']
        if up is not None:
            edges.append((up, down))
    
    # Build adjacency lists
    children_map = defaultdict(list)
    for parent, child in edges:
        children_map[parent].append(child)
    
    # Identify leaves
    leaves = [n for n in all_nodes if heights[n] == 0]
    n_leaves = len(leaves)
    
    # x-range is [0, 1]
    x_min, x_max = 0.0, 1.0
    
    # Default min_distance = 1/(2*n_samples)
    if min_distance is None:
        min_distance = 1.0 / (2 * n_leaves)
    
    # Group nodes by height
    layers = defaultdict(list)
    for node in all_nodes:
        layers[heights[node]].append(node)
    sorted_heights = sorted(layers.keys())
    
    x_pos = {}
    
    # Use original leaf positions if available (assuming they're in [0,1])
    original_leaf_xpos = {}
    for leaf in arg_data['Leaf']:
        if 'xpos' in leaf and leaf['xpos'] is not None:
            original_leaf_xpos[leaf['nodeid']] = leaf['xpos']
    
    if original_leaf_xpos and len(original_leaf_xpos) == n_leaves:
        for node in leaves:
            x_pos[node] = original_leaf_xpos[node]
    else:
        # Default: space leaves evenly from 0 to 1
        leaves_sorted = sorted(leaves)
        for i, node in enumerate(leaves_sorted):
            if n_leaves > 1:
                x_pos[node] = i / (n_leaves - 1)
            else:
                x_pos[node] = 0.5
    
    # Initialize internal nodes at children centroid
    def get_child_centroid(node):
        if children_map[node]:
            child_xs = [x_pos[c] for c in children_map[node] if c in x_pos]
            if child_xs:
                return np.mean(child_xs)
        return (x_min + x_max) / 2
    
    def get_child_range(node):
        """Get the x-range of immediate children."""
        if children_map[node]:
            child_xs = [x_pos[c] for c in children_map[node] if c in x_pos]
            if child_xs:
                return min(child_xs), max(child_xs)
        return x_min, x_max
    
    for h in sorted_heights:
        if h == 0:
            continue
        for node in layers[h]:
            x_pos[node] = get_child_centroid(node)
    
    def euclidean_dist(n1, n2):
        dx = x_pos[n1] - x_pos[n2]
        dy = heights[n1] - heights[n2]
        return np.sqrt(dx**2 + dy**2)
    
    # Track which nodes are "released" to go outside children range
    released = set()
    
    # Iterative constraint satisfaction
    for iteration in range(n_iterations):
        lr = 0.4 * (1 - 0.5 * iteration / n_iterations)
        
        # Find all distance violations
        violations = []
        for i, n1 in enumerate(all_nodes):
            for n2 in all_nodes[i+1:]:
                dist = euclidean_dist(n1, n2)
                if dist < min_distance:
                    violations.append((n1, n2, dist))
        
        if not violations:
            break  # All constraints satisfied
        
        # Compute repulsion forces only for violating pairs
        forces = defaultdict(float)
        
        for n1, n2, dist in violations:
            dx = x_pos[n1] - x_pos[n2]
            
            # Deficit: how much more distance we need
            deficit = min_distance - dist
            
            # Direction to push
            if abs(dx) < 0.001:
                direction = 1 if n1 < n2 else -1
            else:
                direction = np.sign(dx)
            
            # Force magnitude proportional to deficit
            force_mag = deficit * 0.3
            
            # Apply force to non-leaf nodes
            if heights[n1] > 0:
                forces[n1] += force_mag * direction
            if heights[n2] > 0:
                forces[n2] -= force_mag * direction
        
        # Apply forces with conditional constraints
        for node, force in forces.items():
            if heights[node] == 0:
                continue
            
            new_x = x_pos[node] + lr * force
            child_lo, child_hi = get_child_range(node)
            
            # Check if new position would be within children's range
            if child_lo <= new_x <= child_hi:
                # Stay within range - no problem
                x_pos[node] = new_x
            else:
                # Would go outside - check if we NEED to
                # Release node if it still has violations at the boundary
                clipped_x = np.clip(new_x, child_lo, child_hi)
                temp_x = x_pos[node]
                x_pos[node] = clipped_x
                
                # Check if any violations remain
                still_violated = False
                for n1, n2, _ in violations:
                    if node in (n1, n2):
                        if euclidean_dist(n1, n2) < min_distance:
                            still_violated = True
                            break
                
                if still_violated:
                    # Need to go outside children's range
                    released.add(node)
                    x_pos[node] = np.clip(new_x, x_min, x_max)
                else:
                    x_pos[node] = clipped_x
        
        # For released nodes, allow them to move freely
        for node in released:
            if node in forces:
                new_x = x_pos[node] + lr * forces[node]
                x_pos[node] = np.clip(new_x, x_min, x_max)
        
        # Weak centering force to children centroid (only for unreleased nodes)
        for node in all_nodes:
            if heights[node] > 0 and node not in released:
                if children_map[node]:
                    centroid = get_child_centroid(node)
                    x_pos[node] += 0.02 * (centroid - x_pos[node])
    
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


def check_min_distances(arg_data):
    """Check minimum distances between all node pairs."""
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
    
    all_nodes = list(x_pos.keys())
    min_dist = float('inf')
    min_pair = None
    
    for i, n1 in enumerate(all_nodes):
        for n2 in all_nodes[i+1:]:
            dx = x_pos[n1] - x_pos[n2]
            dy = heights[n1] - heights[n2]
            dist = np.sqrt(dx**2 + dy**2)
            if dist < min_dist:
                min_dist = dist
                min_pair = (n1, n2)
    
    return {'min_distance': min_dist, 'min_pair': min_pair}


def check_parents_outside_children(arg_data):
    """Find parents positioned outside their children's x-range."""
    x_pos = {}
    for leaf in arg_data['Leaf']:
        x_pos[leaf['nodeid']] = leaf['xpos']
    for coal in arg_data['Coalescent']:
        x_pos[coal['nodeid']] = coal['xpos']
    for recomb in arg_data['Recombination']:
        x_pos[recomb['nodeid']] = recomb['xpos']
    
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
                    'children_range': (lo, hi)
                })
    
    return outside


if __name__ == '__main__':
    import copy
    
    arg_json = {
        "Coalescent": [
            {"nodeid": 5, "height": 0.137, "xpos": 0.125},
            {"nodeid": 7, "height": 0.390, "xpos": 0.5125},
            {"nodeid": 8, "height": 0.550, "xpos": 0.606},
            {"nodeid": 10, "height": 0.737, "xpos": 0.578},
            {"nodeid": 11, "height": 0.991, "xpos": 0.312},
            {"nodeid": 12, "height": 1.0, "xpos": 0.445}
        ],
        "Recombination": [
            {"nodeid": 6, "height": 0.332, "xpos": 0.125},
            {"nodeid": 9, "height": 0.593, "xpos": 0.5}
        ],
        "Leaf": [
            {"nodeid": 0, "height": 0.0, "xpos": 0.5},
            {"nodeid": 1, "height": 0.0, "xpos": 0.0},
            {"nodeid": 2, "height": 0.0, "xpos": 0.25},
            {"nodeid": 3, "height": 0.0, "xpos": 1.0},
            {"nodeid": 4, "height": 0.0, "xpos": 0.75}
        ],
        "Lineage": [
            {"lineageid": 0, "down": 0, "up": 9},
            {"lineageid": 1, "down": 1, "up": 5},
            {"lineageid": 2, "down": 2, "up": 5},
            {"lineageid": 3, "down": 3, "up": 8},
            {"lineageid": 4, "down": 4, "up": 7},
            {"lineageid": 5, "down": 5, "up": 6},
            {"lineageid": 6, "down": 6, "up": 11},
            {"lineageid": 7, "down": 6, "up": 7},
            {"lineageid": 8, "down": 7, "up": 8},
            {"lineageid": 9, "down": 8, "up": 10},
            {"lineageid": 10, "down": 9, "up": 10},
            {"lineageid": 11, "down": 9, "up": 11},
            {"lineageid": 12, "down": 10, "up": 12},
            {"lineageid": 13, "down": 11, "up": 12},
            {"lineageid": 14, "down": 12, "up": None}
        ]
    }
    
    n_samples = len(arg_json['Leaf'])
    default_min_dist = 1.0 / (2 * n_samples)
    
    print(f"=== Minimum Distance Layout ===")
    print(f"n_samples = {n_samples}")
    print(f"x-range = [0, 1]")
    print(f"default min_distance = 1/(2*{n_samples}) = {default_min_dist:.3f}")
    print()
    
    # With default min_distance
    result = compute_arg_xpos(copy.deepcopy(arg_json))
    result_info = check_min_distances(result)
    outside = check_parents_outside_children(result)
    
    print(f"Result with default min_distance={default_min_dist:.3f}:")
    print(f"  Actual min distance: {result_info['min_distance']:.3f} (nodes {result_info['min_pair']})")
    print(f"  Crossings: {count_crossings(result)}")
    print(f"  Parents outside children: {len(outside)}")
    for o in outside:
        print(f"    Node {o['node']}: x={o['x']:.3f}, children=[{o['children_range'][0]:.3f}, {o['children_range'][1]:.3f}]")
    
    print(f"\nLeaf positions: ", end="")
    for leaf in sorted(result['Leaf'], key=lambda x: x['nodeid']):
        print(f"{leaf['nodeid']}:{leaf['xpos']:.2f}", end=" ")
    print()
    
    print(f"Internal positions: ", end="")
    for coal in sorted(result['Coalescent'], key=lambda x: x['nodeid']):
        print(f"{coal['nodeid']}:{coal['xpos']:.2f}", end=" ")
    for recomb in sorted(result['Recombination'], key=lambda x: x['nodeid']):
        print(f"{recomb['nodeid']}:{recomb['xpos']:.2f}", end=" ")
    print()
