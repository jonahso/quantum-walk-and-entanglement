import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from typing import Tuple, List, Dict


def hamiltonian_to_graph(H: np.ndarray) -> nx.Graph:
    """
    Convert a Hamiltonian (adjacency matrix) to a NetworkX graph.
    
    Parameters:
    -----------
    H : np.ndarray
        Hamiltonian matrix (adjacency matrix)
        
    Returns:
    --------
    G : networkx.Graph
        NetworkX graph representation
    """
    # Create an empty graph with nodes corresponding to matrix indices
    G = nx.Graph()
    G.add_nodes_from(range(H.shape[0]))
    
    # Add edges based on the Hamiltonian
    for i in range(H.shape[0]):
        for j in range(i+1, H.shape[1]):  # Only upper triangle to avoid duplicates
            if H[i, j] == 1:
                G.add_edge(i, j)
    
    return G

def visualize_glued_tree_graph(G: nx.Graph, depth: int, node_size: int = 300, save_file: str = None) -> None:
    """
    Visualize the glued tree graph with the following properties:
    - Both trees are horizontally aligned (side by side)
    - Nodes at the same depth in each tree are on the same horizontal level
    - Leaf layers of the two trees are placed on separate vertical lines to clearly show the gluing pattern
    - The tree on the left grows from its root rightward, the tree on the right grows from its root leftward
    
    Parameters:
    -----------
    G : networkx.Graph
        Graph to visualize
    depth : int
        Depth of each binary tree
    node_size : int, optional
        Size of nodes in visualization
    save_file : str, optional
        Filename to save the visualization (if None, will not save)
    """
    # Calculate number of vertices in each tree
    vertices_per_tree = 2**(depth+1) - 1
    
    # Create position dictionary for nodes
    pos = {}
    
    # Set horizontal spacing parameters
    level_spacing = 3  # Horizontal spacing between levels
    tree_spacing = 5   # Extra spacing between the two trees
    
    # Now assign positions for the first tree (left side)
    for level in range(depth+1):
        level_start = 2**level - 1
        level_end = 2**(level+1) - 1
        level_width = level_end - level_start
        
        # Calculate x-coordinate based on level
        # For the leaf level, place it at a special position
        if level == depth:
            x = -tree_spacing / 2  # Special position for left tree leaves
        else:
            x = -level_spacing * (depth - level) - tree_spacing
        
        for i, node in enumerate(range(level_start, min(level_end, vertices_per_tree))):
            # Determine y-position (varies within each level)
            if level_width > 1:
                y_spread = 8.0  # Total vertical spread for the level
                y_offset = (i - level_width/2 + 0.5) * (y_spread / level_width)
                y = -y_offset  # Center the vertical spread
            else:
                y = 0
            
            # Position the node
            pos[node] = (x, y)
    
    # Assign positions for the second tree (right side)
    for level in range(depth+1):
        level_start = 2**level - 1
        level_end = 2**(level+1) - 1
        level_width = level_end - level_start
        
        # Calculate x-coordinate based on level
        # For the leaf level, place it at a special position
        if level == depth:
            x = tree_spacing / 2  # Special position for right tree leaves
        else:
            x = level_spacing * (depth - level) + tree_spacing
            
        for i, rel_node in enumerate(range(level_start, min(level_end, vertices_per_tree))):
            node = rel_node + vertices_per_tree
            
            # Determine y-position (varies within each level)
            # Use same vertical positioning as corresponding nodes in first tree for horizontal alignment
            if level_width > 1:
                y_spread = 8.0  # Total vertical spread for the level
                y_offset = (i - level_width/2 + 0.5) * (y_spread / level_width)
                y = -y_offset  # Center the vertical spread
            else:
                y = 0
            
            # Position the node
            pos[node] = (x, y)
    
    # Create node color map: roots are red, internal nodes are blue, leaves are green
    # Glued connections are gold
    color_map = []
    edge_colors = []
    
    # Identify the leaves to find the glued edges
    tree0_leaves = list(range(2**depth - 1, 2**(depth+1) - 1))
    tree1_leaves = [x + vertices_per_tree for x in range(2**depth - 1, 2**(depth+1) - 1)]
    
    for node in G.nodes():
        if node == 0:  # First tree root (entrance)
            color_map.append('red')
        elif node == vertices_per_tree:  # Second tree root (exit)
            color_map.append('red')
        elif node in tree0_leaves or node in tree1_leaves:  # Leaves
            color_map.append('green')
        else:  # Internal nodes
            color_map.append('skyblue')
    
    for u, v in G.edges():
        if (u in tree0_leaves and v in tree1_leaves) or (v in tree0_leaves and u in tree1_leaves):
            edge_colors.append('gold')
        else:
            edge_colors.append('black')
    
    # Draw the graph
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color=color_map, edge_color=edge_colors, 
            node_size=node_size, font_size=10, font_color='white', font_weight='bold',
            width=2)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Entrance/Exit'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', markersize=10, label='Internal Node'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Leaf Node'),
        plt.Line2D([0], [0], color='black', lw=2, label='Tree Edge'),
        plt.Line2D([0], [0], color='gold', lw=2, label='Glued Connection')
    ]
    
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    plt.title(f'Glued Tree Graph (depth = {depth})', size=16)
    # plt.tight_layout()
    plt.axis('off')
    
    # Save the figure if a filename is provided
    if save_file:
        plt.savefig(save_file, dpi=300)
        
    plt.show()

def print_hamiltonian(H: np.ndarray) -> None:
    """
    Print the Hamiltonian matrix in a readable format
    
    Parameters:
    -----------
    H : np.ndarray
        Hamiltonian matrix to print
    """
    print("\nHamiltonian (Adjacency Matrix):")
    for i in range(len(H)):
        row = " ".join(str(x) for x in H[i])
        print(f"{i:2d}: [{row}]")


def identify_graph_components(H: np.ndarray, depth: int) -> Dict:
    """
    Identify important components of the glued tree graph based on the Hamiltonian.
    
    Parameters:
    -----------
    H : np.ndarray
        Hamiltonian matrix
    depth : int
        Depth of each tree
        
    Returns:
    --------
    components : dict
        Dictionary containing identified components
    """
    vertices_per_tree = 2**(depth+1) - 1
    total_vertices = 2 * vertices_per_tree
    
    tree0_leaves = list(range(2**depth - 1, 2**(depth+1) - 1))
    tree1_leaves = [x + vertices_per_tree for x in range(2**depth - 1, 2**(depth+1) - 1)]
    
    # Find glued connections
    glued_pairs = []
    for leaf0 in tree0_leaves:
        for leaf1 in tree1_leaves:
            if H[leaf0, leaf1] == 1:
                glued_pairs.append((leaf0, leaf1))
    
    components = {
        "entrance": 0,
        "exit": vertices_per_tree,
        "tree0_leaves": tree0_leaves,
        "tree1_leaves": tree1_leaves,
        "glued_pairs": glued_pairs
    }
    
    return components


def generate_glued_tree_hamiltonian(depth: int, seed: int = None, degree=3, verbose: bool = True) -> np.ndarray:
    """
    Generate a Hamiltonian for the glued tree problem with specified depth.
    
    Parameters:
    -----------
    depth : int
        Depth of each binary tree
    seed : int, optional
        Random seed for reproducibility
    verbose : bool, optional
        Whether to print details about the graph structure
        
    Returns:
    --------
    H : np.ndarray
        Hamiltonian matrix (adjacency matrix) of the glued tree
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Calculate number of vertices in each tree
    vertices_per_tree = 2**(depth+1) - 1
    total_vertices = 2 * vertices_per_tree
    
    if verbose:
        print(f"Generating glued tree with depth {depth}")
        print(f"Each tree has {vertices_per_tree} vertices")
        print(f"Total number of vertices: {total_vertices}")
    
    # Initialize Hamiltonian matrix with zeros
    H = np.zeros((total_vertices, total_vertices), dtype=int)
    
    # Helper function to get leaf nodes of a tree
    def get_leaves(tree_root, tree_depth):
        start_idx = 2**(tree_depth) - 1 + tree_root
        end_idx = 2**(tree_depth+1) - 1 + tree_root
        return list(range(start_idx, end_idx))
    
    # Connect first tree (tree 0)
    tree0_start = 0
    for level in range(depth):
        level_start = 2**level - 1
        level_end = 2**(level+1) - 1
        
        for parent in range(level_start, level_end):
            left_child = 2 * parent + 1
            right_child = 2 * parent + 2
            
            if left_child < vertices_per_tree:
                H[parent][left_child] = H[left_child][parent] = 1
                
            if right_child < vertices_per_tree:
                H[parent][right_child] = H[right_child][parent] = 1
    
    # Connect second tree (tree 1)
    tree1_start = vertices_per_tree
    for level in range(depth):
        level_start = 2**level - 1
        level_end = 2**(level+1) - 1
        
        for rel_parent in range(level_start, level_end):
            parent = rel_parent + tree1_start
            left_child = 2 * rel_parent + 1 + tree1_start
            right_child = 2 * rel_parent + 2 + tree1_start
            
            if left_child < total_vertices:
                H[parent][left_child] = H[left_child][parent] = 1
                
            if right_child < total_vertices:
                H[parent][right_child] = H[right_child][parent] = 1
    
    # Get leaves of both trees
    tree0_leaves = get_leaves(0, depth)
    tree1_leaves = get_leaves(tree1_start, depth)
    # print('tree leaves: ', tree0_leaves, tree1_leaves)
    
    # Create random cycle between leaves
    tree1_leaves_shuffled = tree1_leaves.copy()
    random.shuffle(tree1_leaves_shuffled)
    
    # Connect leaves of both trees
    if degree == 2:
        gluing_pairs = []
        for i, leaf0 in enumerate(tree0_leaves):
            leaf1 = tree1_leaves_shuffled[i]
            H[leaf0][leaf1] = H[leaf1][leaf0] = 1
            gluing_pairs.append((leaf0, leaf1))
    elif degree == 3:
        gluing_pairs = []
        tree0_leaves_shuffled = tree0_leaves.copy()
        random_cycle = interleave(tree0_leaves_shuffled, tree1_leaves_shuffled)
        random_cycle.append(tree0_leaves_shuffled[0])  # Close the cycle
        for i in range(len(random_cycle) - 1):
            leaf0 = random_cycle[i]
            leaf1 = random_cycle[i + 1]
            H[leaf0][leaf1] = H[leaf1][leaf0] = 1
            gluing_pairs.append((leaf0, leaf1))
    else:
        raise ValueError(" central layer Degree must be either 2 or 3 for the glued tree problem.")
        
    if verbose:
        print("\nGlued connections between leaves:")
        for leaf0, leaf1 in gluing_pairs:
            print(f"  Leaf {leaf0} (Tree 0) ↔ Leaf {leaf1} (Tree 1)")
    
    return H

def interleave(t1, t2):
    interleaved = []
    for a, b in zip(t1, t2):
        interleaved.append(a)
        interleaved.append(b)
    return interleaved

# [x for pair in zip(t1, t2) for x in pair] # Alternative interleave method


import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl

import matplotlib.pyplot as plt

import scipy
from scipy.sparse import csr_matrix, csc_matrix
# from quantum_simulation_recipe.bounds import *

def expH(H, t, use_jax=False):
    # # check H is Hermitian
    # if not np.allclose(H, H.conj().T):
    #     raise ValueError('H is not Hermitian')

    if use_jax: 
        if isinstance(H, np.ndarray):
            return jax.scipy.linalg.expm(-1j * t * H)
        else:
            return jax.scipy.linalg.expm(-1j * t * H.to_matrix())
    elif isinstance(H, csr_matrix):
        return scipy.sparse.linalg.expm(-1j * t * H)
    else:
        return scipy.linalg.expm(-1j * t * H)