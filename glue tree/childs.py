import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

def get_column_states_and_projectors(depth: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    vertices_per_tree = 2**(depth+1) - 1
    total_vertices = 2 * vertices_per_tree
    num_columns = 2 * depth + 2  # Total number of columns (0 to 2n+1)
    
    column_states = []
    projectors = []
    
    for j in range(num_columns):
        # Initialize column state vector
        col_state = np.zeros(total_vertices, dtype=complex)
        
        # Determine which vertices belong to column j
        if j == 0:
            # Column 0: entrance (root of left tree)
            vertices_in_col = [0]
            N_j = 1
        elif j <= depth:
            # Columns 1 to n: levels of left tree
            level = j - 1
            level_start = 2**level
            level_end = 2**(level+1)
            vertices_in_col = list(range(level_start, min(level_end, vertices_per_tree)))
            N_j = len(vertices_in_col)  # This equals 2^j for j <= n
        elif j == depth + 1:
            # Column n+1: root of right tree (exit)
            vertices_in_col = [vertices_per_tree]
            N_j = 1
        else:
            # Columns n+2 to 2n+1: levels of right tree
            right_level = j - depth - 2
            level_start = 2**right_level + vertices_per_tree
            level_end = 2**(right_level+1) + vertices_per_tree
            vertices_in_col = list(range(level_start, min(level_end, total_vertices)))
            N_j = len(vertices_in_col)  # This equals 2^(2n+1-j) for j >= n+1
        
        # Create normalized column state |col j⟩ = (1/√N_j) Σ |vertex⟩
        if N_j > 0:
            normalization = 1.0 / np.sqrt(N_j)
            for vertex in vertices_in_col:
                col_state[vertex] = normalization
        
        column_states.append(col_state)
        
        # Create projector P_j = |col j⟩⟨col j|
        P_j = np.outer(col_state, col_state.conj())
        projectors.append(P_j)
    
    return column_states, projectors

def compute_column_probabilities(psi: np.ndarray, projectors: List[np.ndarray]) -> np.ndarray:
    """
    Compute probability distribution across columns for a given quantum state.
    
    Parameters:
    -----------
    psi : np.ndarray
        Quantum state vector
    projectors : List[np.ndarray]
        List of column projection operators
        
    Returns:
    --------
    probabilities : np.ndarray
        Probability for each column
    """
    probabilities = []
    for P in projectors:
        # Probability = ⟨ψ|P_j|ψ⟩ = ||P_j|ψ⟩||²
        prob = np.real(psi.conj().T @ P @ psi)
        probabilities.append(prob)
    
    return np.array(probabilities)

def get_reduced_hamiltonian(depth: int) -> np.ndarray:
    """
    Get the reduced Hamiltonian acting on the column subspace.
    Following Eq. (27) from Childs et al.
    
    Parameters:
    -----------
    depth : int
        Depth of each binary tree (n in the paper)
        
    Returns:
    --------
    H_reduced : np.ndarray
        Reduced Hamiltonian matrix on column subspace
    """
    num_columns = 2 * depth + 2  # Columns 0 to 2n+1
    H_reduced = np.zeros((num_columns, num_columns), dtype=complex)
    
    # Set γ = 1/√2 as in the paper
    gamma = 1.0 / np.sqrt(2)
    
    for j in range(num_columns - 1):
        if 0 <= j <= depth - 1:
            # ⟨col j|H|col(j+1)⟩ = √2γ for 0 ≤ j ≤ n-1
            H_reduced[j, j+1] = np.sqrt(2) * gamma
            H_reduced[j+1, j] = np.sqrt(2) * gamma
        elif j == depth:
            # ⟨col n|H|col(n+1)⟩ = 2γ for j = n
            H_reduced[j, j+1] = 2 * gamma
            H_reduced[j+1, j] = 2 * gamma
        elif depth + 1 <= j <= 2 * depth:
            # ⟨col j|H|col(j+1)⟩ = √2γ for n+1 ≤ j ≤ 2n
            H_reduced[j, j+1] = np.sqrt(2) * gamma
            H_reduced[j+1, j] = np.sqrt(2) * gamma
    
    return H_reduced

def simulate_quantum_walk(H: np.ndarray, initial_state: np.ndarray, 
                         times: List[float]) -> List[np.ndarray]:
    """
    Simulate quantum walk at specified times.
    
    Parameters:
    -----------
    H : np.ndarray
        Hamiltonian matrix
    initial_state : np.ndarray
        Initial quantum state
    times : List[float]
        List of times at which to evaluate the state
        
    Returns:
    --------
    states : List[np.ndarray]
        List of quantum states at specified times
    """
    states = []
    for t in times:
        # Compute time evolution operator
        U_t = expH(H, t)
        # Evolve the state
        psi_t = U_t @ initial_state
        states.append(psi_t)
    
    return states

def create_childs_figure_5_4(depth: int = 4, times: List[float] = None, 
                            figsize: Tuple[int, int] = (12, 8), use_reduced: bool = True):
    """
    Recreate Childs' thesis Figure 5-4: Probability vs columns at different timesteps.
    
    Parameters:
    -----------
    depth : int
        Depth of the glued tree (default: 4)
    times : List[float]
        Times at which to plot the probability distribution
    figsize : Tuple[int, int]
        Figure size for matplotlib
    use_reduced : bool
        Whether to use the reduced column-space Hamiltonian (more efficient and accurate)
    """
    if times is None:
        # Choose times that show interesting wave packet dynamics
        # Based on the paper, propagation speed is ~2, so time to traverse is ~depth
        times = [0.0, depth/4, depth/2, 3*depth/4, depth, 5*depth/4, 3*depth/2]
    
    num_columns = 2 * depth + 2
    
    if use_reduced:
        # Use the reduced Hamiltonian on column subspace (more efficient)
        print(f"Using reduced column-space Hamiltonian (depth {depth}, {num_columns} columns)...")
        H_reduced = get_reduced_hamiltonian(depth)
        
        # Initial state in column subspace (entrance is column 0)
        initial_state_col = np.zeros(num_columns, dtype=complex)
        initial_state_col[0] = 1.0
        
        # Simulate quantum walk in column subspace
        print("Simulating quantum walk in column subspace...")
        states_col = simulate_quantum_walk(H_reduced, initial_state_col, times)
        
        # Probabilities are just |amplitude|²
        prob_distributions = []
        for state_col in states_col:
            probs = np.abs(state_col)**2
            prob_distributions.append(probs)
    
    else:
        # Use full Hamiltonian (slower but shows full physics)
        print(f"Generating full glued tree Hamiltonian (depth {depth})...")
        H = generate_glued_tree_hamiltonian(depth=depth, seed=42, degree=3, verbose=False)
        
        # Scale Hamiltonian appropriately
        gamma = 1.0 / np.sqrt(2)
        H_scaled = gamma * H
        
        # Create initial state (entrance vertex)
        vertices_per_tree = 2**(depth+1) - 1
        total_vertices = 2 * vertices_per_tree
        initial_state = np.zeros(total_vertices, dtype=complex)
        initial_state[0] = 1.0  # Start at entrance (vertex 0)
        
        # Get column projectors
        column_states, projectors = get_column_states_and_projectors(depth)
        
        # Simulate quantum walk
        print("Simulating quantum walk on full graph...")
        states = simulate_quantum_walk(H_scaled, initial_state, times)
        
        # Compute probability distributions
        prob_distributions = []
        for state in states:
            probs = compute_column_probabilities(state, projectors)
            prob_distributions.append(probs)
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Define colors for different time steps (use a colormap)
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(times)))
    
    # Plot probability distributions
    columns = np.arange(num_columns)
    
    for i, (t, probs) in enumerate(zip(times, prob_distributions)):
        plt.plot(columns, probs, 'o-', color=colors[i], 
                label=f't = {t:.2f}', linewidth=2.5, markersize=8, alpha=0.8)
    
    # Formatting
    plt.xlabel('Column Index j', fontsize=14)
    plt.ylabel('Probability |⟨col j|ψ(t)⟩|²', fontsize=14)
    plt.title(f'Quantum Walk on Glued Tree (depth n = {depth})\n' + 
              'Probability Distribution Across Columns at Different Times', 
              fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='upper right')
    
    # Mark special columns
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
    plt.axvline(x=depth, color='orange', linestyle='--', alpha=0.7, linewidth=2)
    plt.axvline(x=depth+1, color='orange', linestyle='--', alpha=0.7, linewidth=2)
    plt.axvline(x=num_columns-1, color='green', linestyle='--', alpha=0.7, linewidth=2)
    
    # Add text annotations
    plt.text(0, plt.ylim()[1]*0.95, 'Entrance', rotation=90, ha='right', va='top', 
             color='red', fontweight='bold')
    plt.text(depth, plt.ylim()[1]*0.95, 'Center L', rotation=90, ha='right', va='top', 
             color='orange', fontweight='bold')
    plt.text(depth+1, plt.ylim()[1]*0.95, 'Center R', rotation=90, ha='right', va='top', 
             color='orange', fontweight='bold')
    plt.text(num_columns-1, plt.ylim()[1]*0.95, 'Exit', rotation=90, ha='right', va='top', 
             color='green', fontweight='bold')
    
    # Set x-axis ticks
    plt.xticks(columns[::max(1, num_columns//10)])
    plt.xlim(-0.5, num_columns-0.5)
    plt.ylim(0, None)
    
    plt.tight_layout()
    
    # Print some statistics
    print(f"\nQuantum Walk Statistics:")
    print(f"Total columns: {num_columns}")
    print(f"Times simulated: {times}")
    print(f"Method: {'Reduced column-space' if use_reduced else 'Full graph'}")
    
    for i, (t, probs) in enumerate(zip(times, prob_distributions)):
        max_prob_col = np.argmax(probs)
        max_prob = probs[max_prob_col]
        entrance_prob = probs[0]
        exit_prob = probs[-1]
        print(f"t = {t:.2f}: Max prob {max_prob:.4f} at col {max_prob_col}, " +
              f"Entrance: {entrance_prob:.4f}, Exit: {exit_prob:.4f}")
    
    plt.show()
    
    # return prob_distributions, timessize=14)

    plt.title(f'Quantum Walk on Glued Tree (depth = {depth})\nProbability Distribution Across Columns', 
              fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Mark entrance and exit columns
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Entrance')
    plt.axvline(x=num_columns-1, color='green', linestyle='--', alpha=0.7, label='Exit')
    
    # Add column labels
    column_labels = []
    for i in range(num_columns):
        if i == 0:
            column_labels.append('Entrance')
        elif i == depth:
            column_labels.append('Center L')
        elif i == depth + 1:
            column_labels.append('Center R')
        elif i == num_columns - 1:
            column_labels.append('Exit')
        else:
            column_labels.append(f'{i}')
    
    plt.xticks(columns, column_labels, rotation=45)
    plt.tight_layout()
    
    # Print some statistics
    print(f"\nQuantum Walk Statistics:")
    print(f"Total columns: {num_columns}")
    print(f"Times simulated: {times}")
    
    for i, (t, probs) in enumerate(zip(times, prob_distributions)):
        max_prob_col = np.argmax(probs)
        max_prob = probs[max_prob_col]
        print(f"t = {t:.2f}: Max probability {max_prob:.4f} at column {max_prob_col}")
    
    plt.show()
    
    return prob_distributions, times

def analyze_hitting_time(depth: int = 4, max_time: float = None, num_points: int = 100):
    """
    Analyze the hitting time to the exit vertex.
    
    Parameters:
    -----------
    depth : int
        Depth of the glued tree
    max_time : float
        Maximum time to simulate (default: 2*depth)
    num_points : int
        Number of time points to evaluate
    """
    if max_time is None:
        max_time = 2 * depth
    
    # Generate the glued tree Hamiltonian
    H = generate_glued_tree_hamiltonian(depth=depth, seed=42, degree=3, verbose=False)
    gamma = 1.0 / np.sqrt(2)
    H_scaled = gamma * H
    
    # Initial state at entrance
    vertices_per_tree = 2**(depth+1) - 1
    total_vertices = 2 * vertices_per_tree
    initial_state = np.zeros(total_vertices)
    initial_state[0] = 1.0
    
    # Exit vertex
    exit_vertex = vertices_per_tree
    
    # Time points
    times = np.linspace(0, max_time, num_points)
    exit_probabilities = []
    
    print("Computing hitting time analysis...")
    for t in times:
        U_t = expH(H_scaled, t)
        psi_t = U_t @ initial_state
        exit_prob = np.abs(psi_t[exit_vertex])**2
        exit_probabilities.append(exit_prob)
    
    # Plot hitting time
    plt.figure(figsize=(10, 6))
    plt.plot(times, exit_probabilities, 'b-', linewidth=2)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Probability at Exit', fontsize=14)
    plt.title(f'Hitting Time Analysis (depth = {depth})', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    # Find time of maximum probability
    max_idx = np.argmax(exit_probabilities)
    max_time_hit = times[max_idx]
    max_prob_hit = exit_probabilities[max_idx]
    
    plt.axvline(x=max_time_hit, color='red', linestyle='--', alpha=0.7)
    plt.text(max_time_hit, max_prob_hit, 
             f'Max: t={max_time_hit:.2f}\nP={max_prob_hit:.4f}', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    print(f"Maximum exit probability: {max_prob_hit:.4f} at time t = {max_time_hit:.2f}")
    
    return times, exit_probabilities

# Example usage
if __name__ == "__main__":
    # Recreate Figure 5-4 with depth 4
    depth = 4
    times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]  # Adjust these times as needed
    
    prob_dists, times_used = create_childs_figure_5_4(depth=depth, times=times)
    
    # Also analyze hitting time
    print("\n" + "="*50)
    analyze_hitting_time(depth=depth)