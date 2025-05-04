import matplotlib.pyplot as plt
import stim

def generate_unrotated_surface_code_grid(d_x, d_y, cut=None, cut_gap=5, vertical=False):
    """
    Generate coordinates for data qubits, X ancillas, and Z ancillas in an unrotated surface code grid.
    
    Args:
        d_x (int): Width of the grid in the x-direction.
        d_y (int): Height of the grid in the y-direction.
        cut (int, optional): The x-coordinate where the lattice is cut. Defaults to None.
        cut_gap (int, optional): The gap introduced at the cut. Defaults to 5.
        vertical (bool, optional): Whether to vertical the roles of X and Z ancillas. Defaults to False.
    
    Returns:
        tuple: Sorted lists of data qubits, X ancillas, Z ancillas, observables, and a qubit dictionary.
    """
    data_coords = set()
    x_measure_coords = set()
    z_measure_coords = set()
    x_observable = []
    z_observable = []

    # Determine the x-offset for positions after the cut
    def get_x_offset(x):
        if cut is not None and x >= cut:
            return cut_gap
        else:
            return 0

    for x in range(2 * d_x - 1):
        x_offset = get_x_offset(x)
        for y in range(2 * d_y - 1):
            q = (x + x_offset, y)
            parity = (x % 2) != (y % 2)
            if parity:
                if x % 2 == 0:
                    z_measure_coords.add(q)
                else:
                    x_measure_coords.add(q)
            else:
                data_coords.add(q)
                if x == 0:
                    x_observable.append(q)
                if y == 0:
                    z_observable.append(q)

    # **Row-Wise Sorting:** Sort primarily by y (rows), then by x (columns)
    def row_wise_sort(coords):
        return sorted(coords, key=lambda coord: (coord[1], coord[0]))

    data_qubits_sorted = row_wise_sort(data_coords)
    x_measure_sorted = row_wise_sort(x_measure_coords)
    z_measure_sorted = row_wise_sort(z_measure_coords)

    # Assign unique numerical IDs with prefixes based on row-wise order
    # Example: '4i' becomes 4i where i starts from 1
    data_qubit_ids = [int(f'4{i+1}') for i in range(len(data_qubits_sorted))]
    x_ancilla_ids = [int(f'3{i+1}') for i in range(len(x_measure_sorted))]
    z_ancilla_ids = [int(f'2{i+1}') for i in range(len(z_measure_sorted))]

    # If inversion is requested, swap X and Z ancilla roles
    if vertical:
        x_measure_sorted, z_measure_sorted = z_measure_sorted, x_measure_sorted
        x_ancilla_ids, z_ancilla_ids = z_ancilla_ids, x_ancilla_ids
        x_observable, z_observable = z_observable, x_observable

    # Create the dictionary categorizing qubits
    qubit_dict = {
        'data_qubits': data_qubit_ids,
        'x_ancillas': x_ancilla_ids,
        'z_ancillas': z_ancilla_ids
    }

    # Return sorted lists and qubit_dict
    return data_qubits_sorted, x_measure_sorted, z_measure_sorted, x_observable, z_observable, qubit_dict

def plot_unrotated_surface_code_grid(d_x, d_y, cut=None, cut_gap=5, annotate=True, vertical=False):
    """
    Plot the unrotated surface code grid with an option to vertical X and Z ancillas.
    
    Args:
        d_x (int): Width of the grid in the x-direction.
        d_y (int): Height of the grid in the y-direction.
        cut (int, optional): The x-coordinate where the lattice is cut. Defaults to None.
        cut_gap (int, optional): The gap introduced at the cut. Defaults to 5.
        annotate (bool, optional): Whether to annotate qubits with their IDs. Defaults to True.
        vertical (bool, optional): Whether to vertical the roles of X and Z ancillas. Defaults to False.
    
    Returns:
        tuple: Sorted lists of data qubits, X ancillas, Z ancillas, observables, and a qubit dictionary.
    """
    # Generate grid coordinates and qubit dictionary
    data_qubits_sorted, x_measure_sorted, z_measure_sorted, x_observable, z_observable, qubit_dict = generate_unrotated_surface_code_grid(
        d_x, d_y, cut, cut_gap, vertical
    )
    
    # Extract coordinates for plotting
    data_x = [x for (x, y) in data_qubits_sorted]
    data_y = [y for (x, y) in data_qubits_sorted]
    
    x_measure_x = [x for (x, y) in x_measure_sorted]
    x_measure_y = [y for (x, y) in x_measure_sorted]
    
    z_measure_x = [x for (x, y) in z_measure_sorted]
    z_measure_y = [y for (x, y) in z_measure_sorted]
    
    plt.figure(figsize=(12, 10))
    
    # Plot Data Qubits with reduced opacity
    plt.scatter(data_x, data_y, c='black', marker='o', label='Data Qubits', alpha=0.2, zorder=1)
    # Plot X Ancillas with reduced opacity
    plt.scatter(x_measure_x, x_measure_y, c='red', marker='s', label='X Ancillas', alpha=0.2, zorder=1)
    # Plot Z Ancillas with reduced opacity
    plt.scatter(z_measure_x, z_measure_y, c='blue', marker='^', label='Z Ancillas', alpha=0.2, zorder=1)
    
    # Highlight the logical observables
    x_obs_x = [x for (x, y) in x_observable]
    x_obs_y = [y for (x, y) in x_observable]
    plt.scatter(x_obs_x, x_obs_y, facecolors='none', edgecolors='green', s=200, linewidths=2, label='X Observable Qubits', zorder=2)
    
    z_obs_x = [x for (x, y) in z_observable]
    z_obs_y = [y for (x, y) in z_observable]
    plt.scatter(z_obs_x, z_obs_y, facecolors='none', edgecolors='orange', s=200, linewidths=2, label='Z Observable Qubits', zorder=2)
    
    # Annotate qubits with their names
    if annotate:
        # Annotate Data Qubits
        for qubit_id, (x, y) in zip(qubit_dict['data_qubits'], data_qubits_sorted):
            plt.text(x, y, str(qubit_id), fontsize=8, ha='center', va='center', color='black', zorder=3)
        # Annotate X Ancillas
        for qubit_id, (x, y) in zip(qubit_dict['x_ancillas'], x_measure_sorted):
            plt.text(x, y, str(qubit_id), fontsize=8, ha='center', va='center', color='red', zorder=3)
        # Annotate Z Ancillas
        for qubit_id, (x, y) in zip(qubit_dict['z_ancillas'], z_measure_sorted):
            plt.text(x, y, str(qubit_id), fontsize=8, ha='center', va='center', color='blue', zorder=3)
    
    # Adjust legend to prevent overlap
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Construct the title
    title = f'{"Inverted " if vertical else ""}Unrotated Surface Code Grid (Width={d_x}, Height={d_y}'
    if cut is not None:
        title += f', Cut at {cut}'
    title += ')'
    
    plt.title(title, fontsize=14)
    plt.xlabel('X coordinate', fontsize=12)
    plt.ylabel('Y coordinate', fontsize=12)
    plt.gca().invert_yaxis()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    
    # Return sorted lists and qubit_dict for further use
    return data_qubits_sorted, x_measure_sorted, z_measure_sorted, x_observable, z_observable, qubit_dict

def create_coords(circuit, qubit_dict, data_qubits_sorted, x_measure_sorted, z_measure_sorted):
    """
    Assign spatial coordinates to each qubit using QUBIT_COORDS.
    
    Args:
        circuit (stim.Circuit): The Stim circuit to append instructions to.
        qubit_dict (dict): Dictionary containing lists of qubit IDs categorized as 'data_qubits', 'x_ancillas', 'z_ancillas'.
        data_qubits_sorted (list of tuples): List of coordinates for data qubits.
        x_measure_sorted (list of tuples): List of coordinates for X ancillas.
        z_measure_sorted (list of tuples): List of coordinates for Z ancillas.
    """
    # Create a mapping from qubit IDs to their coordinates
    qubit_id_to_coords = {}
    
    # Map data qubits
    for qubit_id, coord in zip(qubit_dict['data_qubits'], data_qubits_sorted):
        qubit_id_to_coords[qubit_id] = coord
    
    # Map X ancillas
    for qubit_id, coord in zip(qubit_dict['x_ancillas'], x_measure_sorted):
        qubit_id_to_coords[qubit_id] = coord
    
    # Map Z ancillas
    for qubit_id, coord in zip(qubit_dict['z_ancillas'], z_measure_sorted):
        qubit_id_to_coords[qubit_id] = coord
    
    # Create QUBIT_COORDS instructions
    coords_instructions = "\n".join([
        f"QUBIT_COORDS({x}, {y}) {qubit_id}" 
        for qubit_id, (x, y) in qubit_id_to_coords.items()
    ])
    
    # Append to the circuit
    circuit += stim.Circuit(coords_instructions + "\n")

def initial_reset(circuit, all_qubits):
    """
    Reset all qubits to the |0⟩ state.
    
    Args:
        circuit (stim.Circuit): The Stim circuit to append instructions to.
        all_qubits (list of int): List of all qubit IDs to reset.
    """
    # Create a RESET instruction for all qubits
    reset_instruction = "R " + " ".join(map(str, all_qubits)) + "\n"
    circuit += stim.Circuit(reset_instruction)


def begin_round(circuit, all_qubits,p1,r):
    """
    Depolarize all qubits to the |0⟩ state.

    Args:
        circuit (stim.Circuit): The Stim circuit to append instructions to.
        all_qubits (list of int): List of all qubit IDs to reset.
    """
    if not all_qubits:
        return  # No qubits to reset
    if r==0:
        reset_instruction = f"X_ERROR({p1}) " + " ".join(map(str, all_qubits)) + "\n"
        circuit += stim.Circuit(reset_instruction)
        circuit.append('TICK')
    else:

        reset_instruction = f"DEPOLARIZE1({p1}) " + " ".join(map(str, all_qubits)) + "\n"
        circuit += stim.Circuit(reset_instruction)
        circuit.append('TICK')

def apply_hadamard_ancillas(circuit, ancilla_ids, p1, ancilla_type='X'):
    """
    Apply Hadamard gates to ancillas and introduce Depolarize1 noise.
    
    Args:
        circuit (stim.Circuit): The Stim circuit to append instructions to.
        ancilla_ids (list of int): List of ancilla qubit IDs.
        p1 (float): Depolarization probability for DEPOLARIZE1.
        ancilla_type (str, optional): Type of ancilla ('X' or 'Z'). Defaults to 'X'.
    """
    # Apply Hadamard gates
    hadamard_instruction = "H " + " ".join(map(str, ancilla_ids)) + "\n"
    
    # Apply Depolarize1 to each ancilla individually
    depolarize1_instructions = "\n".join([
        f"DEPOLARIZE1({p1}) {qubit_id}" 
        for qubit_id in ancilla_ids
    ]) + "\n"
    
    # Combine instructions with TICK
    combined_instructions = f"{hadamard_instruction}{depolarize1_instructions}TICK\n"
    
    # Append to the circuit
    circuit += stim.Circuit(combined_instructions)

def stabilizer_round(circuit, qubit_dict, data_qubits_sorted, x_measure_sorted, z_measure_sorted, p1, p2, cut, spacing=1, vertical=False):
    """
    Perform stabilizer interactions with depolarizing noise, including special handling for boundary ancillas.
    
    Args:
        circuit (stim.Circuit): The Stim circuit to append instructions to.
        qubit_dict (dict): Dictionary containing lists of qubit IDs categorized as 'data_qubits', 'x_ancillas', 'z_ancillas'.
        data_qubits_sorted (list of tuples): List of coordinates for data qubits.
        x_measure_sorted (list of tuples): List of coordinates for X ancillas.
        z_measure_sorted (list of tuples): List of coordinates for Z ancillas.
        p1 (float): Depolarization probability for DEPOLARIZE2.
        p2 (float): Depolarization probability for boundary DEPOLARIZE2.
        cut (int): The x-coordinate where the lattice is cut.
        spacing (int, optional): The spacing between qubits (default is 1).
        vertical (bool, optional): Whether the circuit is inverted. Defaults to False.
    """
    # Define the order of directions for interactions
    interaction_order = ['Up', 'Right', 'Left', 'Down']
    
    # Create a mapping from data qubit coordinates to qubit IDs for data qubits
    data_qubit_coords_to_id = {
        coord: qubit_id 
        for qubit_id, coord in zip(qubit_dict['data_qubits'], data_qubits_sorted)
    }
    
    def get_neighbor_id(ancilla_coord, direction):
        x, y = ancilla_coord

        # Special treatment for ancillas at the cut boundary
        if x == cut-1:
            if direction == 'Right':
                neighbor_coord = (x + 6, y)  # Adjacent to the right side across the cut
            else:
                neighbors = {
                    'Up': (x, y - 1),
                    'Left': (x - 1, y),
                    'Down': (x, y + 1)
                }
                neighbor_coord = neighbors.get(direction, None)
        elif x == cut+5:
            if direction == 'Left':
                neighbor_coord = (x - 6, y)  # Adjacent to the left side across the cut
            else:
                neighbors = {
                    'Up': (x, y - 1),
                    'Right': (x + 1, y),
                    'Down': (x, y + 1)
                }
                neighbor_coord = neighbors.get(direction, None)
        else:
            # Ancillas exactly at the cut can have neighbors on both sides
            if direction == 'Right':
                neighbor_coord = (x + 1, y)
            elif direction == 'Left':
                neighbor_coord = (x - 1, y)
            else:
                neighbors = {
                    'Up': (x, y - 1),
                    'Down': (x, y + 1)
                }
                neighbor_coord = neighbors.get(direction, None)
        
        return data_qubit_coords_to_id.get(neighbor_coord, None)
    
    # Iterate over each direction
    for direction in interaction_order:
        cx_pairs = []
        depol2_pairs = []
        
        # Iterate over each ancilla type and their respective lists
        ancilla_types = [('X', qubit_dict['x_ancillas']), ('Z', qubit_dict['z_ancillas'])]
        if vertical:
            ancilla_types = [('Z', qubit_dict['z_ancillas']), ('X', qubit_dict['x_ancillas'])]
        
        for ancilla_type, ancilla_list in ancilla_types:
            for ancilla_id in ancilla_list:
                # Retrieve the coordinates of the current ancilla
                if ancilla_type == 'X':
                    ancilla_index = qubit_dict['x_ancillas'].index(ancilla_id)
                    ancilla_coord = x_measure_sorted[ancilla_index]
                elif ancilla_type == 'Z':
                    ancilla_index = qubit_dict['z_ancillas'].index(ancilla_id)
                    ancilla_coord = z_measure_sorted[ancilla_index]
                else:
                    continue  # Skip if not found
                
                # Get the neighbor ID based on the direction
                neighbor_id = get_neighbor_id(ancilla_coord, direction)
                
                if neighbor_id is not None:
                    # Determine if the ancilla is at the boundary (cut site)
                    is_boundary_left = (ancilla_coord[0] == cut-1 and (direction in ['Right'])) 
                    is_boundary_right = (ancilla_coord[0] == cut+5 and (direction in ['Left']))
                    # Determine the order of qubits in the CX gate based on ancilla type
                    if ancilla_type == 'Z':
                        # For Z ancillas: Data qubit is control, ancilla is target
                        pair = (neighbor_id, ancilla_id)
                    elif ancilla_type == 'X':
                        # For X ancillas: Ancilla is control, Data qubit is target
                        pair = (ancilla_id, neighbor_id)
                    else:
                        continue  # Skip if not applicable
                    
                    # Add the pair to the list for this direction
                    cx_pairs.append(pair)
                    
                    # Apply special depolarization for boundary ancillas
                    if is_boundary_left:
                        depol2_pairs.append(pair)
                    if is_boundary_right:
                        depol2_pairs.append(pair)

        
        if cx_pairs:
            # Prepare all CX gates for this direction on a single line
            cx_instructions = "CX "
            cx_instructions += " ".join([f"{q1} {q2}" for q1, q2 in cx_pairs]) + "\n"
            
            # Prepare all DEPOLARIZE2 gates for this direction on a single line
            depol2_instructions = ""
            if depol2_pairs:
                depol2_instructions += f"DEPOLARIZE2({p2}) " + " ".join([f"{q1} {q2}" for q1, q2 in depol2_pairs]) + "\n"
            non_boundary_pairs = [pair for pair in cx_pairs if pair not in depol2_pairs]
            if non_boundary_pairs:
                depol2_instructions += f"DEPOLARIZE2({p1}) " + " ".join([f"{q1} {q2}" for q1, q2 in non_boundary_pairs]) + "\n"
            
            # Combine instructions with TICK
            combined_instructions = f"{cx_instructions}{depol2_instructions}TICK\n"
            
            # Append to the circuit
            circuit += stim.Circuit(combined_instructions)

def apply_hadamard_ancillas_again(circuit, ancilla_ids, p1, ancilla_type='X'):
    """
    Apply Hadamard gates again to ancillas and introduce Depolarize1 noise.
    
    Args:
        circuit (stim.Circuit): The Stim circuit to append instructions to.
        ancilla_ids (list of int): List of ancilla qubit IDs.
        p1 (float): Depolarization probability for DEPOLARIZE1.
        ancilla_type (str, optional): Type of ancilla ('X' or 'Z'). Defaults to 'X'.
    """
    # Apply Hadamard gates
    hadamard_instruction = "H " + " ".join(map(str, ancilla_ids)) + "\n"
    
    # Apply Depolarize1 to each ancilla individually
    depolarize1_instructions = "\n".join([
        f"DEPOLARIZE1({p1}) {qubit_id}" 
        for qubit_id in ancilla_ids
    ]) + "\n"
    
    # Combine instructions with TICK
    combined_instructions = f"{hadamard_instruction}{depolarize1_instructions}TICK\n"
    
    # Append to the circuit
    circuit += stim.Circuit(combined_instructions)

def measure_and_reset(circuit, ancilla_qubits, p1, measurement_list):
    """
    Measure and reset ancillas, introducing depolarizing noise before and after measurement.
    Updates the measurement_list with the order of measurements.
    
    Args:
        circuit (stim.Circuit): The Stim circuit to append instructions to.
        ancilla_qubits (list of int): List of ancilla qubit IDs to measure and reset.
        p1 (float): Depolarization probability for DEPOLARIZE1.
        measurement_list (list of int): List to append measured qubit IDs.
    """
    # Apply Depolarize1 before measurement to each ancilla individually
    depolarize_before = f"X_ERROR({p1}) " + " ".join(map(str, ancilla_qubits)) + "\n"
    
    # Measure and reset ancillas
    measure_instruction = "MR " + " ".join(map(str, ancilla_qubits)) + "\n"
    
    # Apply Depolarize1 after measurement to each ancilla individually
    depolarize_after = f"X_ERROR({p1}) " + " ".join(map(str, ancilla_qubits)) + "\n"
    
    # Combine instructions with TICK
    combined_instructions = f"{depolarize_before}{measure_instruction}{depolarize_after}TICK\n"
    
    # Append to the circuit
    circuit += stim.Circuit(combined_instructions)
    
    # Update measurement list
    measurement_list.extend(ancilla_qubits)

def add_detectors(circuit, ancillas, measurement_order, round_index=0, total_mr=0):
    """
    Add DETECTOR instructions based on ancilla measurements.
    
    Args:
        circuit (stim.Circuit): The Stim circuit to append instructions to.
        ancillas (list of int): List of ancilla qubit IDs for which detectors are to be added.
        measurement_order (list of int): List of qubit IDs in the order they were measured.
        round_index (int, optional): Current stabilizer round index (default is 0).
        total_mr (int, optional): Total number of measurements in previous rounds (default is 0).
    """
    detectors = []
    total_measurements = len(measurement_order)
    
    for ancilla in ancillas:
        try:
            # Find the index of the ancilla in the measurement_order
            index = measurement_order.index(ancilla)
            
            # Calculate the backward reference
            backwards_reference = -(total_measurements - index)
            
            # For the first round, add a single rec[-n]
            if round_index == 0:
                detector_expression = f"rec[{backwards_reference}]"
            else:
                # For subsequent rounds, reference previous rounds as well
                detector_expression = f"rec[{backwards_reference}] rec[{backwards_reference - total_mr}]"
            
            # Create the DETECTOR instruction
            detector_instruction = f"DETECTOR {detector_expression}"
            
            detectors.append(detector_instruction)
        except ValueError:
            # Ancilla not found in measurement_order
            continue
    
    if detectors:
        # Combine all detectors into a single string separated by newlines
        detectors_instructions = "\n".join(detectors) + "\n"
        
        # Append DETECTOR instructions
        circuit += stim.Circuit(detectors_instructions)
    else:
        print("No detectors to add.")

def final_measurement(circuit, data_qubits, p1, measurement_list):
    """
    Perform the final measurement on data qubits with Depolarize noise.
    
    Args:
        circuit (stim.Circuit): The Stim circuit to append instructions to.
        data_qubits (list of int): List of data qubit IDs to measure.
        p1 (float): Depolarization probability for DEPOLARIZE1.
        measurement_list (list of int): List to append measured qubit IDs.
    """
    # Append TICK
    circuit += stim.Circuit("TICK\n")
    
    # Apply DEPOLARIZE1 before measurement
    depolarize_before = f"X_ERROR({p1}) " + " ".join(map(str, data_qubits)) + "\n"
    circuit += stim.Circuit(depolarize_before)
    
    # Measure data qubits
    measure_instruction = "M " + " ".join(map(str, data_qubits)) + "\n"
    circuit += stim.Circuit(measure_instruction)
    
    # Apply DEPOLARIZE1 after measurement
    #depolarize_after = f"X_ERROR({p1}) " + " ".join(map(str, data_qubits)) + "\n"
    #circuit += stim.Circuit(depolarize_after)
    
    # Append TICK
    circuit += stim.Circuit("TICK\n")
    
    # Update measurement list with data qubits measurements
    measurement_list.extend(data_qubits)

def add_final_detectors(circuit, z_ancilla_to_data_qubits, measurement_list, data_qubits_sorted, vertical=False):
    """
    Add final DETECTOR instructions based on the last round's Z ancilla measurements and the final data qubit measurements.
    
    Args:
        circuit (stim.Circuit): The Stim circuit to append instructions to.
        z_ancilla_to_data_qubits (dict): Mapping from Z ancilla qubit IDs to their neighboring data qubit IDs.
        measurement_list (list of int): List of qubit IDs in the order they were measured.
        data_qubits_sorted (list of tuples): List of coordinates for data qubits (for mapping, if needed).
        vertical (bool, optional): Whether the circuit is inverted. Defaults to False.
    """
    detectors = []
    total_measurements = len(measurement_list)
    
    for z_ancilla, data_qubits in z_ancilla_to_data_qubits.items():
        try:
            # Find the index of the Z ancilla measurement (last round)
            index_z = measurement_list.index(z_ancilla)
            m_z = -(total_measurements - index_z)
            
            # Find backward references for the associated data qubits (final measurements)
            m_data = [-(total_measurements - measurement_list.index(q)) for q in data_qubits]
            
            # Create the DETECTOR instruction
            detector_expression = f"rec[{m_z}] " + " ".join([f"rec[{m}]" for m in m_data])
            detector_instruction = f"DETECTOR {detector_expression}"
            
            detectors.append(detector_instruction)
        except ValueError as e:
            # Handle cases where the ancilla or data qubits are not found
            print(f"Error adding final detector for Z ancilla {z_ancilla}: {e}")
            continue
    
    if detectors:
        # Combine all detectors into a single string separated by newlines
        detectors_instructions = "\n".join(detectors) + "\n"
        
        # Append DETECTOR instructions
        circuit += stim.Circuit(detectors_instructions)
    else:
        print("No final detectors to add.")

def append_logical_observable(circuit, data_qubits_sorted, qubit_dict, measurement_list, vertical=False):
    """
    Appends the logical observable at the end of the Stim circuit by including all data qubits in the last row
    or first column if inverted.
    
    Args:
        circuit (stim.Circuit): The Stim circuit to append instructions to.
        data_qubits_sorted (list of tuples): List of coordinates for data qubits, sorted row-wise.
        qubit_dict (dict): Dictionary containing lists of qubit IDs categorized as 'data_qubits', 'x_ancillas', 'z_ancillas'.
        measurement_list (list of int): List of qubit IDs in the order they were measured.
        vertical (bool, optional): Whether the circuit is inverted. Defaults to False.
    """
    # Step 1: Identify the maximum or minimum y-coordinate or x-coordinate based on inversion
    if not vertical:
        # Logical observable is the data qubits in the last row (maximum y)
        target_y = max(y for (x, y) in data_qubits_sorted)
        last_qubits = [
            qubit_id for qubit_id, (x, y) in zip(qubit_dict['data_qubits'], data_qubits_sorted)
            if y == target_y
        ]
    else:
        # Logical observable is the data qubits in the first column (minimum x)
        target_x = min(x for (x, y) in data_qubits_sorted)
        last_qubits = [
            qubit_id for qubit_id, (x, y) in zip(qubit_dict['data_qubits'], data_qubits_sorted)
            if x == target_x
        ]
    
    # Step 2: Determine the measurement references for these qubits
    recs = []
    for qubit_id in last_qubits:
        try:
            # Find the index of the qubit in the measurement_list
            idx = measurement_list.index(qubit_id)
            # Calculate the reference: rec[-n] where n = total_measurements - idx
            ref = -(len(measurement_list) - idx)
            recs.append(f"rec[{ref}]")
        except ValueError:
            # Handle the case where the qubit ID is not found in the measurement_list
            continue
    
    # Step 3: Create the recs string
    recs_str = " ".join(recs)
    
    # Step 4: Append a TICK to separate the logical observable from previous operations
    circuit += stim.Circuit("TICK\n")
    
    # Step 5: Append the OBSERVABLE_INCLUDE instruction with the gathered references
    if recs_str:
        circuit += stim.Circuit(f"OBSERVABLE_INCLUDE(0) {recs_str}\n")
    else:
        print("No logical observable qubits found to include.")

def create_circuit(n, m, cut, num_rounds=1, p1=0.02, p2=0.05, vertical=False, show_grid=False):
    """
    Create the complete Stim circuit with specified number of rounds, final measurement, and inversion option.
    
    Args:
        n (int): Width of the lattice.
        m (int): Height of the lattice.
        cut (int): The x-coordinate where the lattice is cut.
        num_rounds (int, optional): Number of stabilizer rounds to perform. Defaults to 1.
        p1 (float, optional): Depolarization probability for regular operations. Defaults to 0.02.
        p2 (float, optional): Elevated depolarization probability for boundary operations. Defaults to 0.05.
        vertical (bool, optional): Whether to vertical the roles of X and Z ancillas. Defaults to False.
    
    Returns:
        stim.Circuit: The complete Stim circuit.
    """



    if show_grid: 
        data_qubits_sorted, x_measure_sorted, z_measure_sorted, x_observable, z_observable, qubit_dict = plot_unrotated_surface_code_grid(
        n, m, cut,annotate=True, cut_gap=5, vertical=vertical)
    else:
        data_qubits_sorted, x_measure_sorted, z_measure_sorted, x_observable, z_observable, qubit_dict = generate_unrotated_surface_code_grid(
        n, m, cut, cut_gap=5, vertical=vertical)
    # Initialize an empty circuit
    circuit = stim.Circuit()
    
    # Step 1: Assign coordinates to qubits (only once in the first round)
    create_coords(circuit, qubit_dict, data_qubits_sorted, x_measure_sorted, z_measure_sorted)
    
    # Step 2: Reset all qubits (only once in the first round)
    all_qubits = qubit_dict['data_qubits'] + qubit_dict['x_ancillas'] + qubit_dict['z_ancillas']
    initial_reset(circuit, all_qubits)
    
    # Create a mapping from data qubit coordinates to qubit IDs
    data_qubit_coords_to_id = {
        tuple(coord): qubit_id
        for qubit_id, coord in zip(qubit_dict['data_qubits'], data_qubits_sorted)
    }
    
    # Create a mapping from Z ancilla to their neighboring data qubits with boundary handling
    z_ancilla_to_data_qubits = {}
    for z_ancilla, coord in zip(qubit_dict['z_ancillas'], z_measure_sorted):
        x, y = coord
        neighbor_coords = [
            (x, y + 1),  # Up
            (x + 1, y),  # Right
            (x - 1, y),  # Left
            (x, y - 1)   # Down
        ]
        data_qubits = []
        
        # Define the order of directions corresponding to neighbor_coords
        directions = ['Up', 'Right', 'Left', 'Down']
        
        for direction, nc in zip(directions, neighbor_coords):
            # Apply the boundary handling logic based on the current x-coordinate and direction
            if x == cut - 1:
                if direction == 'Right':
                    neighbor_coord = (x + 6, y)  # Adjacent to the right side across the cut
                else:
                    neighbors = {
                        'Up': (x, y - 1),
                        'Left': (x - 1, y),
                        'Down': (x, y + 1)
                    }
                    neighbor_coord = neighbors.get(direction, None)
            elif x == cut + 5:
                if direction == 'Left':
                    neighbor_coord = (x - 6, y)  # Adjacent to the left side across the cut
                else:
                    neighbors = {
                        'Up': (x, y - 1),
                        'Right': (x + 1, y),
                        'Down': (x, y + 1)
                    }
                    neighbor_coord = neighbors.get(direction, None)
            else:
                # Ancillas not adjacent to the cut use standard neighbor coordinates
                if direction == 'Right':
                    neighbor_coord = (x + 1, y)
                elif direction == 'Left':
                    neighbor_coord = (x - 1, y)
                else:
                    neighbors = {
                        'Up': (x, y - 1),
                        'Down': (x, y + 1)
                    }
                    neighbor_coord = neighbors.get(direction, None)
            
            # If neighbor_coord is defined, attempt to retrieve the corresponding data qubit ID
            if neighbor_coord:
                data_qubit_id = data_qubit_coords_to_id.get(tuple(neighbor_coord))
                if data_qubit_id is not None:
                    data_qubits.append(data_qubit_id)
        
        # Assign the list of neighboring data qubits to the Z ancilla
        z_ancilla_to_data_qubits[z_ancilla] = data_qubits    
    
    # Perform stabilizer rounds
    for round_idx in range(num_rounds):
        measurement_list = []

        # Step 3: Apply Hadamard gates to X ancillas and Depolarize1
        ancilla_type = 'X'
        begin_round(circuit, all_qubits,p1,round_idx)
        apply_hadamard_ancillas(circuit, qubit_dict[f'{ancilla_type.lower()}_ancillas'], p1, ancilla_type=ancilla_type)
        
        # Step 4: Stabilizer round with Depolarize2 and boundary handling
        stabilizer_round(
            circuit, qubit_dict, data_qubits_sorted, x_measure_sorted, z_measure_sorted, 
            p1, p2, cut, spacing=5, vertical=vertical
        )
        
        # Step 5: Apply Hadamard gates again to X ancillas and Depolarize1
        apply_hadamard_ancillas_again(circuit, qubit_dict[f'{ancilla_type.lower()}_ancillas'], p1, ancilla_type=ancilla_type)
        
        # Step 6: Measure and reset ancillas, capturing measurement indices
        ancilla_qubits = qubit_dict['x_ancillas'] + qubit_dict['z_ancillas']
        if vertical:
            ancilla_qubits = qubit_dict['z_ancillas'] + qubit_dict['x_ancillas']
        measure_and_reset(circuit, ancilla_qubits, p1, measurement_list)
        
        # Step 7: Add Detectors
        if round_idx == 0:
            # Only add Z detectors in the first round
            add_detectors(circuit, qubit_dict['z_ancillas'], measurement_list, round_index=round_idx, total_mr=0)
        else:
            # Add both Z and X detectors in subsequent rounds
            add_detectors(circuit, qubit_dict['z_ancillas'], measurement_list, round_index=round_idx, total_mr=len(qubit_dict['x_ancillas']) + len(qubit_dict['z_ancillas']))
            add_detectors(circuit, qubit_dict['x_ancillas'], measurement_list, round_index=round_idx, total_mr=len(qubit_dict['x_ancillas']) + len(qubit_dict['z_ancillas']))
    
    # Perform Final Measurement
    final_measurement(circuit, qubit_dict['data_qubits'], p1, measurement_list)
    
    # Add Final Detectors
    add_final_detectors(circuit, z_ancilla_to_data_qubits, measurement_list, data_qubits_sorted, vertical=vertical)
    
    # Append Logical Observable at the End
    append_logical_observable(circuit, data_qubits_sorted, qubit_dict, measurement_list, vertical=vertical)
    
    return circuit


