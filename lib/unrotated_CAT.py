import matplotlib.pyplot as plt
import stim

def generate_unrotated_surface_code_grid(d_x, d_y, cut=None, cut_gap=5, vertical=False):
    """
    Generate coordinates for data qubits, X ancillas, Z ancillas, and CAT copies in an unrotated surface code grid.

    Args:
        d_x (int): Width of the grid in the x-direction.
        d_y (int): Height of the grid in the y-direction.
        cut (int, optional): The x-coordinate where the lattice is cut. Defaults to None.
        cut_gap (int, optional): The gap introduced at the cut. Defaults to 5.
        vertical (bool, optional): Whether to invert the roles of X and Z ancillas. Defaults to False.

    Returns:
        tuple: Sorted lists of data qubits, X ancillas, Z ancillas, X CAT copies, Z CAT copies, observables, and a qubit dictionary.
    """
    data_coords = set()
    x_measure_coords = set()
    z_measure_coords = set()
    x_observable = []
    z_observable = []
    x_cat_copies = []
    z_cat_copies = []

    # Fixed offset to generate unique and consistent copy IDs
    fixed_offset = 50000

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
    data_qubit_ids = [int(f'4{i+1}') for i in range(len(data_qubits_sorted))]  # e.g., 41, 42, ...
    x_original_ancilla_ids = [int(f'3{i+1}') for i in range(len(x_measure_sorted))]  # e.g., 31, 32, ...
    z_original_ancilla_ids = [int(f'2{i+1}') for i in range(len(z_measure_sorted))]  # e.g., 21, 22, ...

    # If vertical inversion is requested, swap X and Z ancilla roles
    if vertical:
        x_measure_sorted, z_measure_sorted = z_measure_sorted, x_measure_sorted
        x_original_ancilla_ids, z_original_ancilla_ids = z_original_ancilla_ids, x_original_ancilla_ids
        x_observable, z_observable = z_observable, x_observable

    # Identify ancillas on the boundary (cut - 1 and cut + cut_gap)
    if cut is not None:
        boundary_ancillas = []
        boundary_x = cut - 1  # Corrected as per your instruction
        boundary_gap_x = cut + cut_gap

        # Identify X ancillas on boundary
        for ancilla_id, coord in zip(x_original_ancilla_ids, x_measure_sorted):
            if coord[0] == boundary_x or coord[0] == boundary_gap_x:
                boundary_ancillas.append(('X', ancilla_id, coord))

        # Identify Z ancillas on boundary
        for ancilla_id, coord in zip(z_original_ancilla_ids, z_measure_sorted):
            if coord[0] == boundary_x or coord[0] == boundary_gap_x:
                boundary_ancillas.append(('Z', ancilla_id, coord))

        # Create copies with corrected shift of ±5
        for ancilla_type, ancilla_id, coord in boundary_ancillas:
            original_x, original_y = coord
            if ancilla_type == 'X' and original_x == boundary_x:
                copy_x = original_x + 5
            elif ancilla_type == 'X' and original_x == boundary_gap_x:
                copy_x = original_x - 5
            elif ancilla_type == 'Z' and original_x == boundary_x:
                copy_x = original_x + 5
            elif ancilla_type == 'Z' and original_x == boundary_gap_x:
                copy_x = original_x - 5
            else:
                continue  # Skip if not on exact boundary

            copy_coord = (copy_x, original_y)
            copy_id = ancilla_id + fixed_offset  # Ensure consistent and unique copy IDs

            # Append to respective CAT copy lists
            if ancilla_type == 'X':
                x_cat_copies.append(copy_id)
                x_measure_sorted.append(copy_coord)
            elif ancilla_type == 'Z':
                z_cat_copies.append(copy_id)
                z_measure_sorted.append(copy_coord)

    # Create the dictionary categorizing qubits
    qubit_dict = {
        'data_qubits': data_qubit_ids,
        'x_original_ancillas': x_original_ancilla_ids,
        'z_original_ancillas': z_original_ancilla_ids,
        'x_cat_copies': x_cat_copies,
        'z_cat_copies': z_cat_copies
    }

    # Return sorted lists and qubit_dict
    return (data_qubits_sorted, x_measure_sorted, z_measure_sorted, 
            x_observable, z_observable, x_cat_copies, z_cat_copies, qubit_dict)

def plot_unrotated_surface_code_grid(d_x, d_y, cut=None, cut_gap=5, annotate=True, vertical=False):
    """
    Plot the unrotated surface code grid with Vertical inversion and CAT functionality.

    Args:
        d_x (int): Width of the grid in the x-direction.
        d_y (int): Height of the grid in the y-direction.
        cut (int, optional): The x-coordinate where the lattice is cut. Defaults to None.
        cut_gap (int, optional): The gap introduced at the cut. Defaults to 5.
        annotate (bool, optional): Whether to annotate qubits with their IDs. Defaults to True.
        vertical (bool, optional): Whether to invert the roles of X and Z ancillas. Defaults to False.

    Returns:
        tuple: Sorted lists of data qubits, X ancillas, Z ancillas, X CAT copies, Z CAT copies, observables, and a qubit dictionary.
    """
    # Generate grid coordinates and qubit dictionary
    (data_qubits_sorted, x_measure_sorted, z_measure_sorted, 
     x_observable, z_observable, x_cat_copies, z_cat_copies, qubit_dict) = generate_unrotated_surface_code_grid(
        d_x, d_y, cut, cut_gap, vertical
    )

    # Extract coordinates for plotting
    data_x = [x for (x, y) in data_qubits_sorted]
    data_y = [y for (x, y) in data_qubits_sorted]

    # Original Ancillas
    x_original_ancillas_sorted = x_measure_sorted[:len(qubit_dict['x_original_ancillas'])]
    x_original_x = [x for (x, y) in x_original_ancillas_sorted]
    x_original_y = [y for (x, y) in x_original_ancillas_sorted]

    z_original_ancillas_sorted = z_measure_sorted[:len(qubit_dict['z_original_ancillas'])]
    z_original_x = [x for (x, y) in z_original_ancillas_sorted]
    z_original_y = [y for (x, y) in z_original_ancillas_sorted]

    # CAT Copies
    x_cat_copy_coords = x_measure_sorted[len(qubit_dict['x_original_ancillas']):]
    z_cat_copy_coords = z_measure_sorted[len(qubit_dict['z_original_ancillas']):]

    plt.figure(figsize=(12, 10))

    # Plot Data Qubits with reduced opacity
    plt.scatter(data_x, data_y, c='black', marker='o', label='Data Qubits', alpha=0.2, zorder=1)
    # Plot X Original Ancillas with reduced opacity
    plt.scatter(x_original_x, x_original_y, c='red', marker='s', label='X Ancillas', alpha=0.2, zorder=1)
    # Plot Z Original Ancillas with reduced opacity
    plt.scatter(z_original_x, z_original_y, c='blue', marker='^', label='Z Ancillas', alpha=0.2, zorder=1)

    # Plot X CAT Copies with distinct markers
    if x_cat_copy_coords:
        copy_x = [x for (x, y) in x_cat_copy_coords]
        copy_y = [y for (x, y) in x_cat_copy_coords]
        plt.scatter(copy_x, copy_y, c='magenta', marker='D', label='X CAT Copies', alpha=0.6, zorder=2)

    # Plot Z CAT Copies with distinct markers
    if z_cat_copy_coords:
        copy_x = [x for (x, y) in z_cat_copy_coords]
        copy_y = [y for (x, y) in z_cat_copy_coords]
        plt.scatter(copy_x, copy_y, c='cyan', marker='D', label='Z CAT Copies', alpha=0.6, zorder=2)

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
        # Annotate X Original Ancillas
        for qubit_id, (x, y) in zip(qubit_dict['x_original_ancillas'], x_original_ancillas_sorted):
            plt.text(x, y, str(qubit_id), fontsize=8, ha='center', va='center', color='red', zorder=3)
        # Annotate Z Original Ancillas
        for qubit_id, (x, y) in zip(qubit_dict['z_original_ancillas'], z_original_ancillas_sorted):
            plt.text(x, y, str(qubit_id), fontsize=8, ha='center', va='center', color='blue', zorder=3)
        # Annotate X CAT Copies
        for copy_id, (x, y) in zip(qubit_dict['x_cat_copies'], x_cat_copy_coords):
            plt.text(x, y, str(copy_id), fontsize=8, ha='center', va='center', color='magenta', zorder=4)
        # Annotate Z CAT Copies
        for copy_id, (x, y) in zip(qubit_dict['z_cat_copies'], z_cat_copy_coords):
            plt.text(x, y, str(copy_id), fontsize=8, ha='center', va='center', color='cyan', zorder=4)

    # Adjust legend to prevent overlap
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

    # Construct the title
    title = f"{'CAT ' if x_cat_copies or z_cat_copies else ''}{'Vertical ' if vertical else ''}Unrotated Surface Code Grid (Width={d_x}, Height={d_y}"
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
    return (data_qubits_sorted, x_measure_sorted, z_measure_sorted, 
            x_observable, z_observable, x_cat_copies, z_cat_copies, qubit_dict)

def create_coords(circuit, qubit_dict, data_qubits_sorted, x_measure_sorted, z_measure_sorted, vertical=False):
    """
    Assign spatial coordinates to each qubit using QUBIT_COORDS.

    Args:
        circuit (stim.Circuit): The Stim circuit to append instructions to.
        qubit_dict (dict): Dictionary containing lists of qubit IDs categorized as 'data_qubits', 'x_original_ancillas', 'z_original_ancillas', 'x_cat_copies', 'z_cat_copies'.
        data_qubits_sorted (list of tuples): List of coordinates for data qubits.
        x_measure_sorted (list of tuples): List of coordinates for X ancillas and X CAT copies.
        z_measure_sorted (list of tuples): List of coordinates for Z ancillas and Z CAT copies.
        vertical (bool, optional): Whether X and Z ancillas have been inverted. Defaults to False.
    """
    # Create a mapping from qubit IDs to their coordinates
    qubit_id_to_coords = {}

    # Map data qubits
    for qubit_id, coord in zip(qubit_dict['data_qubits'], data_qubits_sorted):
        qubit_id_to_coords[qubit_id] = coord

    # Map X original ancillas
    for qubit_id, coord in zip(qubit_dict['x_original_ancillas'], x_measure_sorted[:len(qubit_dict['x_original_ancillas'])]):
        qubit_id_to_coords[qubit_id] = coord

    # Map Z original ancillas
    for qubit_id, coord in zip(qubit_dict['z_original_ancillas'], z_measure_sorted[:len(qubit_dict['z_original_ancillas'])]):
        qubit_id_to_coords[qubit_id] = coord

    # Map X CAT copies
    for qubit_id, coord in zip(qubit_dict['x_cat_copies'], x_measure_sorted[len(qubit_dict['x_original_ancillas']):]):
        qubit_id_to_coords[qubit_id] = coord

    # Map Z CAT copies
    for qubit_id, coord in zip(qubit_dict['z_cat_copies'], z_measure_sorted[len(qubit_dict['z_original_ancillas']):]):
        qubit_id_to_coords[qubit_id] = coord

    # Create QUBIT_COORDS instructions
    coords_instructions = "\n".join([
        f"QUBIT_COORDS({x}, {y}) {qubit_id}"
        for qubit_id, (x, y) in qubit_id_to_coords.items()
        if (x, y) is not None
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
    if not all_qubits:
        return  # No qubits to reset

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
    if not ancilla_ids:
        return  # No ancillas to process

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

def apply_hadamard_ancillas_again(circuit, ancilla_ids, p1, ancilla_type='X'):
    """
    Apply Hadamard gates again to ancillas and introduce Depolarize1 noise.

    Args:
        circuit (stim.Circuit): The Stim circuit to append instructions to.
        ancilla_ids (list of int): List of ancilla qubit IDs.
        p1 (float): Depolarization probability for DEPOLARIZE1.
        ancilla_type (str, optional): Type of ancilla ('X' or 'Z'). Defaults to 'X'.
    """
    if not ancilla_ids:
        return  # No ancillas to process

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

def apply_cat_operations(circuit, qubit_dict, p2,p1):
    """
    Apply Hadamard gates to CAT copies and perform CX operations between copies and originals with DEPOLARIZE2(p2) noise.

    Args:
        circuit (stim.Circuit): The Stim circuit to append instructions to.
        qubit_dict (dict): Dictionary containing lists of qubit IDs categorized as 'data_qubits', 'x_original_ancillas', 'z_original_ancillas', 'x_cat_copies', 'z_cat_copies'.
        p2 (float): Depolarization probability for DEPOLARIZE2 during CX operations.
    """
    x_cat_copies = qubit_dict.get('x_cat_copies', [])
    z_cat_copies = qubit_dict.get('z_cat_copies', [])

    # Apply Hadamard gates to X CAT copies
    if x_cat_copies:
        hadamard_instruction_x = "H " + " ".join(map(str, x_cat_copies)) + "\n"
        circuit += stim.Circuit(hadamard_instruction_x)

    # Apply Hadamard gates to Z CAT copies
    if z_cat_copies:
        hadamard_instruction_z = "H " + " ".join(map(str, z_cat_copies)) + "\n"
        circuit += stim.Circuit(hadamard_instruction_z)

    circuit.append("DEPOLARIZE1", x_cat_copies,p1)
    circuit.append("DEPOLARIZE1", z_cat_copies,p1)

    # Create CX pairs between copies and originals
    cx_pairs = []

    # X CAT copies
    for copy_id in x_cat_copies:
        original_id = copy_id - 50000  # Retrieve original_id using fixed offset
        cx_pairs.append((copy_id, original_id))

    # Z CAT copies
    for copy_id in z_cat_copies:
        original_id = copy_id - 50000  # Retrieve original_id using fixed offset
        cx_pairs.append((copy_id, original_id))

    if cx_pairs:
        cx_instructions = "CX " + " ".join([f"{q1} {q2}" for q1, q2 in cx_pairs]) + "\n"
        depol2_instructions = f"DEPOLARIZE2({p2}) " + " ".join([f"{q1} {q2}" for q1, q2 in cx_pairs]) + "\n"

        # Combine instructions with TICK
        combined_instructions = f"{cx_instructions}{depol2_instructions}\n"

        # Append to the circuit
        circuit += stim.Circuit(combined_instructions)

    # After all CAT operations, apply DEPOLARIZE1 to all data qubits and original ancillas
    original_ancillas_in_pairs = [pair[1] for pair in cx_pairs]  # Extract original ancilla IDs
    qubits_for_depolarize1 = (
        qubit_dict['data_qubits'] + 
        [x for x in qubit_dict['x_original_ancillas'] if x not in original_ancillas_in_pairs] + 
        [z for z in qubit_dict['z_original_ancillas'] if z not in original_ancillas_in_pairs]
    )
    
    if qubits_for_depolarize1:
        depolarize1_instruction = f"DEPOLARIZE1({p1}) " + " ".join(map(str, qubits_for_depolarize1)) + "\n"
        circuit += stim.Circuit(depolarize1_instruction)
        circuit += stim.Circuit("TICK\n")



def stabilizer_round(circuit, qubit_dict, data_qubits_sorted, x_measure_sorted, z_measure_sorted, p1, p2, cut, spacing=1):
    """
    Perform stabilizer interactions with depolarizing noise.

    Args:
        circuit (stim.Circuit): The Stim circuit to append instructions to.
        qubit_dict (dict): Dictionary containing lists of qubit IDs categorized as 'data_qubits', 'x_original_ancillas', 'z_original_ancillas', 'x_cat_copies', 'z_cat_copies'.
        data_qubits_sorted (list of tuples): List of coordinates for data qubits.
        x_measure_sorted (list of tuples): List of coordinates for X ancillas and X CAT copies.
        z_measure_sorted (list of tuples): List of coordinates for Z ancillas and Z CAT copies.
        p1 (float): Depolarization probability for DEPOLARIZE2.
        p2 (float): Depolarization probability for CAT CX operations.
        cut (int): The x-coordinate where the lattice is cut.
        spacing (int, optional): The spacing between qubits (default is 1).
    """
    # Define the order of directions for interactions
    interaction_order = ['Up', 'Right', 'Left', 'Down']

    # Create a mapping from data qubit coordinates to qubit IDs for data qubits
    data_qubit_coords_to_id = {
        coord: qubit_id
        for qubit_id, coord in zip(qubit_dict['data_qubits'], data_qubits_sorted)
    }

    # No special boundary handling in CAT mode; use standard neighbor rules
    def get_neighbor_id(ancilla_coord, direction):
        x, y = ancilla_coord

        if direction == 'Right':
            neighbor_coord = (x + 1, y)
        elif direction == 'Left':
            neighbor_coord = (x - 1, y)
        elif direction == 'Up':
            neighbor_coord = (x, y - 1)
        elif direction == 'Down':
            neighbor_coord = (x, y + 1)
        else:
            neighbor_coord = None

        return data_qubit_coords_to_id.get(neighbor_coord, None)

    # Iterate over each direction
    for direction in interaction_order:
        cx_pairs = []

        # Iterate over each ancilla type and their respective lists
        ancilla_types = [
            ('X', qubit_dict['x_original_ancillas']),
            ('Z', qubit_dict['z_original_ancillas']),
            ('X', qubit_dict['x_cat_copies']),
            ('Z', qubit_dict['z_cat_copies'])
        ]

        for ancilla_type, ancilla_list in ancilla_types:
            for ancilla_id in ancilla_list:
                # Retrieve the coordinates of the current ancilla based on its type
                if ancilla_type == 'X':
                    if ancilla_id in qubit_dict['x_original_ancillas']:
                        ancilla_index = qubit_dict['x_original_ancillas'].index(ancilla_id)
                        ancilla_coord = x_measure_sorted[ancilla_index]
                    else:
                        ancilla_index = qubit_dict['x_cat_copies'].index(ancilla_id)
                        ancilla_coord = x_measure_sorted[len(qubit_dict['x_original_ancillas']) + ancilla_index]
                elif ancilla_type == 'Z':
                    if ancilla_id in qubit_dict['z_original_ancillas']:
                        ancilla_index = qubit_dict['z_original_ancillas'].index(ancilla_id)
                        ancilla_coord = z_measure_sorted[ancilla_index]
                    else:
                        ancilla_index = qubit_dict['z_cat_copies'].index(ancilla_id)
                        ancilla_coord = z_measure_sorted[len(qubit_dict['z_original_ancillas']) + ancilla_index]
                else:
                    continue  # Skip if not found

                # Get the neighbor ID based on the direction
                neighbor_id = get_neighbor_id(ancilla_coord, direction)

                if neighbor_id is not None:
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

        if cx_pairs:
            # Prepare all CX gates for this direction on a single line with DEPOLARIZE2(p1) noise
            cx_instructions = "CX " + " ".join([f"{q1} {q2}" for q1, q2 in cx_pairs]) + "\n"
            depol2_instructions = f"DEPOLARIZE2({p1}) " + " ".join([f"{q1} {q2}" for q1, q2 in cx_pairs]) + "\n"

            # Combine instructions with TICK
            combined_instructions = f"{cx_instructions}{depol2_instructions}TICK\n"

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
    if not ancilla_qubits:
        return  # No ancillas to process

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

def add_detectors(circuit, ancillas, measurement_order, ancilla_copies, round_index=0, total_mr=0):
    """
    Add DETECTOR instructions based on ancilla and their CAT copy measurements.

    Args:
        circuit (stim.Circuit): The Stim circuit to append instructions to.
        ancillas (list of int): List of original ancilla qubit IDs.
        measurement_order (list of int): List of qubit IDs in the order they were measured.
        ancilla_copies (dict): Mapping from original ancilla IDs to their CAT copy IDs.
        round_index (int, optional): Current stabilizer round index (default is 0).
        total_mr (int, optional): Total number of measurements in each round (excluding data qubits).
    """
    detectors = []
    total_measurements = len(measurement_order)

    for ancilla in ancillas:
        try:
            # Find the index of the ancilla in the measurement_order
            index = measurement_order.index(ancilla)
            m_ancilla = -(total_measurements - index)

            # Find the corresponding copy
            copy_id = ancilla_copies.get(ancilla, None)
            if copy_id is not None:
                index_copy = measurement_order.index(copy_id)
                m_copy = -(total_measurements - index_copy)
            else:
                m_copy = None

            # Create the DETECTOR expression
            if round_index == 0:
                # First round: only current measurements
                if m_copy is not None:
                    detector_expression = f"rec[{m_ancilla}] rec[{m_copy}]"
                else:
                    detector_expression = f"rec[{m_ancilla}]"
            else:
                # Subsequent rounds: include previous round measurements
                if m_copy is not None:
                    m_ancilla_prev = m_ancilla - total_mr
                    m_copy_prev = m_copy - total_mr
                    detector_expression = f"rec[{m_ancilla}] rec[{m_ancilla_prev}] rec[{m_copy}] rec[{m_copy_prev}]"
                else:
                    m_ancilla_prev = m_ancilla - total_mr
                    detector_expression = f"rec[{m_ancilla}] rec[{m_ancilla_prev}]"

            # Create the DETECTOR instruction
            detector_instruction = f"DETECTOR {detector_expression}"

            detectors.append(detector_instruction)
        except ValueError:
            # Ancilla or its copy not found in measurement_order
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
    if not data_qubits:
        return  # No data qubits to process

    # Append TICK
    circuit += stim.Circuit("TICK\n")

    # Apply DEPOLARIZE1 before measurement
    depolarize_before = f"X_ERROR({p1}) " + " ".join(map(str, data_qubits)) + "\n"
    circuit += stim.Circuit(depolarize_before)

    # Measure data qubits
    measure_instruction = "M " + " ".join(map(str, data_qubits)) + "\n"
    circuit += stim.Circuit(measure_instruction)

    # Apply Depolarize1 after measurement
    #depolarize_after = f"X_ERROR({p1}) " + " ".join(map(str, data_qubits)) + "\n"
    #circuit += stim.Circuit(depolarize_after)

    # Append TICK
    circuit += stim.Circuit("TICK\n")

    # Update measurement list with data qubits measurements
    measurement_list.extend(data_qubits)

def add_final_detectors(
    circuit, 
    z_ancilla_to_data_qubits, 
    measurement_list, 
    data_qubits_sorted, 
    ancilla_copies
):
    """
    Add final DETECTOR instructions based on the last round's Z ancilla measurements and the final data qubit measurements.
    Includes data qubits associated with both the Z ancilla and its CAT copy.

    Args:
        circuit (stim.Circuit): The Stim circuit to append instructions to.
        z_ancilla_to_data_qubits (dict): Mapping from Z ancilla qubit IDs to their neighboring data qubit IDs.
        measurement_list (list of int): List of qubit IDs in the order they were measured.
        data_qubits_sorted (list of tuples): List of coordinates for data qubits (for mapping, if needed).
        ancilla_copies (dict): Mapping from original Z ancilla IDs to their CAT copy IDs.
    """
    detectors = []
    total_measurements = len(measurement_list)

    for z_ancilla, data_qubits in z_ancilla_to_data_qubits.items():
        if str(z_ancilla)[0] not in ['5']:
            try:
                # Find the index of the Z ancilla measurement
                index_z = measurement_list.index(z_ancilla)
                m_z = -(total_measurements - index_z)

                # Initialize detector expression with the Z ancilla measurement
                detector_expression = f"rec[{m_z}]"

                # Check if the Z ancilla has a CAT copy
                copy_id = ancilla_copies.get(z_ancilla, None)
                if copy_id is not None:
                    try:
                        # Find the index of the CAT copy measurement
                        index_copy = measurement_list.index(copy_id)
                        m_copy = -(total_measurements - index_copy)
                        # Append the CAT copy measurement to the detector expression
                        detector_expression += f" rec[{m_copy}]"
                        
                        # Retrieve data qubits associated with the CAT copy
                        copy_data_qubits = z_ancilla_to_data_qubits.get(copy_id, [])
                    except ValueError:
                        # CAT copy measurement not found
                        print(f"Warning: CAT copy {copy_id} for Z ancilla {z_ancilla} not found in measurement list.")
                        copy_data_qubits = []
                else:
                    copy_data_qubits = []

                # Include data qubits measurements for the original Z ancilla
                m_data_original = []
                for q in data_qubits:
                    if q in measurement_list:
                        m = -(total_measurements - measurement_list.index(q))
                        m_data_original.append(f"rec[{m}]")

                # Include data qubits measurements for the CAT copy, if it exists
                m_data_copy = []
                for q in copy_data_qubits:
                    if q in measurement_list:
                        m = -(total_measurements - measurement_list.index(q))
                        m_data_copy.append(f"rec[{m}]")

                # Combine all data qubit measurements
                data_expression = " ".join(m_data_original + m_data_copy)

                # Create the DETECTOR instruction
                if data_expression:
                    detector_instruction = f"DETECTOR {detector_expression} {data_expression}"
                else:
                    detector_instruction = f"DETECTOR {detector_expression}"

                detectors.append(detector_instruction)
            except ValueError:
                # Handle cases where the Z ancilla is not found in measurement_list
                print(f"Error: Z ancilla {z_ancilla} not found in measurement list.")
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
    Appends the logical observable at the end of the Stim circuit by including all data qubits in the last row or first column.

    Args:
        circuit (stim.Circuit): The Stim circuit to append instructions to.
        data_qubits_sorted (list of tuples): List of coordinates for data qubits, sorted row-wise.
        qubit_dict (dict): Dictionary containing lists of qubit IDs categorized as 'data_qubits', 'x_original_ancillas', 'z_original_ancillas', 'x_cat_copies', 'z_cat_copies'.
        measurement_list (list of int): List of qubit IDs in the order they were measured.
        vertical (bool, optional): Whether the circuit was inverted. Defaults to False.
    """
    # Step 1: Identify the target qubits based on inversion
    if not vertical:
        # Logical observable is the data qubits in the last row (maximum y)
        target_y = max(y for (x, y) in data_qubits_sorted)
        target_qubits = [
            qubit_id for qubit_id, (x, y) in zip(qubit_dict['data_qubits'], data_qubits_sorted)
            if y == target_y
        ]
    else:
        # Logical observable is the data qubits in the first column (minimum x)
        target_x = min(x for (x, y) in data_qubits_sorted)
        target_qubits = [
            qubit_id for qubit_id, (x, y) in zip(qubit_dict['data_qubits'], data_qubits_sorted)
            if x == target_x
        ]

    # Step 2: Determine the measurement references for these qubits
    recs = []
    for qubit_id in target_qubits:
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

def create_circuit(n, m, cut, num_rounds=1, p1=0.02, p2=0.05, vertical=False,show_grid=False):
    """
    Create the complete Stim circuit with specified number of rounds, final measurement, and CAT functionality.

    Args:
        n (int): Width of the lattice.
        m (int): Height of the lattice.
        cut (int): The x-coordinate where the lattice is cut.
        num_rounds (int, optional): Number of stabilizer rounds to perform. Defaults to 1.
        p1 (float, optional): Depolarization probability for regular operations (DEPOLARIZE2). Defaults to 0.02.
        p2 (float, optional): Depolarization probability for CAT CX operations. Defaults to 0.05.
        vertical (bool, optional): Whether to invert the roles of X and Z ancillas. Defaults to False.

    Returns:
        stim.Circuit: The complete Stim circuit.
    """
    if not show_grid : 
        (data_qubits_sorted, x_measure_sorted, z_measure_sorted, 
         x_observable, z_observable, x_cat_copies, z_cat_copies, qubit_dict) = generate_unrotated_surface_code_grid(
        n, m, cut, cut_gap=5,  vertical=vertical) 
    else:
        (data_qubits_sorted, x_measure_sorted, z_measure_sorted, 
        x_observable, z_observable, x_cat_copies, z_cat_copies, qubit_dict) = plot_unrotated_surface_code_grid(
        n, m, cut, cut_gap=5, annotate=True, vertical=vertical)

    # Initialize an empty circuit
    circuit = stim.Circuit()

    # Step 1: Assign coordinates to qubits (only once in the first round)
    create_coords(circuit, qubit_dict, data_qubits_sorted, x_measure_sorted, z_measure_sorted, vertical)

    # Step 2: Reset all qubits (only once in the first round)
    all_qubits = qubit_dict['data_qubits'] + qubit_dict['x_original_ancillas'] + qubit_dict['z_original_ancillas'] + qubit_dict['x_cat_copies'] + qubit_dict['z_cat_copies']
    initial_reset(circuit, all_qubits)

    # Create a mapping from original ancillas to their CAT copies
    ancilla_copies = {}
    for copy_id in qubit_dict.get('x_cat_copies', []) + qubit_dict.get('z_cat_copies', []):
        # Original ancilla ID is the copy ID minus the fixed offset
        original_id = copy_id - 50000  # Fixed offset used during copy creation
        ancilla_copies[original_id] = copy_id

    # Create a mapping from Z ancilla to their neighboring data qubits with standard rules
    z_ancilla_to_data_qubits = {}
    for z_ancilla, coord in zip(qubit_dict['z_original_ancillas'], z_measure_sorted[:len(qubit_dict['z_original_ancillas'])]):
        x, y = coord
        neighbor_coords = [
            (x, y + 1),  # Up
            (x + 1, y),  # Right
            (x - 1, y),  # Left
            (x, y - 1)   # Down
        ]
        data_qubits = []

        for nc in neighbor_coords:
            # Ancillas use standard neighbor coordinates; no special boundary handling
            data_qubit_id = next((qid for qid, qcoord in zip(qubit_dict['data_qubits'], data_qubits_sorted) if qcoord == nc), None)
            if data_qubit_id is not None:
                data_qubits.append(data_qubit_id)

        # Assign the list of neighboring data qubits to the Z ancilla
        z_ancilla_to_data_qubits[z_ancilla] = data_qubits
    
    # **Step 2.1: Map CAT Copies to Their Neighboring Data Qubits**
    # This ensures that final detectors can include data qubits from CAT copies
    for original_id, copy_id in ancilla_copies.items():
        # Ensure that the copy is a Z ancilla copy
        if copy_id in qubit_dict['z_cat_copies']:
            # Find the index of the CAT copy in z_cat_copies
            try:
                copy_index = qubit_dict['z_cat_copies'].index(copy_id)
                # Get the coordinate of the CAT copy
                copy_coord = z_measure_sorted[len(qubit_dict['z_original_ancillas']) + copy_index]
                
                x, y = copy_coord
                neighbor_coords = [
                    (x, y + 1),  # Up
                    (x + 1, y),  # Right
                    (x - 1, y),  # Left
                    (x, y - 1)   # Down
                ]
                data_qubits_copy = []

                for nc in neighbor_coords:
                    # Ancillas use standard neighbor coordinates; no special boundary handling
                    data_qubit_id = next((qid for qid, qcoord in zip(qubit_dict['data_qubits'], data_qubits_sorted) if qcoord == nc), None)
                    if data_qubit_id is not None:
                        data_qubits_copy.append(data_qubit_id)

                # Assign the list of neighboring data qubits to the CAT copy
                z_ancilla_to_data_qubits[copy_id] = data_qubits_copy

            except ValueError:
                print(f"Warning: CAT copy {copy_id} for Z ancilla {original_id} not found in z_cat_copies.")
                continue

    # Number of measurements per round (excluding data qubits)
    total_mr = len(qubit_dict['x_original_ancillas']) + len(qubit_dict['z_original_ancillas']) + len(qubit_dict['x_cat_copies']) + len(qubit_dict['z_cat_copies'])

    # Perform stabilizer rounds
    for round_idx in range(num_rounds):
        measurement_list = []

        # Step 3: Apply Hadamard gates to original ancillas and Depolarize1
        if round_idx == 0:
            ancilla_list = qubit_dict['z_original_ancillas']
        else:
            ancilla_list = qubit_dict['x_original_ancillas'] + qubit_dict['z_original_ancillas']
        begin_round(circuit, all_qubits,p1,round_idx)
        # Step 4: Apply CAT operations
        apply_cat_operations(circuit, qubit_dict, p2,p1)

        # Step 4.1: Apply Hadamard gates to X ancillas and their CAT copies
        apply_hadamard_ancillas(circuit, qubit_dict['x_original_ancillas'] + qubit_dict['x_cat_copies'], p1, ancilla_type='X')

        # Step 5: Stabilizer round with Depolarize2
        stabilizer_round(
            circuit, qubit_dict, data_qubits_sorted, x_measure_sorted, z_measure_sorted,
            p1, p2, cut, spacing=5
        )

        # Step 6: Apply Hadamard gates again to X ancillas and their CAT copies and Depolarize1
        apply_hadamard_ancillas_again(circuit, qubit_dict['x_original_ancillas'] + qubit_dict['x_cat_copies'], p1, ancilla_type='X')

        # Step 7: Measure and reset ancillas and their CAT copies, capturing measurement indices
        ancilla_qubits = qubit_dict['x_original_ancillas'] + qubit_dict['z_original_ancillas'] + qubit_dict['x_cat_copies'] + qubit_dict['z_cat_copies']
        measure_and_reset(circuit, ancilla_qubits, p1, measurement_list)

        # Step 8: Add Detectors, considering ancilla and their copies
        # Only original ancillas are considered for detectors, copies are included within the same detectors
        add_detectors(circuit, ancilla_list, measurement_list, ancilla_copies, round_index=round_idx, total_mr=total_mr)

    # Perform Final Measurement
    final_measurement(circuit, qubit_dict['data_qubits'], p1, measurement_list)

    add_final_detectors(circuit, z_ancilla_to_data_qubits, measurement_list, data_qubits_sorted, ancilla_copies)

    # Append Logical Observable at the End
    append_logical_observable(circuit, data_qubits_sorted, qubit_dict, measurement_list, vertical=vertical)

    return circuit

