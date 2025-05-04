import numpy as np
import stim
import numpy as np
import svgwrite # type: ignore
from IPython.display import SVG, display
#from Surface import surface_data

# Function to map qubit IDs to numbers with specific prefixes
def map_qubit_ids(matrix):
    qubit_map = {}
    counter = 0
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            if isinstance(matrix[i, j], str) and matrix[i, j] != '#':
                qubit_id = matrix[i, j]
                if qubit_id.startswith('A'):
                    mapped_id = '10' + qubit_id[1:]
                elif qubit_id.startswith('BB'):
                    mapped_id = '20' + qubit_id[2:] + '01'
                elif qubit_id.startswith('CB'):
                    mapped_id = '20' + qubit_id[2:] + '02'

                elif qubit_id.startswith('B'):
                    mapped_id = '20' + qubit_id[1:]
                elif qubit_id.startswith('T'):
                    mapped_id = '30' + qubit_id[1:]
                elif qubit_id.startswith('Q'):
                    mapped_id = '40' + qubit_id[1:]
                else:
                    mapped_id = qubit_id
                
                if mapped_id not in qubit_map:
                        qubit_map[qubit_id] = int(mapped_id)
    
    return qubit_map

# Function to initialize qubits with coordinates
def initialize_qubits(circuit, cut_matrix, qubit_map):
    rows, cols = cut_matrix.shape
    for i in range(rows):
        for j in range(cols):
            if isinstance(cut_matrix[i, j], str) and cut_matrix[i, j] != '#':
                qubit_id = qubit_map[cut_matrix[i, j]]
                circuit += stim.Circuit(f"QUBIT_COORDS({i},{j}) {qubit_id}")
    return list(qubit_map.values())

# Function to reset qubits
def reset_qubits(circuit, qubit_ids, noise,noise2,r):
    high=[]
    low=[]
    for i in qubit_ids:
        if str(i).startswith('3'):
            high.append(i)
        else:
            low.append(i)

    if r == 0:
        circuit.append('R', qubit_ids)
        circuit.append('X_ERROR', low, noise)
        circuit.append('X_ERROR', high, noise)
        circuit.append('TICK')


    else:    
        circuit.append('DEPOLARIZE1', low, noise)
        circuit.append('DEPOLARIZE1', high, noise)



# Function to create Bell pairs
def create_bell_pairs(circuit, t1_ancillas, t2_ancillas, t3_ancillas, qubit_map,noise,noise2):
    t1_ancilla_ids = [qubit_map[q] for q in t1_ancillas]
    t2_ancilla_ids = [qubit_map[q] for q in t2_ancillas]
    t3_ancilla_ids = [qubit_map[q] for q in t3_ancillas]
    
    # Apply H gates to T1 ancillas
    circuit.append("H", t1_ancilla_ids)
    circuit.append('DEPOLARIZE1',t1_ancilla_ids, noise)
    circuit.append('TICK')
    
    # Apply CNOT gates from T1 to T2 ancillas
    all_cx_pairs_t1_t2 = []
    for t1, t2 in zip(t1_ancilla_ids, t2_ancilla_ids):
        all_cx_pairs_t1_t2.append([t1, t2])
        
    circuit.append("CX", [q for pair in all_cx_pairs_t1_t2 for q in pair])
    circuit.append('DEPOLARIZE2', [q for pair in all_cx_pairs_t1_t2 for q in pair], noise2)
    circuit.append('TICK')
    
    # Apply CNOT gates from T2 to T3 ancillas
    all_cx_pairs_t2_t3 = []
    for t2, t3 in zip(t2_ancilla_ids, t3_ancilla_ids):
        all_cx_pairs_t2_t3.append([t2, t3])
        
    circuit.append("CX", [q for pair in all_cx_pairs_t2_t3 for q in pair])
    circuit.append('DEPOLARIZE2', [q for pair in all_cx_pairs_t2_t3 for q in pair], noise2)
    circuit.append('TICK')

# Function to create Bell pairs, CAT STATE
def create_cat_state(circuit, CB_ancillas,BB_ancillas,t1_ancillas, t2_ancillas, t3_ancillas, qubit_map, noise,noise2):
    t1_ancilla_ids = [qubit_map[q] for q in t1_ancillas]
    t2_ancilla_ids = [int(str(q)[:-1] + '2') for q in t1_ancilla_ids]
    t3_ancilla_ids = [qubit_map[q] for q in t3_ancillas]
    CB_ancilla_ids = [qubit_map[q] for q in CB_ancillas]
    BB_ancilla_ids = [qubit_map[q] for q in BB_ancillas]
    mid_ancilla_ids = [int(str(q)[:-1] + '0') for q in CB_ancilla_ids]

    # Apply H gates to T3 ancillas
    circuit.append("H", t3_ancilla_ids)
    circuit.append("H", BB_ancilla_ids)
    circuit.append('DEPOLARIZE1',t3_ancilla_ids, noise)
    circuit.append('DEPOLARIZE1',BB_ancilla_ids, noise)
    circuit.append('TICK')
    
    # Apply CNOT gates from T3 to T2 ancillas   (3 to 2)
    all_cx_pairs_t3_t2 = []
    
    for t3, t2 in zip(t3_ancilla_ids, t2_ancilla_ids):
        all_cx_pairs_t3_t2.append([t3, t2])
    all_cx_pairs_BB_mid = []
    for BB, mid in zip(BB_ancilla_ids, mid_ancilla_ids):
        all_cx_pairs_BB_mid.append([BB, mid])

    # Apply CNOT gates from T2 to T1 ancillas   (3 to 1)
    all_cx_pairs_t2_t1 = []
    for t2, t1 in zip(t2_ancilla_ids, t1_ancilla_ids):
        all_cx_pairs_t2_t1.append([t2, t1])
    all_cx_pairs_mid_CB = []
    for  mid,CB in zip(mid_ancilla_ids, CB_ancilla_ids):
        all_cx_pairs_mid_CB.append([mid, CB])

    # Get all qubits involved in CX gates
    cx_involved_qubits = set([q for pair in all_cx_pairs_t2_t1+ all_cx_pairs_mid_CB + all_cx_pairs_t3_t2 + all_cx_pairs_BB_mid for q in pair])
    
    # Get all qubits in the circuit
    all_qubits = set(qubit_map.values())
    
    # Find qubits not involved in CX gates
    uninvolved_qubits = list(all_qubits - cx_involved_qubits)


    circuit.append("CX", [q for pair in all_cx_pairs_t3_t2 for q in pair])
    circuit.append("CX", [q for pair in all_cx_pairs_BB_mid for q in pair])
    circuit.append('DEPOLARIZE2', [q for pair in all_cx_pairs_t3_t2 for q in pair], noise2)
    circuit.append('DEPOLARIZE2', [q for pair in all_cx_pairs_BB_mid for q in pair], noise2)
    if uninvolved_qubits:
        circuit.append('DEPOLARIZE1', uninvolved_qubits, noise)
    circuit.append('TICK') 


    circuit.append("CX", [q for pair in all_cx_pairs_t2_t1 for q in pair])
    circuit.append("CX", [q for pair in all_cx_pairs_mid_CB for q in pair])
    circuit.append('DEPOLARIZE2', [q for pair in all_cx_pairs_mid_CB for q in pair], noise)
    circuit.append('DEPOLARIZE2', [q for pair in all_cx_pairs_t2_t1 for q in pair], noise)
    
    circuit.append('TICK') 



def create_cat_state_zigzag(circuit, CB_ancillas,BB_ancillas, qubit_map, noise,noise2):

    CB_ancilla_ids = [qubit_map[q] for q in CB_ancillas]
    BB_ancilla_ids = [qubit_map[q] for q in BB_ancillas]
    mid_ancilla_ids = [int(str(q)[:-1] + '0') for q in CB_ancilla_ids]
    # Apply H gates to T3 ancillas
    circuit.append("H", BB_ancilla_ids)
    circuit.append('DEPOLARIZE1',BB_ancilla_ids, noise)

    circuit.append('TICK')
    
    # Apply CNOT gates from  BB MID
    all_cx_pairs_BB_mid = []
    for BB, mid in zip(BB_ancilla_ids, mid_ancilla_ids):
        all_cx_pairs_BB_mid.append([BB, mid])
        
    circuit.append("CX", [q for pair in all_cx_pairs_BB_mid for q in pair])
    circuit.append('DEPOLARIZE2', [q for pair in all_cx_pairs_BB_mid for q in pair], noise2)
    circuit.append('TICK')

    # Apply CNOT gates from  MID to CB
    all_cx_pairs_mid_CB = []
    for  mid,CB in zip(mid_ancilla_ids, CB_ancilla_ids):
        all_cx_pairs_mid_CB.append([mid, CB])
        
    circuit.append("CX", [q for pair in all_cx_pairs_mid_CB for q in pair])
    circuit.append('DEPOLARIZE2', [q for pair in all_cx_pairs_mid_CB for q in pair], noise2)
    circuit.append('TICK')    


# Function to apply Hadamard gates to all X ancillas
def apply_hadamard_x_ancillas(circuit, x_ancillas, t1_ancillas, qubit_map,noise,noise2):
    t1_ids = set(qubit_map[q] for q in t1_ancillas)
    x_ancilla_ids = [qubit_map[q] for q in x_ancillas]
    circuit.append("H", x_ancilla_ids)
    high=[]
    low=[]
    for i in x_ancillas:
        if i.startswith('T'):
            high.append(qubit_map[i])
        else:
            low.append(qubit_map[i]) 
    circuit.append('DEPOLARIZE1',high, noise)
    circuit.append('DEPOLARIZE1',low, noise)

    circuit.append('TICK')
## Function to create stabilizers for a given direction
def create_stabilizers_direction(circuit, data, qubit_map, noise,noise2, direction,inverted):
    # Get diagonal neighbors and extract the given direction neighbors
    diagonal_neighbors = data['diagonal_neighbors']
    if inverted: 
        index_z = {'nw': 2, 'ne': 3, 'se': 4, 'sw': 5}[direction]
        index_x = {'nw': 4, 'ne': 3, 'se': 2, 'sw': 5}[direction]
    else:
        index_x = {'nw': 2, 'ne': 3, 'se': 4, 'sw': 5}[direction]
        index_z = {'nw': 4, 'ne': 3, 'se': 2, 'sw': 5}[direction]
  
    # Initialize a string to collect all qubits involved in CX gates
    depolarize_string1 = f'DEPOLARIZE2({noise}) '
    depolarize_string2 = f'DEPOLARIZE2({noise}) '

    # For Z ancillas, apply CX between the neighbor and the ancilla
    
    for ancilla_type, ancilla, *neighbors in diagonal_neighbors:
        #index = {'nw': 2, 'ne': 3, 'se': 4, 'sw': 5}[direction]
        neighbor = neighbors[index_z - 2]
        if ancilla_type == 'Z' and neighbor is not None and neighbor != '#':
            neighbor_id = qubit_map[neighbor]
            ancilla_id = qubit_map[ancilla]
            circuit.append("CX", [neighbor_id, ancilla_id])
            if str(ancilla_id).startswith('3'):
                depolarize_string2 += f'{neighbor_id} {ancilla_id} '
            else:
                depolarize_string1 += f'{neighbor_id} {ancilla_id} '

    
    # For X ancillas, apply CX between the ancilla and its neighbor
    for ancilla_type, ancilla, *neighbors in diagonal_neighbors:
        #index = {'nw': 4, 'ne': 3, 'se': 2, 'sw': 5}[direction]

        neighbor = neighbors[index_x - 2]
        if ancilla_type == 'X' and neighbor is not None and neighbor != '#':
            neighbor_id = qubit_map[neighbor]
            ancilla_id = qubit_map[ancilla]
            circuit.append("CX", [ancilla_id, neighbor_id])
            if str(ancilla_id).startswith('3'):
                depolarize_string2 += f'{ancilla_id} {neighbor_id} '
            else:
                depolarize_string1 += f'{ancilla_id} {neighbor_id} '
    # Apply DEPOLARIZE2 to all involved qubits
    circuit += stim.Circuit(f'{depolarize_string1.strip()}')
    circuit += stim.Circuit(f'{depolarize_string2.strip()}')

    circuit.append('TICK')



# Function to create all stabilizers
#def create_stabilizers(circuit, data, qubit_map, noise,noise2,inverted):
    #create_stabilizers_direction(circuit, data, qubit_map, noise,noise2, 'nw',inverted)
    #create_stabilizers_direction(circuit, data, qubit_map, noise,noise2, 'ne',inverted)
    #create_stabilizers_direction(circuit, data, qubit_map, noise,noise2, 'sw',inverted)
    #create_stabilizers_direction(circuit, data, qubit_map, noise,noise2, 'se',inverted)


def apply_hadamard_t2(circuit, t2_ancillas, qubit_map,noise):
    t2_ancilla_ids = [qubit_map[q] for q in t2_ancillas]
    circuit.append("H", t2_ancilla_ids)
    circuit.append('DEPOLARIZE1',t2_ancilla_ids, noise)
    circuit.append('TICK')


# Function to apply Measure and Reset (MR) to all elements except Qs
def apply_measure_and_reset(circuit, data, qubit_map,noise,noise2):
    elements = data['T1_Ancillas'] + data['T2_Ancillas'] + data['T3_Ancillas'] + data['A_Ancillas'] + data['B_Ancillas'] + data['BB_Ancillas'] + data['CB_Ancillas']
    element_ids = [qubit_map[q] for q in elements]
    high_elements = data['T1_Ancillas'] + data['T2_Ancillas'] + data['T3_Ancillas']
    high_element_ids = [qubit_map[q] for q in high_elements]
    low_elements = data['A_Ancillas'] + data['B_Ancillas']
    low_element_ids = [qubit_map[q] for q in low_elements]

    circuit.append('X_ERROR',high_element_ids, noise)
    circuit.append('X_ERROR',low_element_ids, noise)
    circuit.append("MR", element_ids)
    circuit.append('X_ERROR',high_element_ids, noise)
    circuit.append('X_ERROR',low_element_ids, noise)

    circuit.append('TICK')
    #print(data['CB_Ancillas'])
    #print(elements) 
    return element_ids  # Return the order of measurements

# Function to add Z detectors for all Z ancillas (excluding T1)
def add_z_detectors(circuit, data, qubit_map, measurement_order, round_index,CAT):
    z_ancillas = [qubit_map[q] for q in data['Z_Ancillas'] if q not in data['T1_Ancillas'] and q not in data['CB_Ancillas']]
    t1_ancillas = {qubit_map[q]: q for q in data['T1_Ancillas']}
    t3_ancillas = {qubit_map[q]: q for q in data['T3_Ancillas']}
    cb_ancillas = {qubit_map[q]: q for q in data['CB_Ancillas']}
    bb_ancillas = {qubit_map[q]: q for q in data['BB_Ancillas']}

    total_mr = len(measurement_order)
    
    for z_ancilla in z_ancillas:
        # Initialize the recs string
        recs = " "
        
        # Find the index of the measurement in the measurement order
        index = measurement_order.index(z_ancilla)
        # Calculate the backwards reference
        backwards_reference = -(len(measurement_order) - index)
        
        if round_index == 0:
            recs += f" rec[{backwards_reference}]"
        else:
            recs += f" rec[{backwards_reference}] rec[{backwards_reference - total_mr}]"

        if z_ancilla in bb_ancillas:
            # Replace the last digit of the BB ancilla with '1' to find the corresponding CB ancilla
            corresponding_cb = int(str(z_ancilla)[:-1] + '2')
            #corresponding_cb = qubit_map[corresponding_cb_name]
            # Find the index of the corresponding T1 in the measurement order
            cb_index = measurement_order.index(corresponding_cb)
            # Calculate the backwards reference for the corresponding T1
            cb_backwards_reference = -(len(measurement_order) - cb_index)
            
            if round_index == 0:
                recs += f" rec[{cb_backwards_reference}] "
            else:
                recs += f" rec[{cb_backwards_reference}] rec[{cb_backwards_reference - total_mr}] "

        if z_ancilla in t3_ancillas:
            # Replace the last digit of the T3 ancilla with '1' to find the corresponding T1 ancilla
            corresponding_t1_name = t3_ancillas[z_ancilla][:-1] + '1'
            corresponding_t1 = qubit_map[corresponding_t1_name]
            # Find the index of the corresponding T1 in the measurement order
            t1_index = measurement_order.index(corresponding_t1)
            # Calculate the backwards reference for the corresponding T1
            t1_backwards_reference = -(len(measurement_order) - t1_index)
            
            if round_index == 0:
                recs += f" rec[{t1_backwards_reference}] "
            else:
                recs += f" rec[{t1_backwards_reference}] rec[{t1_backwards_reference - total_mr}] "
            
            if not CAT:
                # Additional correction logic
                row_index = int(t3_ancillas[z_ancilla][1:-1]) - 1
                up_correction_row = row_index - 2
                down_correction_row = row_index + 2

                up_correction_name = f'T{up_correction_row + 1}2'
                down_correction_name = f'T{down_correction_row + 1}2'
                if up_correction_name in qubit_map:
                    up_correction_ancilla = qubit_map[up_correction_name]
                    up_correction_index = measurement_order.index(up_correction_ancilla)
                    up_correction_backwards_reference = -(len(measurement_order) - up_correction_index)
                    if round_index == 0:
                        recs += f" rec[{up_correction_backwards_reference}] "
                    else:
                        recs += f" rec[{up_correction_backwards_reference}] "
                if down_correction_name in qubit_map:
                    down_correction_ancilla = qubit_map[down_correction_name]
                    down_correction_index = measurement_order.index(down_correction_ancilla)
                    down_correction_backwards_reference = -(len(measurement_order) - down_correction_index)
                    if round_index > 0:
                        recs += f"  rec[{down_correction_backwards_reference - total_mr}]"

        # Append DETECTOR instruction
        circuit += stim.Circuit(f"DETECTOR{recs}")
    if round_index ==0:
        circuit.append("TICK")

# Function to add X detectors for all X ancillas (excluding T1)
def add_x_detectors(circuit, data, qubit_map, measurement_order, round_index,CAT):
    x_ancillas = [qubit_map[q] for q in data['X_Ancillas'] if q not in data['T1_Ancillas'] and q not in data['CB_Ancillas']]
    t1_ancillas = {qubit_map[q]: q for q in data['T1_Ancillas']}
    t3_ancillas = {qubit_map[q]: q for q in data['T3_Ancillas']}
    cb_ancillas = {qubit_map[q]: q for q in data['CB_Ancillas']}
    bb_ancillas = {qubit_map[q]: q for q in data['BB_Ancillas']}

    total_mr = len(measurement_order)
    
    for x_ancilla in x_ancillas:
        # Initialize the recs string
        recs = " "
        
        # Find the index of the measurement in the measurement order
        index = measurement_order.index(x_ancilla)
        # Calculate the backwards reference
        backwards_reference = -(len(measurement_order) - index)
        
        if round_index == 0:
            recs += f" rec[{backwards_reference}]"
        else:
            recs += f" rec[{backwards_reference}] rec[{backwards_reference - total_mr}]"


        if x_ancilla in bb_ancillas:
            # Replace the last digit of the BB ancilla with '1' to find the corresponding CB ancilla
            corresponding_cb = int(str(x_ancilla)[:-1] + '2')
            #corresponding_cb = qubit_map[corresponding_cb_name]
            # Find the index of the corresponding T1 in the measurement order
            cb_index = measurement_order.index(corresponding_cb)
            # Calculate the backwards reference for the corresponding T1
            cb_backwards_reference = -(len(measurement_order) - cb_index)
            
            if round_index == 0:
                recs += f" rec[{cb_backwards_reference}] "
            else:
                recs += f" rec[{cb_backwards_reference}] rec[{cb_backwards_reference - total_mr}] "


        if x_ancilla in t3_ancillas:
            # Replace the last digit of the T3 ancilla with '1' to find the corresponding T1 ancilla
            corresponding_t1_name = t3_ancillas[x_ancilla][:-1] + '1'
            corresponding_t1 = qubit_map[corresponding_t1_name]
            # Find the index of the corresponding T1 in the measurement order
            t1_index = measurement_order.index(corresponding_t1)
            # Calculate the backwards reference for the corresponding T1
            t1_backwards_reference = -(len(measurement_order) - t1_index)
            
            if round_index == 0:
                recs += f" rec[{t1_backwards_reference}] "
            else:
                recs += f" rec[{t1_backwards_reference}] rec[{t1_backwards_reference - total_mr}] "
    
            if not CAT:
                # Additional correction logic
                row_index = int(t3_ancillas[x_ancilla][1:-1]) - 1
                up_correction_row = row_index - 2
                down_correction_row = row_index + 2

                up_correction_name = f'T{up_correction_row + 1}2'
                down_correction_name = f'T{down_correction_row + 1}2'
                if up_correction_name in qubit_map:
                    up_correction_ancilla = qubit_map[up_correction_name]
                    up_correction_index = measurement_order.index(up_correction_ancilla)
                    up_correction_backwards_reference = -(len(measurement_order) - up_correction_index)
                    if round_index == 0:
                        recs += f" rec[{up_correction_backwards_reference}] "
                    else:
                        recs += f" rec[{up_correction_backwards_reference}] "
                if down_correction_name in qubit_map:
                    down_correction_ancilla = qubit_map[down_correction_name]
                    down_correction_index = measurement_order.index(down_correction_ancilla)
                    down_correction_backwards_reference = -(len(measurement_order) - down_correction_index)
                    if round_index > 0:
                        recs += f"  rec[{down_correction_backwards_reference - total_mr}]"

        # Append DETECTOR instruction
        circuit += stim.Circuit(f"DETECTOR{recs}")
    circuit.append("TICK")

# Function to apply "M" gate to all Q qubits and keep track of the measurement order
def measure_q_qubits(circuit, qubit_map, data, measurement_order,noise):
    q_qubits = [qubit_map[q] for q in data['Qubits']]
    circuit.append('X_ERROR',q_qubits, noise)

    circuit.append("M", q_qubits)
    circuit.append('TICK')
    measurement_order.extend(q_qubits)  # Track measurement order

# Function to add final detectors for all Z ancillas
def add_final_detectors(circuit, data, qubit_map, measurement_order,CAT):
    z_ancillas = [qubit_map[q] for q in data['Z_Ancillas'] if q not in data['CB_Ancillas']]
    t1_ancillas = {qubit_map[q]: q for q in data['T1_Ancillas']}
    t3_ancillas = {qubit_map[q]: q for q in data['T3_Ancillas']}
    bb_ancillas = {qubit_map[q]: q for q in data['BB_Ancillas']}

    diagonal_neighbors = data['diagonal_neighbors']

    for ancilla_type, ancilla, nw, ne, se, sw in diagonal_neighbors:
        if qubit_map[ancilla] in z_ancillas:
            recs = ""
            neighbors = [nw, ne, se, sw]
            # Include the ancilla itself
            ancilla_id = qubit_map[ancilla]
            index = measurement_order.index(ancilla_id)
            backwards_reference = -(len(measurement_order) - index)
            if qubit_map[ancilla] not in t1_ancillas:
                recs += f" rec[{backwards_reference}]"
            
                # Include only valid neighbors
                valid_neighbors = [n for n in neighbors if n is not None and n != '#']
                for neighbor in valid_neighbors:
                    neighbor_id = qubit_map[neighbor]
                    index = measurement_order.index(neighbor_id)
                    backwards_reference = -(len(measurement_order) - index)
                    recs += f" rec[{backwards_reference}]"
                
                if qubit_map[ancilla] in bb_ancillas:
                    # Replace the last digit of the BB ancilla with '1' to find the corresponding CB ancilla
                    corresponding_cb = int(str(qubit_map[ancilla])[:-1] + '2')
                    corresponding_cb_name = 'CB'+ str(corresponding_cb)[2:-2] 
                    #corresponding_cb = qubit_map[corresponding_cb_name]
                    # Find the index of the corresponding T1 in the measurement order
                    cb_index = measurement_order.index(corresponding_cb)
                    # Calculate the backwards reference for the corresponding T1
                    cb_backwards_reference = -(len(measurement_order) - cb_index)
                    recs += f" rec[{cb_backwards_reference}] "

                    corresponding_cb_neighbors = next((nw, ne, se, sw) for ancilla_type, anc, nw, ne, se, sw in diagonal_neighbors if anc == corresponding_cb_name)
                    valid_cb_neighbors = [n for n in corresponding_cb_neighbors if n is not None and n != '#']
                    for neighbor in valid_cb_neighbors:
                            neighbor_id = qubit_map[neighbor]
                            index = measurement_order.index(neighbor_id)
                            backwards_reference = -(len(measurement_order) - index)
                            recs += f" rec[{backwards_reference}]"


       

                if qubit_map[ancilla] in t3_ancillas:
                    # Handle T3 ancillas separately
                    corresponding_t1_name = t3_ancillas[qubit_map[ancilla]][:-1] + '1'
                    corresponding_t1 = qubit_map[corresponding_t1_name]
                    
                    # Include the corresponding T1 ancilla itself
                    corresponding_t1_id = qubit_map[corresponding_t1_name]
                    index = measurement_order.index(corresponding_t1_id)
                    backwards_reference = -(len(measurement_order) - index)
                    recs += f" rec[{backwards_reference}]"
                    
                    # Find the neighbors of the corresponding T1 ancilla
                    corresponding_t1_neighbors = next((nw, ne, se, sw) for ancilla_type, anc, nw, ne, se, sw in diagonal_neighbors if anc == corresponding_t1_name)
                    valid_t1_neighbors = [n for n in corresponding_t1_neighbors if n is not None and n != '#']
                    
                    for neighbor in valid_t1_neighbors:
                        neighbor_id = qubit_map[neighbor]
                        index = measurement_order.index(neighbor_id)
                        backwards_reference = -(len(measurement_order) - index)
                        recs += f" rec[{backwards_reference}]"

                    if not CAT:
                        # Additional correction logic
                        row_index = int(t3_ancillas[qubit_map[ancilla]][1:-1]) - 1
                        down_correction_row = row_index + 2

                        down_correction_name = f'T{down_correction_row + 1}2'
                        if down_correction_name in qubit_map:
                            down_correction_ancilla = qubit_map[down_correction_name]
                            down_correction_index = measurement_order.index(down_correction_ancilla)
                            down_correction_backwards_reference = -(len(measurement_order) - down_correction_index)
                            recs += f" rec[{down_correction_backwards_reference}]" 

                # Append DETECTOR instruction
                circuit += stim.Circuit(f"DETECTOR{recs}")
    circuit.append("TICK")


# Function to include observable
def include_observable(circuit, data, qubit_map, measurement_order, rounds, CAT=False, inverted=False):
    if inverted:
        # Use row instead of column
        penultimate_row = data['cut_matrix'][-2]
        
        # Extract Q qubits in the penultimate row
        q_qubits = [qubit_map[q] for q in penultimate_row if isinstance(q, str) and q.startswith('Q')]
        
        # Create the list of references
        references = []
        for q in q_qubits:
            index = measurement_order.index(q)
            backwards_reference = -(len(measurement_order) - index)
            references.append(backwards_reference)
        
        if not CAT:
            # Add extra correction logic if CAT is False
            # Calculate the row index for the row above the penultimate row (-3)
            above_row_index = len(data['cut_matrix']) - 3
            above_row = data['cut_matrix'][above_row_index]
            
            # Create the correction string
            correction_string = 'T' + f'{above_row_index + 1}' + '2'
            elements = data['T1_Ancillas'] + data['T2_Ancillas'] + data['T3_Ancillas'] + data['A_Ancillas'] + data['B_Ancillas']
            
            # Find the backward measurement for the ancilla identified by the correction string
            if correction_string in qubit_map:
                correction_ancilla = qubit_map[correction_string]
                correction_index = measurement_order.index(correction_ancilla)
                correction_backwards_reference = -(len(measurement_order) - correction_index)
                references.append(correction_backwards_reference)
                
                for i in range(rounds - 1):
                    additional_reference = correction_backwards_reference - ((1 + i) * len(elements))
                    references.append(additional_reference)
        
        # Create the OBSERVABLE_INCLUDE line
        recs = " ".join([f"rec[{ref}]" for ref in references])
        circuit += stim.Circuit(f"OBSERVABLE_INCLUDE(0) {recs}")
        circuit.append("TICK")
    else:
        # Use column
        penultimate_column = data['cut_matrix'][:, -2]
        
        # Extract Q qubits in the penultimate column
        q_qubits = [qubit_map[q] for q in penultimate_column if isinstance(q, str) and q.startswith('Q')]
        
        # Create the list of references
        references = []
        for q in q_qubits:
            index = measurement_order.index(q)
            backwards_reference = -(len(measurement_order) - index)
            references.append(backwards_reference)
        
        # Create the OBSERVABLE_INCLUDE line
        recs = " ".join([f"rec[{ref}]" for ref in references])
        circuit += stim.Circuit(f"OBSERVABLE_INCLUDE(0) {recs}")
        circuit.append("TICK")


def Classical_correction1(circuit, CB_ancillas,t1_ancillas, qubit_map,noise):
    t1_ancilla_ids = [qubit_map[q] for q in t1_ancillas]
    t2_ancilla_ids = [int(str(q)[:-1] + '2') for q in t1_ancilla_ids]
    CB_ancilla_ids = [qubit_map[q] for q in CB_ancillas]
    mid_ancilla_ids = [int(str(q)[:-1] + '0') for q in CB_ancilla_ids]

    element_ids = t2_ancilla_ids+  mid_ancilla_ids


    circuit.append("H", t2_ancilla_ids+mid_ancilla_ids)
    circuit.append('DEPOLARIZE1', t2_ancilla_ids+mid_ancilla_ids, noise)
    circuit.append('TICK')
    circuit.append("MR", t2_ancilla_ids+mid_ancilla_ids)
    circuit.append('X_ERROR', t2_ancilla_ids+mid_ancilla_ids, noise)
    circuit.append('TICK')


    return element_ids  # Return the order of measurements

def Classical_correction2_inverted(circuit,data, CB_ancillas,t1_ancillas, qubit_map,noise,measurement_order):
    diagonal_neighbors = data['diagonal_neighbors']
    #print(diagonal_neighbors)
    z_ancillas_CB = [qubit_map[q] for q in data['Z_Ancillas'] if q  in data['CB_Ancillas']]
    z_ancillas_t2 = [qubit_map[q] for q in data['Z_Ancillas'] if q in (data['T1_Ancillas'])]
    t2_Zancilla_ids = [int(str(q)[:-1] + '2') for q in z_ancillas_t2]
    mid_Zancilla_ids = [int(str(q)[:-1] + '0') for q in z_ancillas_CB]
    z_ancilla = t2_Zancilla_ids + mid_Zancilla_ids


    x_ancillas_CB = [qubit_map[q] for q in data['X_Ancillas'] if q  in data['CB_Ancillas']]
    x_ancillas_t2 = [qubit_map[q] for q in data['X_Ancillas'] if q in (data['T1_Ancillas'])]
    t2_Xancilla_ids = [int(str(q)[:-1] + '2') for q in x_ancillas_t2]
    mid_Xancilla_ids = [int(str(q)[:-1] + '0') for q in x_ancillas_CB]
    x_ancilla = t2_Xancilla_ids + mid_Xancilla_ids

    id_to_qubit = {v: k for k, v in qubit_map.items()}

    def get_qubit_neighbors(data, qubit_name):
        neighbors = set()
        for entry in data:
            a_type, q_name, n1, n2, n3, n4 = entry
            if q_name == qubit_name:
                for neighbor in [n1, n2, n3, n4]:
                    if neighbor is not None and neighbor != '#':
                        neighbors.add(neighbor)
        #print(neighbors)    
        return list(neighbors)


    def get_aso_qubit(item):
        if item in ( t2_Xancilla_ids + t2_Zancilla_ids):
            item = int(str(item)[:-1] + '3')
            qubit_name = id_to_qubit.get(item)
            #print(qubit_name)
            aso_qubit_name = get_qubit_neighbors(diagonal_neighbors,qubit_name)
            aso_qubit_ids = [qubit_map[q] for q in aso_qubit_name]
            return aso_qubit_ids
        elif item in ( mid_Xancilla_ids + mid_Zancilla_ids):
            item = int(str(item)[:-1] + '2')
            qubit_name = id_to_qubit.get(item)
            #print(qubit_name)
            aso_qubit_name = get_qubit_neighbors(diagonal_neighbors,qubit_name)
            aso_qubit_ids = [qubit_map[q] for q in aso_qubit_name]
            return aso_qubit_ids  


    for item in x_ancilla:
        index = measurement_order.index(item)
        backwards_reference = -(len(measurement_order) - index)
        aso_qubit = get_aso_qubit(item)
        if aso_qubit is not None:
            aso_qubit = aso_qubit[0]
            instruction= f"CX rec[{backwards_reference}] {aso_qubit}"
            #print(instruction)
            circuit += stim.Circuit(instruction)



        if item in mid_Xancilla_ids and aso_qubit is not None:
            #print(id_to_qubit.get(aso_qubit))
            matching_entries1 = [entry for entry in diagonal_neighbors if entry[2] == id_to_qubit.get(aso_qubit)]   
            matching_entries2 = [entry for entry in diagonal_neighbors if entry[4] == id_to_qubit.get(aso_qubit)]   
            #print(f"ENTRY IS{matching_entries1}")
            #print(f"ENTRY2 IS{matching_entries2}")
            if len(matching_entries1) != 0 and int(str(qubit_map[matching_entries1[0][1]])[:-1] +"0") != item: 
                extra_correction1 = qubit_map[matching_entries1[0][1]]
                instruction= f"CX rec[{backwards_reference}] {extra_correction1}"
                circuit += stim.Circuit(instruction)


            if len(matching_entries2) != 0 and qubit_map[matching_entries1[0][1]] != item: 
       

                extra_correction2 = qubit_map[matching_entries2[0][1]]

                instruction= f"CX rec[{backwards_reference}] {extra_correction2}"
                circuit += stim.Circuit(instruction)


        if item in t2_Xancilla_ids and aso_qubit is not None:
            #print(id_to_qubit.get(aso_qubit))
            matching_entries1 = [entry for entry in diagonal_neighbors if entry[3] == id_to_qubit.get(aso_qubit)]   
            #print(f"ENTRY IS{matching_entries1}")
            if len(matching_entries1) != 0 and int(str(qubit_map[matching_entries1[0][1]])[:-1] +"2") != item: 
                    extra_correction1 = qubit_map[matching_entries1[0][1]]
                    instruction= f"CX rec[{backwards_reference}] {extra_correction1}"
                    circuit += stim.Circuit(instruction)




    for item in z_ancilla:
        index = measurement_order.index(item)
        backwards_reference = -(len(measurement_order) - index)
        aso_qubit = get_aso_qubit(item)
        #print(aso_qubit)
        if aso_qubit is not None:
            aso_qubit = aso_qubit[0]
            instruction= f"CZ rec[{backwards_reference}] {aso_qubit}"
            #print(instruction)
            circuit += stim.Circuit(instruction)

        if item in mid_Zancilla_ids and aso_qubit is not None:
            #print(id_to_qubit.get(aso_qubit))
            matching_entries = [entry for entry in diagonal_neighbors if entry[3] == id_to_qubit.get(aso_qubit)]   
            if len(matching_entries) != 0 and qubit_map[matching_entries1[0][1]] != item: 
                extra_correction = qubit_map[matching_entries[0][1]]
                instruction= f"CX rec[{backwards_reference}] {extra_correction}"
                circuit += stim.Circuit(instruction)

        if item in mid_Zancilla_ids and aso_qubit is not None:
            #print(id_to_qubit.get(aso_qubit))
            matching_entries1 = [entry for entry in diagonal_neighbors if entry[2] == id_to_qubit.get(aso_qubit)]   
            matching_entries2 = [entry for entry in diagonal_neighbors if entry[5] == id_to_qubit.get(aso_qubit)]   
            #print(f"ENTRY IS{matching_entries1}")
            #print(f"ENTRY2 IS{matching_entries2}")
            if len(matching_entries1) != 0 and int(str(qubit_map[matching_entries1[0][1]])[:-1] +"0") != item: 
                extra_correction1 = qubit_map[matching_entries1[0][1]]
                instruction= f"CX rec[{backwards_reference}] {extra_correction1}"
                circuit += stim.Circuit(instruction)                

def Classical_correction2(circuit,data, CB_ancillas,t1_ancillas, qubit_map,noise,measurement_order):
    diagonal_neighbors = data['diagonal_neighbors']
    #print(diagonal_neighbors)
    z_ancillas_CB = [qubit_map[q] for q in data['Z_Ancillas'] if q  in data['CB_Ancillas']]
    z_ancillas_t2 = [qubit_map[q] for q in data['Z_Ancillas'] if q in (data['T1_Ancillas'])]
    t2_Zancilla_ids = [int(str(q)[:-1] + '2') for q in z_ancillas_t2]
    mid_Zancilla_ids = [int(str(q)[:-1] + '0') for q in z_ancillas_CB]
    z_ancilla = t2_Zancilla_ids + mid_Zancilla_ids


    x_ancillas_CB = [qubit_map[q] for q in data['X_Ancillas'] if q  in data['CB_Ancillas']]
    x_ancillas_t2 = [qubit_map[q] for q in data['X_Ancillas'] if q in (data['T1_Ancillas'])]
    t2_Xancilla_ids = [int(str(q)[:-1] + '2') for q in x_ancillas_t2]
    mid_Xancilla_ids = [int(str(q)[:-1] + '0') for q in x_ancillas_CB]
    x_ancilla = t2_Xancilla_ids + mid_Xancilla_ids

    id_to_qubit = {v: k for k, v in qubit_map.items()}

    def get_qubit_neighbors(data, qubit_name):
        neighbors = set()
        for entry in data:
            a_type, q_name, n1, n2, n3, n4 = entry
            if q_name == qubit_name:
                for neighbor in [n1, n2, n3, n4]:
                    if neighbor is not None and neighbor != '#':
                        neighbors.add(neighbor)
        #print(neighbors)    
        return list(neighbors)


    def get_aso_qubit(item):
        if item in ( t2_Xancilla_ids + t2_Zancilla_ids):
            item = int(str(item)[:-1] + '3')
            qubit_name = id_to_qubit.get(item)
            #print(qubit_name)
            aso_qubit_name = get_qubit_neighbors(diagonal_neighbors,qubit_name)
            aso_qubit_ids = [qubit_map[q] for q in aso_qubit_name]
            return aso_qubit_ids
        elif item in ( mid_Xancilla_ids + mid_Zancilla_ids):
            item = int(str(item)[:-1] + '2')
            qubit_name = id_to_qubit.get(item)
            #print(qubit_name)
            aso_qubit_name = get_qubit_neighbors(diagonal_neighbors,qubit_name)
            aso_qubit_ids = [qubit_map[q] for q in aso_qubit_name]
            return aso_qubit_ids  


    for item in x_ancilla:
        index = measurement_order.index(item)
        backwards_reference = -(len(measurement_order) - index)
        aso_qubit = get_aso_qubit(item)
        if aso_qubit is not None:
            aso_qubit = aso_qubit[0]
            instruction= f"CX rec[{backwards_reference}] {aso_qubit}"
            #print(instruction)
            circuit += stim.Circuit(instruction)



        if item in mid_Xancilla_ids and aso_qubit is not None:
            #print(id_to_qubit.get(aso_qubit))
            matching_entries = [entry for entry in diagonal_neighbors if entry[3] == id_to_qubit.get(aso_qubit)]   
            if len(matching_entries) != 0 and qubit_map[matching_entries1[0][1]] != item: 
                extra_correction = qubit_map[matching_entries[0][1]]
                instruction= f"CX rec[{backwards_reference}] {extra_correction}"
                circuit += stim.Circuit(instruction)

        if item in mid_Xancilla_ids and aso_qubit is not None:
            #print(id_to_qubit.get(aso_qubit))
            matching_entries1 = [entry for entry in diagonal_neighbors if entry[2] == id_to_qubit.get(aso_qubit)]   
            matching_entries2 = [entry for entry in diagonal_neighbors if entry[5] == id_to_qubit.get(aso_qubit)]   
            #print(f"ENTRY IS{matching_entries1}")
            #print(f"ENTRY2 IS{matching_entries2}")
            if len(matching_entries1) != 0 and int(str(qubit_map[matching_entries1[0][1]])[:-1] +"0") != item: 
                extra_correction1 = qubit_map[matching_entries1[0][1]]
                instruction= f"CX rec[{backwards_reference}] {extra_correction1}"
                circuit += stim.Circuit(instruction) 


        if item in t2_Xancilla_ids and aso_qubit is not None:
            #print(id_to_qubit.get(aso_qubit))
            matching_entries1 = [entry for entry in diagonal_neighbors if entry[3] == id_to_qubit.get(aso_qubit)]   
            #print(f"ENTRY IS{matching_entries1}")
            if len(matching_entries1) != 0 and int(str(qubit_map[matching_entries1[0][1]])[:-1] +"2") != item: 
                    extra_correction1 = qubit_map[matching_entries1[0][1]]
                    instruction= f"CX rec[{backwards_reference}] {extra_correction1}"
                    circuit += stim.Circuit(instruction)




    for item in z_ancilla:
        index = measurement_order.index(item)
        backwards_reference = -(len(measurement_order) - index)
        aso_qubit = get_aso_qubit(item)
        #print(aso_qubit)
        if aso_qubit is not None:
            aso_qubit = aso_qubit[0]
            instruction= f"CZ rec[{backwards_reference}] {aso_qubit}"
            #print(instruction)
            circuit += stim.Circuit(instruction)

      
        if item in mid_Zancilla_ids and aso_qubit is not None:
            #print(id_to_qubit.get(aso_qubit))
            matching_entries1 = [entry for entry in diagonal_neighbors if entry[2] == id_to_qubit.get(aso_qubit)]   
            matching_entries2 = [entry for entry in diagonal_neighbors if entry[4] == id_to_qubit.get(aso_qubit)]   
            #print(f"ENTRY IS{matching_entries1}")
            #print(f"ENTRY2 IS{matching_entries2}")
            if len(matching_entries1) != 0 and int(str(qubit_map[matching_entries1[0][1]])[:-1] +"0") != item: 
                extra_correction1 = qubit_map[matching_entries1[0][1]]
                instruction= f"CX rec[{backwards_reference}] {extra_correction1}"
                circuit += stim.Circuit(instruction)


            if len(matching_entries2) != 0 and qubit_map[matching_entries1[0][1]] != item: 
       

                extra_correction2 = qubit_map[matching_entries2[0][1]]

                instruction= f"CX rec[{backwards_reference}] {extra_correction2}"
                circuit += stim.Circuit(instruction)
        if item in t2_Zancilla_ids and aso_qubit is not None:
            #print(id_to_qubit.get(aso_qubit))
            matching_entries1 = [entry for entry in diagonal_neighbors if entry[3] == id_to_qubit.get(aso_qubit)]   
            #print(f"ENTRY IS{matching_entries1}")
            if len(matching_entries1) != 0 and int(str(qubit_map[matching_entries1[0][1]])[:-1] +"2") != item: 
                    extra_correction1 = qubit_map[matching_entries1[0][1]]
                    instruction= f"CX rec[{backwards_reference}] {extra_correction1}"
                    circuit += stim.Circuit(instruction)

# Main function to generate the circuit
def generate_circuit(data, noise, noise2, rounds, CAT,inverted,zigzag, a ,b,c,d):
    circuit = stim.Circuit()
    
    # Map qubit IDs to numbers
    qubit_map = map_qubit_ids(data['cut_matrix'])
    
    # Initialize qubits and get their IDs
    qubit_ids = initialize_qubits(circuit, data['cut_matrix'], qubit_map)
    
    # Reset qubits
    for i in range(rounds):
        reset_qubits(circuit, qubit_ids, noise, noise2, i)

        if CAT and not zigzag:
            # CAT STATE PREP
            create_cat_state(circuit, data['T1_Ancillas'], data['T2_Ancillas'], data['T3_Ancillas'], qubit_map, noise,noise2)
        if CAT and zigzag:
            create_cat_state(circuit,data['CB_Ancillas'],data['BB_Ancillas'], data['T1_Ancillas'], data['T2_Ancillas'], data['T3_Ancillas'], qubit_map, noise,noise2)
            #create_cat_state_zigzag(circuit, data['CB_Ancillas'],data['BB_Ancillas'], qubit_map, noise,noise2)        
        else:
            # Create Bell pairs
            create_bell_pairs(circuit, data['T1_Ancillas'], data['T2_Ancillas'], data['T3_Ancillas'], qubit_map, noise,noise2)

        # Apply Hadamard gates to all X ancillas
        apply_hadamard_x_ancillas(circuit, data['X_Ancillas'], data['T1_Ancillas'], qubit_map, noise, noise2)

        # Create stabilizers
        create_stabilizers_direction(circuit, data, qubit_map, noise,noise2, a,inverted)
        create_stabilizers_direction(circuit, data, qubit_map, noise,noise2, b,inverted)
        create_stabilizers_direction(circuit, data, qubit_map, noise,noise2, c,inverted)
        create_stabilizers_direction(circuit, data, qubit_map, noise,noise2, d,inverted)


        apply_hadamard_x_ancillas(circuit, data['X_Ancillas'], data['T1_Ancillas'], qubit_map, noise, noise2)

        if not CAT:
            # Apply Hadamard gates to all T2 ancillas
            apply_hadamard_t2(circuit, data['T2_Ancillas'], qubit_map, noise)

        measurement_order= Classical_correction1(circuit, data['CB_Ancillas'],data['T1_Ancillas'], qubit_map,noise)   
        if inverted:
            Classical_correction2_inverted(circuit,data, data['CB_Ancillas'],data['T1_Ancillas'], qubit_map,noise,measurement_order)
        else:
            Classical_correction2(circuit,data, data['CB_Ancillas'],data['T1_Ancillas'], qubit_map,noise,measurement_order)

        measurement_order  += apply_measure_and_reset(circuit, data, qubit_map, noise, noise2)
        #print(measurement_order)

        if i == 0:
            # Add Z detectors
            add_z_detectors(circuit, data, qubit_map, measurement_order, i,CAT)
        else:
            add_z_detectors(circuit, data, qubit_map, measurement_order, i,CAT)
            # Add X detectors
            add_x_detectors(circuit, data, qubit_map, measurement_order, i,CAT)
   
    measure_q_qubits(circuit, qubit_map, data, measurement_order, noise)

    # Add final detectors
    add_final_detectors(circuit, data, qubit_map, measurement_order,CAT)
    
    # Include observable
    include_observable(circuit, data, qubit_map, measurement_order,rounds,CAT,inverted)

    return circuit





def generate_surface_code_matrix_with_shifts(n, m):
    rows = 2 * n + 1
    cols = 2 * m + 1
    
    # Initialize the matrix with '#' (representing empty spaces)
    matrix = np.full((rows, cols), '#', dtype=object)
    
    q_counter = 1  # Counter for qubits
    b_counter = 1  # Counter for bulk ancillas
    a_counter = 1  # Counter for auxiliary ancillas
    
    # Place physical qubits (Q)
    qubit_positions = {}
    for i in range(1, rows, 2):
        for j in range(1, cols, 2):
            matrix[i, j] = f'Q{q_counter}'
            qubit_positions[f'Q{q_counter}'] = (i, j)
            q_counter += 1
    
    # Place bulk ancillas (B)
    for i in range(2, rows-2, 2):
        for j in range(2, cols-2, 2):
            matrix[i, j] = f'B{b_counter}'
            b_counter += 1

    # Create the boundary qubit list
    boundary_qubits = []

    # Top row (left to right)
    for j in range(1, cols, 2):
        boundary_qubits.append(f'Q{matrix[1, j][1:]}')

    # Right column (top to bottom)
    for i in range(3, rows, 2):
        boundary_qubits.append(f'Q{matrix[i, cols - 2][1:]}')

    # Bottom row (right to left)
    for j in range(cols - 2, 0, -2):
        boundary_qubits.append(f'Q{matrix[rows - 2, j][1:]}')

    # Left column (bottom to top)
    for i in range(rows - 4, 0, -2):
        boundary_qubits.append(f'Q{matrix[i, 1][1:]}')

    # Remove duplicates from the boundary list
    boundary_qubits = list(dict.fromkeys(boundary_qubits))

    
    # Place auxiliary ancillas (A) between pairs of boundary qubits with shifts
    for k in range(1, len(boundary_qubits), 2):
        q1 = boundary_qubits[k - 1]
        q2 = boundary_qubits[k]

        # Get positions of q1 and q2
        q1_pos = qubit_positions[q1]
        q2_pos = qubit_positions[q2]
        
        if q1_pos[0] == q2_pos[0]:  # Same row
            ancilla_position = (q1_pos[0], (q1_pos[1] + q2_pos[1]) // 2)
            if ancilla_position[0] == 1:
                ancilla_position = (0, ancilla_position[1])  # Shift to top row
            elif ancilla_position[0] == rows - 2:
                ancilla_position = (rows - 1, ancilla_position[1])  # Shift to bottom row
        else:  # Same column
            ancilla_position = ((q1_pos[0] + q2_pos[0]) // 2, q1_pos[1])
            if ancilla_position[1] == 1:
                ancilla_position = (ancilla_position[0], 0)  # Shift to left column
            elif ancilla_position[1] == cols - 2:
                ancilla_position = (ancilla_position[0], cols - 1)  # Shift to right column

        # Place auxiliary ancilla (A)
        matrix[ancilla_position] = f'A{a_counter}'
        a_counter += 1
    
    return matrix

def make_vertical_cut(matrix, c, spacing,CAT=False,zigzag=False):
    if c == 0:
        return matrix
    rows, cols = matrix.shape
    keep_columns = 2 * c
    if keep_columns >= cols:
        raise ValueError("Not enough columns in the matrix to make the cut.")

    # Create a new matrix with the additional columns
    new_matrix = np.full((rows, cols + spacing), '#', dtype=object)

    # Copy the initial unchanged part of the matrix
    for i in range(rows):
        if i%4== 3 and zigzag:
            new_matrix[i, :keep_columns+2] = matrix[i, :keep_columns+2]
        else:
            new_matrix[i, :keep_columns] = matrix[i, :keep_columns]

    # Determine the column to modify and copy
    modify_column = keep_columns

    # Change IDs in the modify column to begin with 'T'
    for i in range(rows):
        if isinstance(matrix[i, modify_column], str) and matrix[i, modify_column][0] in {'A', 'B'}:
            new_matrix[i, modify_column] = 'T' + f'{i+1}' 
        else:
            new_matrix[i, modify_column] = matrix[i, modify_column]

    # Make 3 copies of the modify column
    range_count = 1 if CAT else 2

    for i in range(rows):
        if isinstance(matrix[i, modify_column], str) and matrix[i, modify_column][0] in {'A', 'B'}:
            og= new_matrix[i, modify_column]
            for k in range(range_count):
                new_matrix[i, modify_column + spacing -k] = og + f'{3-k}'
            new_matrix[i, modify_column] = new_matrix[i, modify_column] +'1'

    # Copy the rest of the matrix, adjusting for the added columns
    for i in range(rows):
        if i%4 ==3 and zigzag:
            new_matrix[i, modify_column + spacing+2 :cols + spacing] = matrix[i, modify_column + 2:cols]
        else:
            new_matrix[i, modify_column + spacing+1 :cols + spacing] = matrix[i, modify_column + 1:cols]


    # Copy Additional Ancilla For ZigZag Qubits
    for i in range(rows):
        if i%4==3 and zigzag:
            new_matrix[i-1, keep_columns+2] = matrix[i-1, keep_columns+2]
            new_matrix[i+1, keep_columns+2] = matrix[i+1, keep_columns+2]

    # Change IDs in the modify column to begin with 'T'
    if zigzag:
        for i in range(rows):
            if isinstance(matrix[i, modify_column+2], str) and matrix[i, modify_column+2][0] in {'B'}:
                new_matrix[i, modify_column+2] = 'C' + f'{new_matrix[i, modify_column+2]}' 
            
        for i in range(rows):
            if isinstance(matrix[i, modify_column+2], str) and matrix[i, modify_column+2][0] in {'B'}:
                new_matrix[i, modify_column+2 + spacing] = 'B' + f'{new_matrix[i, modify_column+2+spacing]}' 


    return new_matrix

def classify_ancillas(matrix, inverted=False):
    rows, cols = matrix.shape
    B_Ancillas = []
    A_Ancillas = []
    T3_Ancillas = []
    T2_Ancillas = []
    T1_Ancillas = []
    Z_Ancillas = []
    X_Ancillas = []
    BB_Ancillas = []
    CB_Ancillas = []
    Qubits = []

    for i in range(rows):
        for j in range(cols):
            if isinstance(matrix[i, j], str) and matrix[i, j][0] in {'A'}:
                A_Ancillas.append(matrix[i, j])
            if isinstance(matrix[i, j], str) and matrix[i, j][0] in {'B'} and matrix[i,j][1] not in {'B'}:
                B_Ancillas.append(matrix[i, j])
            if isinstance(matrix[i, j], str) and matrix[i, j][0] in {'T'} and matrix[i, j][-1] == '3':
                T3_Ancillas.append(matrix[i, j])
            if isinstance(matrix[i, j], str) and matrix[i, j][0] in {'T'} and matrix[i, j][-1] == '2':
                T2_Ancillas.append(matrix[i, j])
            if isinstance(matrix[i, j], str) and matrix[i, j][0] in {'T'} and matrix[i, j][-1] == '1':
                T1_Ancillas.append(matrix[i, j])
            if isinstance(matrix[i, j], str) and matrix[i, j][0] in {'Q'}:
                Qubits.append(matrix[i, j])
            if isinstance(matrix[i, j], str) and matrix[i,j].startswith('BB'):
                BB_Ancillas.append(matrix[i, j])
            if isinstance(matrix[i, j], str) and matrix[i,j].startswith('CB'):
                CB_Ancillas.append(matrix[i, j])


    if inverted:
        Z_Ancillas = []
        X_Ancillas = []
        for i in range(2, rows - 1):
            counter = ((i / 2) - 1) % 2
            for j in range(cols):
                if isinstance(matrix[i, j], str) and matrix[i, j][0] in {'B'}:
                    if counter % 2 == 0:
                        Z_Ancillas.append(matrix[i, j])
                        counter += 1
                    else:
                        X_Ancillas.append(matrix[i, j])
                        counter += 1
                elif isinstance(matrix[i, j], str) and matrix[i, j][0] in {'T'} and matrix[i, j][-1] == '1':
                    if counter % 2 == 0:
                        Z_Ancillas.append(matrix[i, j])
                        counter += 1
                    else:
                        X_Ancillas.append(matrix[i, j])
                        counter += 1

        for i in (0, rows - 1):
            for j in range(cols):
                if isinstance(matrix[i, j], str) and matrix[i, j][0] in {'A'}:
                    X_Ancillas.append(matrix[i, j])
                elif isinstance(matrix[i, j], str) and matrix[i, j][0] in {'T'} and matrix[i, j][-1] == '1':
                    X_Ancillas.append(matrix[i, j])

        for i in range(0, rows):
            for j in (0, cols - 1):
                if isinstance(matrix[i, j], str) and matrix[i, j][0] in {'A'}:
                    Z_Ancillas.append(matrix[i, j])
                elif isinstance(matrix[i, j], str) and matrix[i, j][0] in {'T'} and matrix[i, j][-1] == '1':
                    Z_Ancillas.append(matrix[i, j])
    else:
        X_Ancillas = []
        Z_Ancillas = []
        for i in range(2, rows - 1):
            counter = ((i / 2) - 1) % 2
            for j in range(cols):
                if isinstance(matrix[i, j], str) and matrix[i, j][0] in {'B'}:
                    if counter % 2 == 0:
                        X_Ancillas.append(matrix[i, j])
                        counter += 1
                    else:
                        Z_Ancillas.append(matrix[i, j])
                        counter += 1
                elif isinstance(matrix[i, j], str) and matrix[i, j][0] in {'T'} and matrix[i, j][-1] == '1':
                    if counter % 2 == 0:
                        X_Ancillas.append(matrix[i, j])
                        counter += 1
                    else:
                        Z_Ancillas.append(matrix[i, j])
                        counter += 1

        for i in (0, rows - 1):
            for j in range(cols):
                if isinstance(matrix[i, j], str) and matrix[i, j][0] in {'A'}:
                    Z_Ancillas.append(matrix[i, j])
                elif isinstance(matrix[i, j], str) and matrix[i, j][0] in {'T'} and matrix[i, j][-1] == '1':
                    Z_Ancillas.append(matrix[i, j])

        for i in range(0, rows):
            for j in (0, cols - 1):
                if isinstance(matrix[i, j], str) and matrix[i, j][0] in {'A'}:
                    X_Ancillas.append(matrix[i, j])
                elif isinstance(matrix[i, j], str) and matrix[i, j][0] in {'T'} and matrix[i, j][-1] == '1':
                    X_Ancillas.append(matrix[i, j])

    for element in Z_Ancillas.copy():  # Use a copy to avoid modifying the list while iterating
        if element.startswith('T') and element.endswith('1'):
            # Create a new element with the last digit changed to '3'
            new_element = element[:-1] + '3'
            # Append the new element to Z_Ancillas
            Z_Ancillas.append(new_element)

    for element in X_Ancillas.copy():  # Use a copy to avoid modifying the list while iterating
        if element.startswith('T') and element.endswith('1'):
            # Create a new element with the last digit changed to '3'
            new_element = element[:-1] + '3'
            # Append the new element to X_Ancillas
            X_Ancillas.append(new_element)

    for element in Z_Ancillas.copy():  # Use a copy to avoid modifying the list while iterating
        if element.startswith('BB'):
            # Create a new element with the last digit changed to '3'
            new_element = 'C' + element[1:]
            # Append the new element to Z_Ancillas
            Z_Ancillas.append(new_element)

    for element in X_Ancillas.copy():  # Use a copy to avoid modifying the list while iterating
        if element.startswith('BB'):
            # Create a new element with the last digit changed to '3'
            new_element = 'C' + element[1:]
            # Append the new element to X_Ancillas
            X_Ancillas.append(new_element)


    return X_Ancillas, Z_Ancillas, B_Ancillas, T3_Ancillas, T2_Ancillas, T1_Ancillas, A_Ancillas, Qubits, CB_Ancillas,BB_Ancillas


def find_diagonal_neighbors_with_type(matrix, X_Ancillas, Z_Ancillas):
    rows, cols = matrix.shape
    diagonal_neighbors = []

    for i in range(rows):
        for j in range(cols):
            if isinstance(matrix[i, j], str) and matrix[i, j][0] in {'A', 'B', 'T','C'}:
                neighbors = []
                # North-West
                if i > 0 and j > 0:
                    neighbors.append(matrix[i-1, j-1])
                else:
                    neighbors.append(None)
                
                # North-East
                if i > 0 and j < cols - 1:
                    neighbors.append(matrix[i-1, j+1])
                else:
                    neighbors.append(None)
                
                # South-East
                if i < rows - 1 and j < cols - 1:
                    neighbors.append(matrix[i+1, j+1])
                else:
                    neighbors.append(None)
                
                # South-West
                if i < rows - 1 and j > 0:
                    neighbors.append(matrix[i+1, j-1])
                else:
                    neighbors.append(None)
                
                # Determine type and create a tuple with the element and its neighbors
                ancilla_type = 'X' if matrix[i, j] in X_Ancillas else 'Z' if matrix[i, j] in Z_Ancillas else ''
                diagonal_neighbors.append((ancilla_type, matrix[i, j], *neighbors))
    
    return diagonal_neighbors

def draw_surface_code_svg(matrix, x_ancillas, z_ancillas, n, m):
    rows, cols = matrix.shape
    cell_size = 40
    legend_height = 80  # Increased space for the legend at the bottom
    legend_cell_size = 10
    title_height = 40   # Space for the title at the top
    dwg_height = rows * cell_size + legend_height + title_height
    dwg = svgwrite.Drawing(size=(cols * cell_size, dwg_height))

    # Add title
    title = f"Modular Surface Code {n}x{m}"
    dwg.add(dwg.text(title, insert=(cols * cell_size / 2, title_height / 2),
                     text_anchor="middle", font_size=20, dominant_baseline="middle"))

    # Draw the matrix
    for i in range(rows):
        for j in range(cols):
            if matrix[i, j] != '#':
                # Determine the fill color
                if matrix[i, j].startswith('Q'):
                    fill_color = 'grey'
                elif matrix[i, j].startswith('T') and matrix[i, j].endswith('1'):
                    fill_color = 'green'
                elif matrix[i, j] in x_ancillas:
                    fill_color = 'red'
                elif matrix[i, j] in z_ancillas:
                    fill_color = 'blue'
                else:
                    fill_color = 'green'
                # Draw the rectangle
                dwg.add(dwg.rect((j * cell_size, i * cell_size + title_height), (cell_size, cell_size), fill=fill_color, stroke='black'))
                # Add the text
                dwg.add(dwg.text(matrix[i, j], insert=((j + 0.5) * cell_size, (i + 0.5) * cell_size + title_height),
                                 text_anchor="middle", dominant_baseline="middle", font_size=12, fill='white'))


    # Determine the legend annotation for green
    contains_t = any(cell.startswith('T') for row in matrix for cell in row if cell != '#')
    green_annotation = 'Bell Pair' if contains_t else 'Cut Along'

    # Add legend
    legend_y = rows * cell_size + title_height + 10
    legend_items = [
        ('grey', 'Physical Qubits'),
        ('red', 'X Ancilla'),
        ('blue', 'Z Ancilla'),
        ('green', green_annotation)
    ]
    for idx, (color, label) in enumerate(legend_items):
        x_start = 10
        y_start = legend_y + idx * (legend_cell_size + 5)
        dwg.add(dwg.rect((x_start, y_start), (legend_cell_size, legend_cell_size), fill=color, stroke='black'))
        dwg.add(dwg.text(label, insert=(x_start + legend_cell_size + 10, y_start + legend_cell_size / 2),
                         text_anchor="start", dominant_baseline="middle", font_size=12, fill='black'))

    return dwg.tostring()


def surface_data(n, m, c=0, spacing=3, display_svg=False, inverted=False,CAT=False,zigzag=False):
    initial_matrix = generate_surface_code_matrix_with_shifts(n, m)
    cut_matrix = make_vertical_cut(initial_matrix, c, spacing,CAT,zigzag)
    X_Ancillas, Z_Ancillas, B_Ancillas, T3_Ancillas, T2_Ancillas, T1_Ancillas, A_Ancillas, Qubits,CB_Ancillas,BB_Ancillas = classify_ancillas(cut_matrix, inverted)
    diagonal_neighbors = find_diagonal_neighbors_with_type(cut_matrix, X_Ancillas, Z_Ancillas)

    result = {
        "initial_matrix": initial_matrix,
        "cut_matrix": cut_matrix,
        "X_Ancillas": X_Ancillas,
        "Z_Ancillas": Z_Ancillas,
        "B_Ancillas": B_Ancillas,
        "T3_Ancillas": T3_Ancillas,
        "T2_Ancillas": T2_Ancillas,
        "T1_Ancillas": T1_Ancillas,
        "A_Ancillas": A_Ancillas,
        "CB_Ancillas": CB_Ancillas,
        "BB_Ancillas": BB_Ancillas,
        "Qubits": Qubits,
        "diagonal_neighbors": diagonal_neighbors
        
    }

    if display_svg:
        initial_svg = draw_surface_code_svg(initial_matrix, X_Ancillas, Z_Ancillas, n, m)
        cut_svg = draw_surface_code_svg(cut_matrix, X_Ancillas, Z_Ancillas,n,m)
        display(SVG(initial_svg))
        display(SVG(cut_svg))

    return result

