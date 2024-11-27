from imports import *


def parse_pdb_and_calculate_distance_matrix(pdb_file):
    """Parses a PDB file, calculates the distance matrix between CA atoms, and extracts the amino acid sequence.

    Args:
        pdb_file: Path to the PDB file.

    Returns:
        A tuple containing the distance matrix and the amino acid sequence.
    """

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    model = structure[0]

    ca_atoms = []
    sequence = []
    for chain in model:
        for residue in chain:
            if residue.has_id("CA"):
                ca_atoms.append(residue["CA"])
                sequence.append(residue.resname)

    # Convert sequence to one-letter code (handle missing CA atoms)
    one_letter_code = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
    }
    sequence = [one_letter_code.get(res, 'X') for res in sequence]

    # Calculate distance matrix
    num_atoms = len(ca_atoms)
    distance_matrix = np.zeros((num_atoms, num_atoms))
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            distance = np.linalg.norm(ca_atoms[i].coord - ca_atoms[j].coord)
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    sequence = "".join(sequence)
    return distance_matrix, sequence


def calculate_distance_matrix(pdb_file):
    """Calculates the distance matrix for a given PDB file.

    Args:
        pdb_file: Path to the PDB file.`

    Returns:
        A NumPy array representing the distance matrix.
    """

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    model = structure[0]

    # Extract CA atoms
    ca_atoms = [atom for atom in model.get_atoms() if atom.name == 'CA']

    # Calculate pairwise distances
    num_atoms = len(ca_atoms)
    distance_matrix = np.zeros((num_atoms, num_atoms))
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            distance_matrix[i, j] = distance_matrix[j, i] = ca_atoms[i] - ca_atoms[j]

    return distance_matrix


def create_contact_map(distance_matrix, threshold=8.0):
    """Creates a contact map from a distance matrix.

  Args:
    distance_matrix: A NumPy array representing the distance matrix.
    threshold: The distance threshold for defining a contact.

  Returns:
    A NumPy array representing the contact map.
  """

    contact_map = np.zeros_like(distance_matrix)
    contact_map[distance_matrix <= threshold] = 1
    return contact_map


def get_sequence_from_pdb(pdb_file):
    """Extracts the amino acid sequence from a PDB file.

    Args:
        pdb_file: Path to the PDB file.

    Returns:
        A string representing the amino acid sequence.
    """

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    model = structure[0]

    sequence = ""
    for chain in model:
        for residue in chain:
            sequence += residue.resname

    return sequence


def convert_3_to_1_letter(sequence):
    """Converts a protein sequence from three-letter to one-letter codes.

        Args:
            sequence: A string containing the protein sequence in three-letter code.

        Returns:
            A string containing the protein sequence in one-letter code.
    """

    three_to_one = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }

    one_letter_sequence = ""
    for i in range(0, len(sequence), 3):
        three_letter_code = sequence[i:i + 3]
        if three_letter_code in three_to_one:
            one_letter_sequence += three_to_one[three_letter_code]
    return one_letter_sequence


# inputs for next function
model_name = "esm2_t6_8M_UR50D"
model_ems_2, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
model_ems_2.eval()


def get_ems2_embeddings(sequence):
    """Converts a protein sequence from three-letter to one-letter codes.

        Args:
            sequence: A string containing the protein sequence in three-letter code.
        Returns:
            A NumPy array representing the ems2 embedding.
    """

    batch_converter = alphabet.get_batch_converter()
    data = [("protein1", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    with torch.no_grad():
        results = model_ems_2(batch_tokens, repr_layers=[6])
        token_embeddings = results["representations"][6]

    return token_embeddings


def pad_matrices_with_mask(matrices, padding_value=0):
    """
    Pads a list of 2D matrices (torch tensors or NumPy arrays) to the same shape,
    creates a mask for padding locations, and stacks them into a batch tensor.

    Args:
        matrices (list of torch.Tensor or np.ndarray): List of 2D matrices to pad.
        padding_value (int, float): Value used for padding.

    Returns:
        torch.Tensor: A batch tensor with all matrices padded to the same shape.
        torch.Tensor: A mask tensor where 1 indicates padding locations and 0 indicates original data locations.
    """
    # Convert all matrices to torch tensors
    matrices = [torch.tensor(matrix) if isinstance(matrix, np.ndarray) else matrix for matrix in matrices]

    # Find the maximum height and width
    max_height = max(matrix.shape[0] for matrix in matrices)
    max_width = max(matrix.shape[1] for matrix in matrices)

    # Pad matrices to the same shape and create a mask
    padded_matrices = []
    masks = []

    for matrix in matrices:
        # Calculate padding amounts
        pad_top = 0
        pad_bottom = max_height - matrix.shape[0]
        pad_left = 0
        pad_right = max_width - matrix.shape[1]

        # Pad the matrix
        padded_matrix = torch.nn.functional.pad(
            matrix,
            (pad_left, pad_right, pad_top, pad_bottom),  # Pad (left, right, top, bottom)
            value=padding_value
        )

        # Create the mask (1 where padding, 0 where original data)
        mask = torch.ones_like(padded_matrix, dtype=torch.bool)
        mask[:, :matrix.shape[1]] = False  # Set to 0 in original data columns
        mask[:matrix.shape[0], :] = False  # Set to 0 in original data rows

        # Append padded matrix and mask
        padded_matrices.append(padded_matrix)
        masks.append(mask)

    # Stack the padded matrices and masks into batch tensors
    padded_batch = torch.stack(padded_matrices, dim=0)
    mask_batch = torch.stack(masks, dim=0)

    return padded_batch, mask_batch
