from Bio.Blast.Applications import NcbiblastpCommandline  # noqa

from imports import *
from pdb_files_utils import (get_ems2_embeddings, parse_pdb_and_calculate_distance_matrix, create_contact_map,
                             pad_matrices_with_mask)


class CostomDataset(Dataset):
    @staticmethod
    def get_files_by_extension(directory, ext='txt'):
        pattern = os.path.join(directory, '**', f'*.{ext}')
        return [os.path.relpath(path, directory) for path in glob.glob(pattern, recursive=True)]

    def __init__(self, data_path):
        self.data_path = data_path
        self.paths = self.get_files_by_extension(self.data_path, 'ent')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        path = os.path.join(self.data_path, path)
        amino_distance_matrix, amino_seq = parse_pdb_and_calculate_distance_matrix(path)
        amino_emb = get_ems2_embeddings(amino_seq)
        amino_contact_map = create_contact_map(amino_distance_matrix)
        contact_map_tensor = torch.tensor(amino_contact_map, dtype=torch.float32)
        return amino_emb, contact_map_tensor


class EmbeddingCollate:

    @staticmethod
    def create_batch_tensor(inputs, targets):
        lengths = [t.size(0) for t in inputs]
        max_len = max(lengths)
        mask_input = torch.arange(max_len).expand(len(lengths), max_len) < torch.tensor(lengths).unsqueeze(1)
        padded_input = pad_sequence(inputs, batch_first=True)
        padded_output, mask_output = pad_matrices_with_mask(targets)

        return padded_input, mask_input, padded_output, mask_output

    def __init__(self, model, alphabet):
        self.batch_converter = alphabet.get_batch_converter()
        self.model = model

    def __call__(self, batch):
        inputs, targets = [], []

        for embeddings, contact_map in batch:
            inputs.append(embeddings.squeeze(0)[1:-1])
            targets.append(contact_map.squeeze(0))

        return self.create_batch_tensor(inputs, targets)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_vram_usage(model, include_gradients=True):
    # Calculate total number of parameters
    total_params = count_parameters(model)
    param_memory = total_params * 4  # 4 bytes per float32 parameter

    # If gradients are stored (common in training), double the memory usage
    if include_gradients:
        param_memory *= 2

    # Convert to (GB)
    return param_memory / (1024 ** 3)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.2, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_factor * loss

        return loss.mean()


def focal_loss(pred, target, gamma=2., alpha=0.5, reduction="sum"):
    """"focal loss function
        Args:
            pred : prediction tensor, shape (N, C)
            target : target tensor, shape (N, C)
            gamma : focal loss parameter
            alpha : focal loss parameter
            reduction : 'sum' or 'mean'
        Returns:
            focal loss tensor, shape (N, C)
    """

    p_t = pred * target + (1 - pred) * (1 - target)
    ce_loss = torch.nn.functional.binary_cross_entropy(pred, target, reduction='none')
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        loss = alpha_t * loss
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss


def calculate_identity_levenshtein(seq1, seq2):
    """
    Calculate the percentage identity between two protein sequences using Levenshtein distance.

    Parameters:
    - seq1 (str): The first protein sequence.
    - seq2 (str): The second protein sequence.

    Returns:
    - float: The percentage identity between the two sequences.
    """
    # Calculate the Levenshtein distance
    distance = Levenshtein.distance(seq1, seq2)

    # Calculate percentage identity
    max_len = max(len(seq1), len(seq2))
    identity_percentage = ((max_len - distance) / max_len) * 100

    return identity_percentage


def calculate_row(file_name_1, file_list, dataset_path):
    """
    Calculate a row in the identity matrix for a specific file.
    """
    _, seq_1 = parse_pdb_and_calculate_distance_matrix(f"{dataset_path}/{file_name_1}")
    row = []
    for file_name_2 in file_list:
        _, seq_2 = parse_pdb_and_calculate_distance_matrix(f"{dataset_path}/{file_name_2}")
        identity = calculate_identity_levenshtein(seq_1, seq_2)
        row.append(identity)
        print(identity, file_name_1)
    return row


def calculate_identity_matrix_multithread(dataset_path, start=0, num_pairs=15):
    """
    Calculate the identity matrix using multithreading.

    Parameters:
    - dataset_path (str): Path to the dataset containing protein files.

    Returns:
    - list of lists: The identity matrix.
    """
    file_list = os.listdir(dataset_path)[start:start + num_pairs]

    # Use ThreadPoolExecutor to parallelize row calculations
    with ThreadPoolExecutor() as executor:
        results = executor.map(
            calculate_row,
            file_list,  # file_name_1 for each row
            [file_list] * len(file_list),  # Pass the entire file list
            [dataset_path] * len(file_list)  # Pass the dataset path
        )
        matrix = list(results)
    return matrix


def plot_similarity_matrix(similarity_matrix):
    similarity_matrix = np.array(similarity_matrix)

    # Plot the heatmap
    plt.figure(figsize=(20, 15))
    sns.heatmap(similarity_matrix, annot=True, fmt=".1f", cmap="viridis", cbar=True)

    # Customize the heatmap
    plt.title("Protein Identity Matrix", fontsize=16)
    plt.xlabel("Proteins", fontsize=14)
    plt.ylabel("Proteins", fontsize=14)
    plt.xticks(ticks=np.arange(len(similarity_matrix)), labels=[f"P{i + 1}" for i in range(len(similarity_matrix))],
               rotation=45)
    plt.yticks(ticks=np.arange(len(similarity_matrix)), labels=[f"P{i + 1}" for i in range(len(similarity_matrix))])

    # Show the plot
    plt.tight_layout()
    plt.show()


def deleting_low_similar_proteins(files_path="toy_dataset", protein_main="pdb106m.ent"):
    """ Delete the low similar proteins based on protein name
        Args:
            files_path (str, optional): Path to the dataset containing protein files. Defaults to "toy_dataset".
            protein_main (str, optional): Name of the protein to delete. Defaults to "pdb106m.ent".
        """
    _, seq_main = parse_pdb_and_calculate_distance_matrix(f"{files_path}/{protein_main}")
    similar_protein_names = []
    similarity_scores = []
    for i, file_name in enumerate(os.listdir(files_path)):
        _, seq_2 = parse_pdb_and_calculate_distance_matrix(f"{files_path}/{file_name}")
        identity = calculate_identity_levenshtein(seq_main, seq_2)
        if identity > 30:
            similar_protein_names.append(file_name)
            similarity_scores.append(identity)
        else:
            os.remove(f"{files_path}/{file_name}")
        if i % 100 == 0:
            print(f"delete less similar proteins in {i} files")

    print(f"number of similar proteins -{len(similar_protein_names)}", f"similarity score list - {similarity_scores}")


def calculate_roc_auc_with_inverted_mask(y_true, y_pred_proba, mask):
    """Calculates the ROC AUC score for a binary classification problem with inverted masks.

    Args:
      y_true: A PyTorch tensor of true binary labels (0 or 1) with shape (batch_size, seq_len).
      y_pred_proba: A PyTorch tensor of predicted probabilities for the positive class with shape (batch_size, seq_len).
      mask: A PyTorch tensor of the same shape as y_true, indicating valid data points with 0s and padding with 1s.

    Returns:
      The ROC AUC score.
    """

    # Flatten the tensors and mask
    y_true_flat = y_true.flatten()
    y_pred_proba_flat = y_pred_proba.flatten()
    mask_flat = mask.flatten()

    # Filter valid data (invert the mask)
    valid_indices = (mask_flat == 0).nonzero().squeeze()
    y_true_valid = y_true_flat[valid_indices]
    y_pred_proba_valid = y_pred_proba_flat[valid_indices]

    # Convert to NumPy arrays
    y_true_np = y_true_valid.cpu().numpy()
    y_pred_proba_np = y_pred_proba_valid.cpu().numpy()

    # Calculate ROC AUC
    roc_auc = roc_auc_score(y_true_np, y_pred_proba_np)
    return roc_auc


def plot_roc_curve(y_true, y_pred_proba, mask):
    """Plots the ROC curve for a binary classification task, considering masked data.

    Args:
      y_true: A NumPy array of true binary labels (0 or 1).
      y_pred_proba: A NumPy array of predicted probabilities for the positive class.
      mask: A boolean NumPy array indicating valid data points.

    Returns:
      None
    """
    y_true_flat = y_true.cpu().flatten()
    y_pred_proba_flat = y_pred_proba.cpu().flatten()
    mask_flat = mask.cpu().flatten()
    valid_indices = (mask_flat == 0).nonzero().squeeze()

    # Filter valid data
    y_true_valid = y_true_flat[valid_indices]
    y_pred_proba_valid = y_pred_proba_flat[valid_indices]

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true_valid, y_pred_proba_valid)
    roc_auc = roc_auc_score(y_true_valid, y_pred_proba_valid)

    # Plot ROC curve
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
