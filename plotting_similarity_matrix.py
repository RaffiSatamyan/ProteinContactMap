from utils import calculate_identity_matrix_multithread, plot_similarity_matrix

# Plotting the similarity matrix
dataset_path = "toy_dataset"
matrix = calculate_identity_matrix_multithread(dataset_path)
plot_similarity_matrix(matrix)
