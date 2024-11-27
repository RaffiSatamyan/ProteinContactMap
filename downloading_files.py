from imports import *
from pdb_files_utils import get_sequence_from_pdb, convert_3_to_1_letter
from utils import deleting_low_similar_proteins

# Path to the file_list.txt (containing the list of file URLs or names)
file_list_path = "file_list.txt"
# Path
pdb_files_path = "pdb_files"
train_dataset = "train_dataset"
test_dataset = "test_dataset"
validation_dataset = "validation_dataset"

# Define the number of threads to use
num_threads = 5  # Adjust as needed


# Function to process a single URL
def process_url(url):
    # Clean up the line and remove extra spaces or newline characters
    url = url.strip()

    if url:  # If the line is not empty
        # Construct the full URL (if only filenames are in the list, prepend the base URL)
        full_url = f"https://files.rcsb.org/pub/pdb/data/structures/all/pdb/{url}"  # Modify if needed

        # Path to save the decompressed file locally (based on the filename)
        decompressed_file_path = f"pdb_files/{url[:-3]}"  # Decompressed content will be saved with .ent extension

        try:
            # Step 1: Fetch the .gz file content from the URL
            response = urlopen(full_url)
            compressed_data = response.read()  # Read data as bytes

            # Step 2: Decompress the content in memory and write to a file
            with BytesIO(compressed_data) as f:
                with gzip.open(f, 'rb') as decompressed_file:
                    decompressed_data = decompressed_file.read()

            # Step 3: Save the decompressed content to a file
            with open(decompressed_file_path, "wb") as f:
                f.write(decompressed_data)

            # Step 4: Remove DNA/RNA files
            seq_3letter = get_sequence_from_pdb(decompressed_file_path)
            sequence = convert_3_to_1_letter(seq_3letter)
            if 100 >= len(sequence) or len(sequence) >= 200:
                os.remove(decompressed_file_path)

            else:
                print(len(sequence), decompressed_file_path)
            # print(f"Processed: {url}")
        except Exception as e:
            print(f"Failed to process {url}: {e}")


# Read the lines from the file_list.txt
with open(file_list_path, "r") as file:
    lines = file.readlines()[:20000]

# Create a list of threads
threads = []
for line in lines:
    thread = threading.Thread(target=process_url, args=(line,))
    threads.append(thread)
    thread.start()

# Wait for all threads to finish
for thread in threads:
    thread.join()

print("All processing finished!")


# Deleting the files of proteins which is not similar with some "main" one to work only with similar proteins, so the
# Model can learn the patterns of contact map

deleting_low_similar_proteins(files_path="pdb_files")

# Number of all files

num_files = len(os.listdir(pdb_files_path))

# Separating data into train/test/validation
for i, file_name in enumerate(os.listdir(pdb_files_path)):
    if i <= int(0.6*num_files):
        destination = os.path.join(train_dataset, file_name)
    elif int(0.6*num_files) <= i <= int(0.8*num_files):
        destination = os.path.join(test_dataset, file_name)
    else:
        destination = os.path.join(validation_dataset, file_name)
    source = os.path.join(pdb_files_path, file_name)
    shutil.move(source, destination)
