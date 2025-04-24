import os


def delete_dotfiles(directory):
    """
    Recursively deletes all files starting with a dot (.) in the given directory.

    Args:
        directory (str): The root directory to start the deletion.
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith("."):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")


if __name__ == "__main__":
    # Specify the directory to clean
    directory_to_clean = "data/valid"  # Replace with your target directory
    delete_dotfiles(directory_to_clean)
