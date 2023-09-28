import os

# Specify the directory path where you want to rename folders
directory_path = './data/dia_platform3/valid'

# List all folders in the directory
folders = [folder for folder in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, folder))]
folders.sort()
# Iterate through the folders and rename them
for old_folder_name in folders:
    # Define the new folder name (you can modify this as needed)
    new_folder_name = "{}".format( int(old_folder_name) -6000)

    # Create the full paths for the old and new folders
    old_folder_path = os.path.join(directory_path, old_folder_name)
    new_folder_path = os.path.join(directory_path, new_folder_name)

    # Rename the folder
    os.rename(old_folder_path, new_folder_path)

    print(f"Renamed: {old_folder_path} to {new_folder_path}")

print("All folders renamed successfully.")