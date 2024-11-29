import os
import shutil

def rename_images(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # List all files in the input folder
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Sort the files for consistent ordering
    files.sort()
    
    # Rename and copy files to the output folder
    for i, file_name in enumerate(files, start=1):
        # Construct full paths
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, f"{i}.png")
        
        # Copy the image to the new location with the new name
        shutil.copy(input_path, output_path)
        print(f"Renamed: {file_name} -> {i}.png")

# Define the input and output folders
input_folder = "../../data/cropped/"
output_folder = "../../data/renamed/"

# Call the function
rename_images(input_folder, output_folder)
