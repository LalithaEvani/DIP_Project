import os
import matplotlib.pyplot as plt
import argparse
import sys
sys.path.append("code")
from line_seg_1 import PPA_Algo_line
from word_seg_2 import PPA_Algo_word
import word_recog_3
from PIL import Image
import cv2

def process_image(input_img_path, output_base_path):
    # Extract the image name (without extension)
    image_name = os.path.splitext(os.path.basename(input_img_path))[0]
    print(f'image name {image_name}')

    # Create subdirectories for lines, words, and outputs
    output_line_folder_path = os.path.join(output_base_path, image_name, 'lines')
    output_word_folder_path = os.path.join(output_base_path, image_name, 'words')
    output_plot_folder_path = os.path.join(output_base_path, image_name, 'outputs')

    os.makedirs(output_line_folder_path, exist_ok=True)
    os.makedirs(output_word_folder_path, exist_ok=True)
    os.makedirs(output_plot_folder_path, exist_ok=True)

    # Run the line segmentation algorithm
    _, _, _, _, _, d_image = PPA_Algo_line(input_img_path, output_line_folder_path, image_name)
    
    #edit: added line to save segmented line image 
    cv2.imwrite(os.path.join(output_plot_folder_path, f'line_segmented.png'), d_image)
    
    # Initialize the handwritten text extractor
    word_r = word_recog_3.HandwrittenTextExtractor()

    # Collect and sort the line images
    # line_imgs = [f for f in os.listdir(output_line_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and f.startswith(image_name)]
    line_imgs = [f for f in os.listdir(output_line_folder_path)]
    line_imgs_sorted = sorted(line_imgs, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # Process each line image for word segmentation and recognition
    for n_lines, filename in enumerate(line_imgs_sorted):
        line_img_path = os.path.join(output_line_folder_path, filename)
        # line_image_name = f"{image_name}_{n_lines}"
        line_image_name = os.path.splitext(os.path.basename(filename))[0]
        # line_image_name = filename


        # Run the word segmentation algorithm
        _, _, _, _, _, _, d_image = PPA_Algo_word(line_img_path, output_word_folder_path, line_image_name)
        #edit: added line to save segmented line image 
        # Ensure the image is rotated 90 degrees clockwise (horizontal orientation)
        d_image_rotated = cv2.transpose(d_image)
        # d_image_rotated = cv2.flip(d_image_rotated, flipCode=1)  # flipCode=1 for horizontal flip
        d_image_rotated = cv2.flip(d_image_rotated, flipCode=0)  # flipCode=1 for horizontal flip
        # d_image_rotated = cv2.flip(d_image_rotated, flipCode=0)
        # Save the rotated image
        cv2.imwrite(os.path.join(output_plot_folder_path, f'{line_image_name}_word_segmented.png'), d_image_rotated)
            
        
        # Predict text for the line image
        results = word_r.predict_text(output_word_folder_path, line_image_name)
        results_sorted = sorted(results, key=lambda x: int(x[0].split('_')[-1].split('.')[0]))

        # Display the word segmentation results
        fig, axes = plt.subplots(1, len(results_sorted), figsize=(5 * len(results_sorted), 5))
        axes = axes.ravel()
        for i, ax in enumerate(axes):
            image_path = os.path.join(output_word_folder_path, results_sorted[i][0])
            image = Image.open(image_path)
            ax.imshow(image, cmap='gray')
            ax.set_title(f"P: {results_sorted[i][1]}", fontsize=40)
            ax.axis('off')

        plt.tight_layout()
        plot_output_path = os.path.join(output_plot_folder_path, f'{line_image_name}_prediction.png')
        plt.savefig(plot_output_path)
        plt.show()

if __name__ == "__main__":
    process_image("../data/renamed/14.png", "../output/")
