# words segmentation
# word_folder_path = r"D:\Digital Image Processing\Project\Project_code_folder\word_imgs"

import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
# Load the grayscale image
def image_thresolding(image):
    # image = cv2.imread('image_2_IAM.jpg', cv2.IMREAD_GRAYSCALE)

    # Apply adaptive thresholding
    # You can experiment with the 'blockSize' and 'C' values for optimal results
    adaptive_thresh = cv2.adaptiveThreshold(
        image, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        blockSize=15,  # Size of the neighborhood area (must be odd)
        C=10  # Constant subtracted from the mean
    )
    return adaptive_thresh

def cols_without_text(adaptive_thresh, text_threshold=2):
    h_projection = np.sum(adaptive_thresh == 0, axis=0)  # Sum binary values along cols
    # some_threshold=30
    # Find rows with low values (i.e., likely to be between lines)
    # Experiment with a threshold for gap detection based on the profile
    line_gaps = np.where(h_projection < text_threshold)[0]
    return line_gaps

# get the final line based on thresold for the number of lines between the two horigental line
# average number of line calculation
    # thresold_width = 0.5*avg_line_width,     0.5 need to be automated or manual entred based on documents

def line_type(image, line_gaps):
    final_lines=[]
    bunch_lines= []
    line_width =[]
    n_line =0
    for i in range(1, len(line_gaps)):
        if line_gaps[i]-line_gaps[i-1]<=1:
            n_line+=1
        else:
            line_width.append(n_line)
            n_line=0 
    # last_line  
    avg_line_width = np.mean(line_width)
    # print("number of isolated lines:",line_width)
    # print('avg_line_width',avg_line_width)
    # print("max line_width:", max(line_width))
    if len(line_width)>20:
        thresold_width = avg_line_width
        # print("Identified as non cursived text")
        return 0
    else: 
        thresold_width = avg_line_width*0.60
        # print("Identified as cursived text")
        return 1

def is_cursived(output__line_folder_path, prefix):
    ls_type = []
    n_lines = 0
    for filename in os.listdir(output__line_folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            if prefix and not filename.startswith(prefix):
                continue  # Skip files that don't match the prefix
            line_img_path = os.path.join(output__line_folder_path, filename)
            n_lines+=1
            image = cv2.imread(line_img_path, cv2.IMREAD_GRAYSCALE)
            # image = cv2.imread(line_folder_path+f'{os.listdir(line_folder_path)[line_number]}', cv2.IMREAD_GRAYSCALE)
            plt.imshow(image, cmap='gray')
            threshoded_img = image_thresolding(image)

            line_gaps = cols_without_text(threshoded_img, text_threshold=1)
            # print(line_gaps)
            l_type = line_type(image,line_gaps)
            ls_type.append(l_type)

    if np.sum(ls_type)>(n_lines/2):
        print("Identified as cursived text")
        return 1 #cursived

    else:
        print("Identified as non cursived text")
        return 0 #Not cursived
