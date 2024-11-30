# words segmentation
word_folder_path = r"D:\Digital Image Processing\Project\Project_code_folder\word_imgs"

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

def find_final_lines(line_gaps, cursived_status):
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
    if cursived_status==0:
        thresold_width = avg_line_width
        # print("Identified as non cursived text")
    else: 
        thresold_width = avg_line_width*0.60
        # print("Identified as cursived text")

    # thresold_width = line_threshold+avg_line_width/2
    n_line=0
    for i in range(1, len(line_gaps)):
        if line_gaps[i]-line_gaps[i-1]<=1 and i<(len(line_gaps)-1):
            bunch_lines.append(line_gaps[i-1])
            n_line+=1
        else:
            if thresold_width<=n_line:
                final_lines.append(np.mean(bunch_lines).astype(np.int32))
            bunch_lines=[]
            n_line=0

    columns_after_last = image[:, final_lines[-1] + 1:]
    if np.any(np.sum(columns_after_last==0)>20):
        # print("yes added")
        final_lines.append(image.shape[1]-1)
    return final_lines
def save_the_words(image,line_number,lines,word_folder_path,prefix):
    for i in range(1, len(lines)):
        line_img = image[0:image.shape[1], lines[i-1]:lines[i]]
        w_image_path = os.path.join(word_folder_path,f'{prefix}_{i}.png')

        cv2.imwrite(w_image_path, line_img)
        print(f'{w_image_path} saved....')



# line_folder_path = r"D:\Digital Image Processing\Project\git_hub\DIP_Project\line_imgs_paper\\"
# for line_number in range(1,1+len(os.listdir(line_folder_path))):
    # line_number=1
def word_segmentation(line_img_path,cursived_status,line_number, word_folder_path, prefix):
    image = cv2.imread(line_img_path, cv2.IMREAD_GRAYSCALE)
    # image = cv2.imread(line_folder_path+f'{os.listdir(line_folder_path)[line_number]}', cv2.IMREAD_GRAYSCALE)
    # plt.imshow(image, cmap='gray')
    # threshoded_img = image_thresolding(image)

    line_gaps = cols_without_text(image, text_threshold=1)
    # print(line_gaps)
    final_lines = find_final_lines(line_gaps, cursived_status)
    result = image.copy()
    # print("no of final lines:",len(final_lines))
    for x in final_lines:
        cv2.line(result, (x, 0), (x,result.shape[0]), (128,), 1)
    # plt.imshow(result, cmap='gray')
    # plt.show()

    save_the_words(image,line_number,final_lines,word_folder_path,prefix)
    return result
# find the method to find the automatic line_threshold and text_threshold