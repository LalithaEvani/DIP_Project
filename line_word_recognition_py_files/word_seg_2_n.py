# words segmentation
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

# Load the grayscale image
def image_thresolding(image):
  
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
 
    line_gaps = np.where(h_projection < text_threshold)[0]
    return line_gaps

def find_final_lines(image, line_gaps, cursived_status):
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

    if cursived_status==0:
        thresold_width = avg_line_width
    
    else: 
        thresold_width = avg_line_width*0.60

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
    try:
        columns_after_last = image[:, final_lines[-1] + 1:]
        if np.any(np.sum(columns_after_last==0)>20):
            # print("yes added")
            final_lines.append(image.shape[1]-1)

    except:
        pass
    return final_lines
def save_the_words(image,line_number,lines,word_folder_path,prefix):
    for i in range(1, len(lines)):
        line_img = image[0:image.shape[1], lines[i-1]:lines[i]]
        w_image_path = os.path.join(word_folder_path,f'{prefix}_{i}.png')

        cv2.imwrite(w_image_path, line_img)
        print(f'{w_image_path} saved....')

def word_segmentation(line_img_path,cursived_status,line_number, word_folder_path, prefix):
    image = cv2.imread(line_img_path, cv2.IMREAD_GRAYSCALE)

    line_gaps = cols_without_text(image, text_threshold=1)
 
    final_lines = find_final_lines(image, line_gaps, cursived_status)
    result = image.copy()

    for x in final_lines:
        cv2.line(result, (x, 0), (x,result.shape[0]), (128,), 1)

    save_the_words(image,line_number,final_lines,word_folder_path,prefix)
    return result
