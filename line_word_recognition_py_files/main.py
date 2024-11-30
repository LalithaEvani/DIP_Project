import matplotlib.pyplot as plt
import cv2
import numpy as np
import argparse
import sys
sys.path.append("line_word_recognition_py_files")
from line_seg_1 import PPA_Algo_line
# from word_seg_2 import PPA_Algo_word
from check_img_type import is_cursived
from word_seg_2_n import word_segmentation
import word_recog_3
import os
from PIL import Image

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("arg1",type=str, help="input image path")
    parser.add_argument("arg2",type=str, help="input image name")
    parser.add_argument("arg3",type=str, help="output folder to save the lines")
    parser.add_argument("arg4",type=str, help="output folder to save the words")
    parser.add_argument("arg5",type=str, help="result folder")
    args=parser.parse_args()

    input_img_path = args.arg1
    prefix = args.arg2
    output__line_folder_path = args.arg3
    output__word_folder_path = args.arg4
    result_folder_path = args.arg5
    # output_folder = r"D:\Digital Image Processing\Project\git_hub\DIP_Project\line_imgs_paper\\"
    # D:\Digital Image Processing\Project\git_hub\DIP_Project\line_imgs_paper
    # image_name = "image_2_IAM"
    # input_path = 
    output__line_folder_path_i = os.path.join(output__line_folder_path, prefix)
    os.makedirs(output__line_folder_path_i, exist_ok=True)
    output__word_folder_path_i = os.path.join(output__word_folder_path, prefix)
    os.makedirs(output__word_folder_path_i, exist_ok=True)

    _,_,_,_,_, d_image= PPA_Algo_line(input_img_path,output__line_folder_path_i,prefix )

    result_folder_path_line = os.path.join(result_folder_path, 'line_results')
    os.makedirs(result_folder_path_line, exist_ok=True)
    cv2.imwrite(os.path.join(result_folder_path_line, f'{prefix}_lines.png'), d_image)

    # plt.imshow(d_image, cmap='gray')
    # plt.show()
    n_lines=0
    word_r = word_recog_3.HandwrittenTextExtractor()
    print("model loaded")
    result_folder_path_w = os.path.join(result_folder_path, 'word_seg_results')
    os.makedirs(result_folder_path_w, exist_ok=True)
    result_folder_path_word = os.path.join(result_folder_path_w, prefix)
    os.makedirs(result_folder_path_word, exist_ok=True)
    

    
    result_folder_path_w_r = os.path.join(result_folder_path, 'word_reco_results')
    os.makedirs(result_folder_path_w_r, exist_ok=True)
    result_folder_path_word_recog = os.path.join(result_folder_path_w_r, prefix)
    os.makedirs(result_folder_path_word_recog, exist_ok=True)
    
    cursived_status = is_cursived(output__line_folder_path, prefix)

    for filename in os.listdir(output__line_folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            if prefix and not filename.startswith(prefix):
                continue  # Skip files that don't match the prefix
            line_img_path = os.path.join(output__line_folder_path, filename)

            line_image_name = f"{n_lines}"

            d_image = word_segmentation(line_img_path,cursived_status,n_lines, output__word_folder_path_i, line_image_name)

            cv2.imwrite(os.path.join(result_folder_path_word, f'{line_image_name}.png'), d_image)
            
            n_lines+=1

            # # line_img_path = r"D:\Digital Image Processing\Project\git_hub\DIP_Project\line_imgs_paper\\"+ "image_2_IAM_0.png"
            # # output_folder = r"D:\Digital Image Processing\Project\git_hub\DIP_Project\line_imgs_paper_words\\"
            # # D:\Digital Image Processing\Project\git_hub\DIP_Project\line_imgs_paper
            # # image_name = "image_2_IAM_0"
            # # _,_,_,_,_,_, d_image= PPA_Algo_word(line_img_path,output__word_folder_path,line_image_name )
            # # plt.imshow(d_image, cmap='gray')
                
            # # output_folder = r"D:\Digital Image Processing\Project\git_hub\DIP_Project\line_imgs_paper_words\\"
            # results = word_r.predict_text(output__word_folder_path, line_image_name)
            # print(results)
            # fig, axes = plt.subplots(1,len(results), figsize=(5*len(results), 5 ))
            # # output__word_folder_path = r"D:\Digital Image Processing\Project\git_hub\DIP_Project\line_imgs_paper_words\\"

            # for i, ax in enumerate(axes):
            #     image_path = os.path.join(output__word_folder_path, results[i][0])
            #     image = Image.open(image_path)
                
            #     ax.imshow(image, cmap='gray')
            #     ax.set_title(f"P: {results[i][1]}", fontsize=20)
            #     ax.axis('off')  # Hide axes

            # plt.tight_layout()
            # plt.show()

            results = word_r.predict_text(output__word_folder_path_i, line_image_name)
            results_sorted = sorted(results, key=lambda x: int(x[0].split('_')[-1].split('.')[0]))

            # Display the word segmentation results
            fig, axes = plt.subplots(1, len(results_sorted), figsize=(5 * len(results_sorted), 5))
            axes = axes.ravel()
            for i, ax in enumerate(axes):
                image_path = os.path.join(output__word_folder_path_i, results_sorted[i][0])
                image = Image.open(image_path)
                ax.imshow(image, cmap='gray')
                ax.set_title(f"P: {results_sorted[i][1]}", fontsize=40)
                ax.axis('off')

            plt.tight_layout()
            plot_output_path = os.path.join(result_folder_path_word_recog, f'{line_image_name}_prediction.png')
            plt.savefig(plot_output_path)
            plt.show()


# ! python "D:\Digital Image Processing\Project\git_hub\DIP_Project\line_word_recognition_py_files\main.py" "image_2_IAM.jpg" "image_2_IAM" "D:\Digital Image Processing\Project\git_hub\DIP_Project\line_imgs_paper\\" "D:\Digital Image Processing\Project\git_hub\DIP_Project\line_imgs_paper_words\\"



    
