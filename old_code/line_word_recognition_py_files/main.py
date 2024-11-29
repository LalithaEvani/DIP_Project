import matplotlib.pyplot as plt
import argparse
import sys
sys.path.append("line_word_recognition_py_files")
from line_seg_1 import PPA_Algo_line
from word_seg_2 import PPA_Algo_word
import word_recog_3
import os
from PIL import Image

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("arg1",type=str, help="input image path")
    parser.add_argument("arg2",type=str, help="input image name")
    parser.add_argument("arg3",type=str, help="output folder to save the lines")
    parser.add_argument("arg4",type=str, help="output folder to save the words")
    args=parser.parse_args()

    input_img_path = args.arg1
    image_name = args.arg2
    output__line_folder_path = args.arg3
    output__word_folder_path = args.arg4

    _,_,_,_,_, d_image= PPA_Algo_line(input_img_path,output__line_folder_path,image_name )
    plt.imshow(d_image, cmap='gray')
    plt.show()
    prefix = image_name
    n_lines=0
    word_r = word_recog_3.HandwrittenTextExtractor()
    line_imgs = []
    for filename in os.listdir(output__line_folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            if prefix and filename.startswith(prefix):
                line_imgs.append(filename)
    line_imgs_n = sorted(line_imgs, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    print("line_imgs_n",line_imgs)
    print("line_imgs_n",line_imgs_n)
    for filename in line_imgs_n:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            if prefix and not filename.startswith(prefix):
                continue  # Skip files that don't match the prefix
            line_img_path = os.path.join(output__line_folder_path, filename)

            line_image_name = prefix + f"_{n_lines}"
            n_lines+=1

            # line_img_path = r"D:\Digital Image Processing\Project\git_hub\DIP_Project\line_imgs_paper\\"+ "image_2_IAM_0.png"
            # output_folder = r"D:\Digital Image Processing\Project\git_hub\DIP_Project\line_imgs_paper_words\\"
            # D:\Digital Image Processing\Project\git_hub\DIP_Project\line_imgs_paper
            # image_name = "image_2_IAM_0"
            _,_,_,_,_,_, d_image= PPA_Algo_word(line_img_path,output__word_folder_path,line_image_name )
            # plt.imshow(d_image, cmap='gray')
                
            # output_folder = r"D:\Digital Image Processing\Project\git_hub\DIP_Project\line_imgs_paper_words\\"
            results = word_r.predict_text(output__word_folder_path, line_image_name)
            results = sorted(results, key=lambda x: int(x[0].split('_')[-1].split('.')[0]))
            print(results)
            fig, axes = plt.subplots(1,len(results), figsize=(5*len(results), 5 ))
            # output__word_folder_path = r"D:\Digital Image Processing\Project\git_hub\DIP_Project\line_imgs_paper_words\\"
            axes = axes.ravel()
            for i, ax in enumerate(axes):
                image_path = os.path.join(output__word_folder_path, results[i][0])
                image = Image.open(image_path)
                
                ax.imshow(image, cmap='gray')
                ax.set_title(f"P: {results[i][1]}", fontsize=20)
                ax.axis('off')  # Hide axes

            plt.tight_layout()
            plt.savefig(f'output_plots/{line_image_name}_final_plot.png')
            plt.show()



# ! python "D:\Digital Image Processing\Project\git_hub\DIP_Project\line_word_recognition_py_files\main.py" "image_2_IAM.jpg" "image_2_IAM" "D:\Digital Image Processing\Project\git_hub\DIP_Project\line_imgs_paper\\" "D:\Digital Image Processing\Project\git_hub\DIP_Project\line_imgs_paper_words\\"



    
