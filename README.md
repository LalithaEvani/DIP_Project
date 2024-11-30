# DIP Project: Handwritten Line and Word Recognition

This project implements a pipeline for handwritten text recognition, consisting of line segmentation, word segmentation, and word recognition. The system is built using Python and applied to cropped images from the IAM Handwritten Forms dataset.

---

## Dataset
The project uses 32 cropped images from the [IAM Handwritten Forms Dataset](https://www.kaggle.com/datasets/naderabdalghani/iam-handwritten-forms-dataset).

---

## Project Structure
The repository contains the following files and folders:

- **`main_2.ipynb`**: Jupyter notebook for step-by-step visualization of the implemented algorithm.  
- **`word_recognizer.ipynb`**: Jupyter notebook for implementing the word recognition module.
- **`line_word_recognition_py_files/`**: Folder containing Python scripts for the three major steps of the pipeline:
  - `line_seg_1.py`: Script for line segmentation.
  - `word_seg_2_n.py`: Script for word segmentation.
  - `check_img_type.py`: Script for checking image type (cursive text and non cursive text).
  - `word_recog_3.py`: Script for word recognition.
  - `main.py`: The main script that integrates all components.
  
---

## Environment Setup
To run this project, set up a Conda environment with the necessary dependencies.

### Step 1: Create a Conda Environment
Run the following commands in your terminal:
```bash
conda create -n dip_project python=3.9 -y
conda activate dip_project
conda install -c conda-forge opencv pillow numpy matplotlib transformers -y
