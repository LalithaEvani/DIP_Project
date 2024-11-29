import os
import warnings
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, logging

#edit: added few lines to ignore warnings
# Suppress all warnings
warnings.filterwarnings("ignore")

# Suppress logging from the transformers library
logging.set_verbosity_error()  # Only show errors and hide warnings/info/debug messages


class HandwrittenTextExtractor:
    def __init__(self, model_name="microsoft/trocr-base-handwritten", device=None):
        """
        Initializes the HandwrittenTextExtractor with a specified model and device.

        :param model_name: Hugging Face model name (default: "microsoft/trocr-base-handwritten")
        :param device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        self.processor = TrOCRProcessor.from_pretrained(model_name)

    def extract_text_from_image(self, image_path):
        """
        Extracts text from a given image.

        :param image_path: Path to the image file
        :return: Extracted text or an error message
        """
        try:
            image = Image.open(image_path).convert("RGB")  # Ensure RGB mode
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)

            # Generate text
            generated_ids = self.model.generate(pixel_values)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return generated_text
        except Exception as e:
            return f"Error processing {image_path}: {e}"

    def predict_text(self, image_folder, prefix=None):
        """
        Processes all images in a folder and extracts text.

        :param image_folder: Folder containing images
        :return: List of tuples [(image_name, extracted_text), ...]
        """
        results = []
        for filename in os.listdir(image_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                if prefix and not filename.startswith(prefix):
                    continue  # Skip files that don't match the prefix
                image_path = os.path.join(image_folder, filename)
                text = self.extract_text_from_image(image_path)
                results.append((filename, text))
        return results
