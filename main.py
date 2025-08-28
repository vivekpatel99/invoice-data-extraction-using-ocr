from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tqdm
from paddleocr import PaddleOCR

import constants as const


def read_images(path: Path):
    """
    Reads image files from the specified directory.

    Args:
        path (Path): The directory containing the image files.

    Yields:
        Path: The path to each image file.

    Raises:
        FileNotFoundError: If the specified path does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Path {path} does not exist")
    yield from path.iterdir()


def crop_upper_right(img) -> np.ndarray:
    """
    Crops the upper-right section of an image.

    This function assumes that relevant client information is located
    in the upper-right quadrant of the invoice.

    Args:
        img (np.ndarray): The input image as a NumPy array.

    Returns:
        np.ndarray: The cropped image.
    """
    height, width = img.shape[:2]

    # Define cropping percentages (can be made configurable)
    # These values might need adjustment based on typical invoice layouts
    crop_height_percent = 0.70
    crop_width_percent = 0.50
    y_offset_percent = 0.10

    # Calculate crop dimensions
    crop_height = int(height * crop_height_percent)
    crop_width = int(width * crop_width_percent)
    y1 = int(height * y_offset_percent)

    # Calculate coordinates for cropping
    x1 = width - crop_width
    x2 = width
    y2 = height - crop_height

    # Ensure coordinates are valid
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    return img[y1:y2, x1:x2]


def main() -> None:
    """
    Main function to process invoice images, extract data using OCR,
    and save the results to an Excel file.
    """
    # Initialize PaddleOCR and load English model
    ocr = PaddleOCR(
        text_recognition_model_name="PP-OCRv4_server_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,  # text detection + text recognition
        lang="en",
    )
    data_dict = defaultdict(list)
    for image_path in tqdm.tqdm(read_images(const.INVOICE_PATH)):
        client_name = ""
        client_address = ""
        tax_id = ""

        try:
            # Read image
            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if img is None:
                print(f"Warning: Could not read image {image_path}. Skipping.")
                continue

            # Crop image
            cropped_img = crop_upper_right(img)

            # Perform OCR
            result = ocr.predict(cropped_img)
            # PaddleOCR result format: list of dicts, each dict for a detected block
            # For simplicity, assuming the first block contains all relevant text lines
            if result and result[0] and "rec_texts" in result[0]:
                required_info = result[0]["rec_texts"]

                if len(required_info) > 1:
                    client_name = required_info[1]
                if len(required_info) > 2:
                    # Join lines from index 2 up to the second-to-last for address
                    client_address = "".join(required_info[2:-1])
                if required_info and len(required_info) > 0:
                    # Assuming tax_id is always at the end and starts after "Tax ID: "
                    last_line = required_info[-1]
                    if "Tax ID:" in last_line:
                        tax_id = last_line.split("Tax ID:")[1].strip()
                    elif len(last_line) > 6:  # Fallback if "Tax ID:" isn't explicit
                        tax_id = last_line[6:]  # Original logic
            else:
                print(
                    f"Warning: No text detected or unexpected OCR result for {image_path}. Skipping."
                )
        except Exception as e:
            print(f"Error processing {image_path}: {e}. Skipping.")

        data_dict["client_name"].append(client_name)
        data_dict["client_address"].append(client_address)
        data_dict["tax_id"].append(tax_id)

    # save text to the excel file
    dataframe = pd.DataFrame(data_dict)
    dataframe.to_excel(const.OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
