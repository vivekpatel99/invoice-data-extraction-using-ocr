import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from paddleocr import PaddleOCR
from tqdm import tqdm

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


def parse_ocr_results(text_lines: list[str]) -> dict[str, str]:
    """
    Parses the OCR text lines to extract structured client information.

    This is a heuristic-based parser and might need tuning for different
    invoice formats. It's more robust than fixed-index parsing.

    Args:
        text_lines (list[str]): A list of text strings from OCR.

    Returns:
        dict[str, str]: A dictionary with extracted client information.
    """
    client_name = ""
    client_address = ""
    tax_id = ""

    if not text_lines:
        return {}

    if len(text_lines) > 1:
        client_name = text_lines[1]
    if len(text_lines) > 2:
        # Join lines from index 2 up to the second-to-last for address
        client_address = "".join(text_lines[2:-1])
    if text_lines and len(text_lines) > 0:
        # Assuming tax_id is always at the end and starts after "Tax ID: "
        last_line = text_lines[-1]
        if "Tax ID:" in last_line:
            tax_id = last_line.split("Tax ID:")[1].strip()
        elif len(last_line) > 6:  # Fallback if "Tax ID:" isn't explicit
            tax_id = last_line[6:]  # Original logic

    return {
        "client_name": client_name,
        "client_address": client_address,
        "tax_id": tax_id,
    }


def process_image(image_path: Path, ocr: PaddleOCR) -> dict[str, str]:
    """Reads, crops, and extracts data from a single image."""
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        logging.warning("Could not read image %s. Skipping.", image_path)
        return {}

    cropped_img = crop_upper_right(img)
    result = ocr.predict(cropped_img)

    if result and result[0] and "rec_texts" in result[0]:
        text_lines = result[0]["rec_texts"]
        return parse_ocr_results(text_lines)

    logging.warning("No text detected or unexpected OCR result for %s.", image_path)
    return {}


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

    image_paths = list(read_images(const.INVOICE_PATH))
    if not image_paths:
        logging.warning("No images found in %s. Exiting.", const.INVOICE_PATH)
        return

    extracted_data = []
    for image_path in tqdm(image_paths, desc="Processing Invoices"):
        try:
            data = process_image(image_path, ocr)
            if data:
                data["source_file"] = image_path.name
                extracted_data.append(data)
        except Exception:
            logging.exception("Error processing %s. Skipping.", image_path)

    if not extracted_data:
        logging.warning(
            "Could not extract data from any image. No output file will be created."
        )
        return

    # Save text to the excel file
    dataframe = pd.DataFrame(extracted_data)

    # Reorder columns for clarity
    cols = ["source_file", "client_name", "client_address", "tax_id"]
    # Filter for existing columns to avoid errors if a field is never found
    existing_cols = [col for col in cols if col in dataframe.columns]
    dataframe = dataframe[existing_cols]

    # Ensure output directory exists
    const.OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_excel(const.OUTPUT_PATH, index=False)
    logging.info("\nSuccessfully processed %d images.", len(extracted_data))
    logging.info("Results saved to %s", const.OUTPUT_PATH)


if __name__ == "__main__":
    main()
