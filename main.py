import logging

import pandas as pd
from paddleocr import PaddleOCR
from tqdm import tqdm

import constants as const
from utils import process_image, read_images


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
