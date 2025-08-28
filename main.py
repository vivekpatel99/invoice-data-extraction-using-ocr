from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tqdm
from paddleocr import PaddleOCR

import constants as const


def read_images(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Path {path} does not exist")
    yield from path.iterdir()


def crop_upper_right(img) -> np.ndarray:
    # get image height and width
    height, width = img.shape[:2]
    print(f"Original image size: {width}x{height}")

    crop_height = int(height * 0.70)
    crop_width = int(width * 0.50) + 10
    x1, y1 = width - crop_width, int(height * 0.10)  # top

    x2, y2 = width, height - crop_height
    return img[y1:y2, x1:x2]  # [rows, columns]


def main() -> None:
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
        # read image
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # crop image and grab only 50% of image (rest is not needed)
        cropped_img = crop_upper_right(img)

        # give the image to OCR and get the text
        result = ocr.predict(cropped_img)
        required_info = result[0]["rec_texts"]

        client_name = required_info[1]
        client_address = required_info[2:-1]
        tax_id = required_info[-1][6:]

        data_dict["client_name"].append(client_name)
        data_dict["client_address"].append(" ".join(client_address))
        data_dict["tax_id"].append(tax_id)

    # save text to the excel file
    dataframe = pd.DataFrame(data_dict)
    dataframe.to_excel(str(const.OUTPUT_PATH), index=False)


if __name__ == "__main__":
    main()
