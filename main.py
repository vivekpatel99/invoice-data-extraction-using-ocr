from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR


def read_images(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Path {path} does not exist")
    yield from path.iterdir()


def crop_upper_right(img) -> np.ndarray:
    # get image height and width
    height, width = img.shape[:2]
    print(f"Original image size: {width}x{height}")
    # Crop coordinates
    # - Start from (0, 0) - top-left corner
    # - End at (25% of width, 25% of height)
    # grabbing only required section
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
    # for p in tqdm.tqdm(read_images(const.INVOICE_PATH)):
    #     print(p)
    image_path = "/home/ultron/freelance/01_active/upwork/radhika_patel/extract_client_info_OCR/03_development/datasets/batch_1/batch_1/batch1_1/batch1-0001.jpg"
    # read image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # crop image and grab only 50% of image (rest is not needed)
    cropped_img = crop_upper_right(img)

    # give the image to OCR and get the text
    result = ocr.predict(cropped_img)
    required_info = result[0]["rec_texts"]
    client_name = required_info[1]
    client_address = required_info[2:-1]
    taxt_id = required_info[-1]
    data_dict["client_name"].append(client_name)
    data_dict["client_address"].append(client_address)
    data_dict["taxt_id"].append(taxt_id)

    # save text to the excel file

    cv2.imshow("image", cropped_img)
    # cv2.imshow("full_image", img)

    # hold the screen until user close it.
    cv2.waitKey(0)

    # It is for removing/deleting created GUI window from screen
    # and memory
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
