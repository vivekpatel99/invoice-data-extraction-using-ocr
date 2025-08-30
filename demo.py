import json

import cv2
import numpy as np
from paddleocr import PaddleOCR

from utils import crop_upper_right


def draw_bboxes_on_original_from_json(original_img, json_result, offset_x, offset_y):
    """
    Draw bounding boxes on original image using JSON coordinates
    """
    img_with_boxes = original_img.copy()

    # Extract data from JSON
    rec_polys = json_result.get("rec_polys", [])
    rec_texts = json_result.get("rec_texts", [])
    rec_scores = json_result.get("rec_scores", [])

    # Draw each bounding box
    for i, poly in enumerate(rec_polys):
        # Adjust coordinates by adding crop offset
        adjusted_poly = []
        for point in poly:
            adjusted_point = [point[0] + offset_x, point[1] + offset_y]
            adjusted_poly.append(adjusted_point)

        # Convert to numpy array for OpenCV
        points = np.array(adjusted_poly, dtype=np.int32)

        # Draw bounding box
        cv2.polylines(img_with_boxes, [points], True, (0, 0, 255), 2)

        # Add text and confidence score
        if i < len(rec_texts) and i < len(rec_scores):
            text = rec_texts[i]
            score = rec_scores[i]

            # Position text on the right side of the bounding box, vertically centered.
            top_right = adjusted_poly[1]
            bottom_right = adjusted_poly[2]
            text_x = int(top_right[0]) + 5  # 5px margin from the right edge
            text_y = int((top_right[1] + bottom_right[1]) / 2)
            text_pos = (text_x, text_y)
            cv2.putText(
                img_with_boxes,
                f"{text} ({score:.2f})",
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

    return img_with_boxes


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

    # Load original image
    image_path = "datasets/batch_1/batch_1/batch1_1/batch1-0001.jpg"
    output_json_path = "output/demo.json"
    original_img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Crop image and get offset
    cropped_img, (offset_x, offset_y) = crop_upper_right(original_img, verbose=True)

    # Run OCR on cropped image
    result = ocr.predict(cropped_img)

    # Process results and save JSON
    # for i, res in enumerate(result):
    result = result[0]
    result.print()

    # Save JSON and get the saved JSON path
    result.save_to_json(output_json_path)

    # Load the JSON file to get coordinates
    with open(output_json_path) as f:
        json_data = json.load(f)

    # Draw bounding boxes on original image
    result_img = draw_bboxes_on_original_from_json(
        original_img, json_data, offset_x, offset_y
    )

    # Save result
    output_path = "output/original_with_bboxes_demo.jpg"
    cv2.imwrite(output_path, result_img)


if __name__ == "__main__":
    main()
