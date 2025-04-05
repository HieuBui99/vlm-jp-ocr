import os
from pathlib import Path
import numpy as np
import cv2
from pdf2image import convert_from_path
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBoxHorizontal, LTChar
from tqdm.contrib.concurrent import process_map

SOURCE = "data/PDF"
DESTINATION_IMAGES = "data/images"
DESTINATION_LABELS = "data/labels"


def pdf_to_images_text_labels(file):
    pdf_path = Path(os.path.join(SOURCE, file))

    # Convert PDF to image
    images = convert_from_path(pdf_path, dpi=150)

    # PDFminer analyze PDF
    pages = extract_pages(pdf_path)

    counter = 0
    try:
        for page, image in zip(pages, images):
            cv_image = np.array(image)

            image_height, image_width, _ = cv_image.shape
            page_width = page.width
            page_height = page.height

            labels_str = ""
            text_boxes = []
            for element in page:
                if isinstance(element, LTTextBoxHorizontal):
                    for e2 in element:
                        for e3 in e2:
                            if isinstance(e3, LTChar):
                                if e3.get_text().strip() != "":
                                    x0 = int((e3.x0 / page_width) * image_width)
                                    x1 = int((e3.x1 / page_width) * image_width)
                                    y0 = int((1 - (e3.y0 / page_height)) * image_height)
                                    y1 = int((1 - (e3.y1 / page_height)) * image_height)

                                    x0 = max(0, x0 - 5)
                                    x1 = min(x1 + 2, image_width)
                                    y0, y1 = max(0, y1) - 4, min(y0, image_height) + 4
                                    # Draw rectangle around character

                                    text_boxes.append((x0, y0, x1, y1, e3.get_text()))

                                    labels_str += (
                                        f"{x0} {y0} {x1} {y1} {e3.get_text()}\n"
                                    )
            # Merge text boxes into vertical lines
            x_groups = {}
            for box_idx, (x0, y0, x1, y1, text) in enumerate(text_boxes):
                x_center = (x0 + x1) // 2

                # Find if this box fits in an existing group
                assigned = False
                for center_x in list(x_groups.keys()):
                    if abs(x_center - center_x) < 20:  # Threshold for vertical line
                        x_groups[center_x].append((x0, y0, x1, y1))
                        assigned = True
                        break

                # Create new group if needed
                if not assigned:
                    x_groups[x_center] = [(x0, y0, x1, y1)]

            # Draw bounding boxes for vertical lines
            # for line_boxes in x_groups.values():
            #     # Sort boxes vertically within the line (top to bottom)
            #     sorted_boxes = sorted(line_boxes, key=lambda box: box[1])

            #     if sorted_boxes:
            #         # Get overall bounding box for the line
            #         min_x = min(box[0] for box in sorted_boxes)
            #         min_y = min(box[1] for box in sorted_boxes)
            #         max_x = max(box[2] for box in sorted_boxes)
            #         max_y = max(box[3] for box in sorted_boxes)

            # Draw rectangle for the vertical line

            # Create directories if they don't exist
            os.makedirs(DESTINATION_IMAGES, exist_ok=True)
            os.makedirs(DESTINATION_LABELS, exist_ok=True)

            # Generate base filename for this page
            base_filename = f"{pdf_path.stem}_page{counter}"

            # Process each vertical line
            for line_idx, center_x in enumerate(x_groups.keys()):
                line_boxes = x_groups[center_x]

                # Sort boxes vertically within the line (top to bottom)
                sorted_boxes = sorted(line_boxes, key=lambda box: box[1])

                if sorted_boxes:
                    # Get overall bounding box for the line
                    min_x = max(0, min(box[0] for box in sorted_boxes))
                    min_y = max(0, min(box[1] for box in sorted_boxes))
                    max_x = min(image_width, max(box[2] for box in sorted_boxes))
                    max_y = min(image_height, max(box[3] for box in sorted_boxes))

                    # Crop the image for this line
                    line_image = cv_image[min_y:max_y, min_x:max_x]

                    # Skip empty lines
                    if line_image.size == 0:
                        continue

                    # Extract text for this line
                    line_text = ""
                    for x0, y0, x1, y1, text in text_boxes:
                        box_center_x = (x0 + x1) // 2
                        # Check if this text box belongs to current line
                        if abs(box_center_x - center_x) < 20 and min_y <= y0 <= max_y:
                            line_text += text

                    # Save image
                    line_filename = f"{base_filename}_line{line_idx}"
                    image_path = os.path.join(
                        DESTINATION_IMAGES, f"{line_filename}.png"
                    )
                    cv2.imwrite(image_path, line_image)

                    # Save text
                    label_path = os.path.join(
                        DESTINATION_LABELS, f"{line_filename}.txt"
                    )
                    with open(label_path, "w", encoding="utf-8") as f:
                        f.write(line_text)
            counter += 1
    except (Exception,) as e:
        print(e, pdf_path)


files = os.listdir(SOURCE)[:300]
# print(len(files))
process_map(pdf_to_images_text_labels, files, max_workers=10, chunksize=1)
