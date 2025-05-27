import cv2
import pytesseract
from pytesseract import Output
import pandas as pd
from collections import defaultdict
import os

def rotate_image(image, angle):
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        # For arbitrary angles (non-90 multiples)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h))

# Path to the input video
video_path = 'test.mp4'

# Output CSV file
output_csv = 'output.csv'

# Initialize video capture
cap = cv2.VideoCapture(video_path)

# Initialize a list to store extracted data
extracted_data = []

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process every nth frame to reduce redundancy (adjust as needed)
    if frame_count % 10 == 0:
        rotated = rotate_image(frame, 270)  # or 180 / 270
        
        # Convert to grayscale
        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)

        # Perform OCR with layout information
        ocr_data = pytesseract.image_to_data(gray, output_type=Output.DICT)

        # Collect words with their positions
        words = []
        for i in range(len(ocr_data["text"])):
            text = ocr_data["text"][i].strip()
            try:
                conf = float(ocr_data["conf"][i])
            except ValueError:
                continue
            if text and conf > 60:
                x, y, w, h = (ocr_data["left"][i], ocr_data["top"][i],
                              ocr_data["width"][i], ocr_data["height"][i])
                words.append({
                    "text": text,
                    "conf": conf,
                    "left": x,
                    "top": y,
                    "right": x + w,
                    "bottom": y + h
                })

        # Group words by approximate vertical position (rows)
        row_buckets = defaultdict(list)
        row_tolerance = 10  # pixels

        for word in words:
            row_key = word['top'] // row_tolerance
            row_buckets[row_key].append(word)

        # Sort rows by vertical position
        sorted_rows = sorted(row_buckets.items(), key=lambda item: min(w['top'] for w in item[1]))

        # Extract leftmost string as name, rightmost number as score
        for _, row_words in sorted_rows:
            if len(row_words) < 2:
                continue
            # Sort row left-to-right
            sorted_row = sorted(row_words, key=lambda w: w['left'])
            name = sorted_row[0]['text']
            numbers = [w['text'] for w in sorted_row if w['text'].replace(',', '').replace('.', '').isdigit()]
            score = numbers[-1] if numbers else ''
            if name and score:
                extracted_data.append((name, score))
    break
    frame_count += 1

cap.release()

# Remove duplicates
unique_data = list(set(extracted_data))

# Save to CSV
df = pd.DataFrame(unique_data, columns=['Name', 'Total Score'])
df.to_csv(output_csv, index=False)

print(f"Extraction complete. Data saved to {output_csv}.")