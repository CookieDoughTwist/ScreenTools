import cv2
import pytesseract
import pandas as pd
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

# Path to tesseract executable (set this if not in PATH)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# === CONFIG ===
video_path = 'test.mp4'
output_csv = 'output.csv'
frame_skip = 10  # Process every 10th frame to reduce redundancy

# === STEP 1: Read Video and Extract Frames ===
cap = cv2.VideoCapture(video_path)
frames = []
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count % frame_skip == 0:
        frames.append(frame)
    frame_count += 1

cap.release()

# === STEP 2: OCR on Frames ===
all_rows = []

for idx, frame in enumerate(frames):
    rotated = rotate_image(frame, 270)  # or 180 / 270
    
    # Convert to grayscale
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    
    # Optional: Improve contrast or thresholding
    # gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
    #                               cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite(f"frame_{idx}.png", gray)
    # OCR
    text = pytesseract.image_to_string(gray, config='--psm 6')  # PSM 6 = assume uniform block of text
    
    # Parse lines
    for line in text.split('\n'):
        if line.strip():  # Non-empty
            # Split by whitespace or tab (adjust as needed)
            row = [col.strip() for col in line.split()]
            all_rows.append(row)

# === STEP 3: Normalize and Save to CSV ===
# Find max number of columns
max_cols = max(len(row) for row in all_rows)

# Pad rows to same length
normalized_rows = [row + ['']*(max_cols - len(row)) for row in all_rows]

# Save to CSV
df = pd.DataFrame(normalized_rows)
df.to_csv(output_csv, index=False, header=False)

print(f"Saved extracted table to: {output_csv}")