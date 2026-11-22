import os
import cv2
import pytesseract

REPORTS_FOLDER = "reports"

files = [
    f for f in os.listdir(REPORTS_FOLDER)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"))
]

if not files:
    print("No image files found in reports folder.")
    raise SystemExit

image_path = os.path.join(REPORTS_FOLDER, files[0])
print("Using file:", image_path)

image = cv2.imread(image_path)

if image is None:
    print("Failed to read image. Check folder name and file.")
    raise SystemExit

text = pytesseract.image_to_string(image)
print("\nOCR TEXT:\n")
print(text)