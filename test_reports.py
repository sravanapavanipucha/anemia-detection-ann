import os
from process_image import extract_cbc_from_file

REPORTS_FOLDER = "reports"

def main():
    if not os.path.exists(REPORTS_FOLDER):
        print(f"Folder not found: {REPORTS_FOLDER}")
        return

    files = os.listdir(REPORTS_FOLDER)
    image_files = [
        f for f in files
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".pdf"))
    ]

    if not image_files:
        print("No report files found in reports folder.")
        return

    for filename in image_files:
        filepath = os.path.join(REPORTS_FOLDER, filename)

        print("\n" + "=" * 80)
        print(f"FILE: {filename}")
        print("=" * 80)

        result = extract_cbc_from_file(filepath)

        print("CBC Keywords Found:", result["keywords_found"])
        print("Extracted CBC Values:")
        for key, value in result["values"].items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()