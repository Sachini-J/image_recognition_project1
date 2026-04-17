import csv
import os

image_dir = "images"
label_dir = "labels"
output_dir = "CSVs"
output_csv = os.path.join(output_dir, "dataset.csv")

os.makedirs(output_dir, exist_ok=True)

rows = []

for i in range(1, 130):  
    img_name = f"img_{i:03}.jpeg"
    txt_name = f"img_{i:03}.txt"

    img_path = os.path.join(image_dir, img_name)
    txt_path = os.path.join(label_dir, txt_name)

    if os.path.exists(img_path) and os.path.exists(txt_path):
        rows.append([img_path, txt_path])
    else:
        print("Missing:", img_path, "or", txt_path)

with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Images", "Labels"])
    writer.writerows(rows)


print("")
print(f" dataset.csv created with {len(rows)} rows")