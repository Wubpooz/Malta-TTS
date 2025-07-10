import glob
import os

folder_path = "icassp2025/"
txt_files = glob.glob(os.path.join(folder_path, "*.txt"))

all_entries = set()
for file_path in txt_files:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        entries = content.strip().split("\n\n")
        for entry in entries:
            normalized = entry.strip()
            if normalized:
                all_entries.add(normalized)

merged_content_all = "\n\n".join(sorted(all_entries))


output_all_path = os.path.join(folder_path, "icassp2025.txt")
with open(output_all_path, "w", encoding="utf-8") as f:
    f.write(merged_content_all)

output_all_path
