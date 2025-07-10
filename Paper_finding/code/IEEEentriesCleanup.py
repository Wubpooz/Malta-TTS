import re

output_all_path = "icassp2024/icassp2024.txt"

with open(output_all_path, "r", encoding="utf-8") as f:
    merged_content = f.read()
raw_entries = merged_content.strip().split("\n\n")

# Remove boilerplate entries like "Author Index", "Front Matter", etc.
boring_titles = {
    "Author Index", "Front Matter", "Copyright Page", "Table of Contents",
    "ICASSP 2025 Cover Page"
}

# Assumes format like: Author(s), "Title," Conference info, etc.
processed_entries = []
for entry in raw_entries:
    title_match = re.search(r'"([^"]+)"', entry)
    keyword_match = re.search(r'keywords:\s*\{([^}]*)\}', entry, re.IGNORECASE)

    if title_match and keyword_match:
        title = title_match.group(1).strip()
        keywords = keyword_match.group(1).strip()
        if title not in boring_titles:
            processed_entries.append(f"{title} - {{{keywords}}}")
    elif title_match:
        title = title_match.group(1).strip()
        if title not in boring_titles:
            processed_entries.append(title)

# # For v1
# titles = re.findall(r'"([^"]+)"', merged_content)
# cleaned_titles = sorted(set(title for title in titles if title not in boring_titles))
# title_output_path = "/mnt/data/unique_cleaned_titles.txt"
# with open(title_output_path, "w", encoding="utf-8") as f:
#     f.write("\n".join(cleaned_titles))

# title_output_path

final_entries = sorted(set(processed_entries))

final_output_path = "icassp2024/cleaned_titles_with_keywords.txt"
with open(final_output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(final_entries))

final_output_path
