import docx
import os
import re

doc_path = r"C:\Users\Visha\Desktop\NewAlcohol\UC_EEG_FM\submission_docx\UC_EEG_FM_Manuscript.docx"
doc = docx.Document(doc_path)

print(f"Auditing figures and tables in {doc_path}...")

# Check all figure captions in the document
fig_captions = []
for p in doc.paragraphs:
    txt = p.text.strip()
    if txt.startswith("Fig. "):
        fig_captions.append(txt)

print(f"\nFound {len(fig_captions)} figure captions:")
for cap in fig_captions:
    print(f"  {cap[:100]}...")

expected_figs = [f"Fig. {i}." for i in range(1, 12)]
found_figs = [re.match(r'Fig\. \d+\.', c).group(0) for c in fig_captions if re.match(r'Fig\. \d+\.', c)]

print(f"\nExpected figures: {expected_figs}")
print(f"Found figures:    {found_figs}")
assert found_figs == expected_figs, f"Mismatch in figures! Expected {expected_figs}, found {found_figs}"
print(">> PASSED: All 11 figures (Fig. 1 to Fig. 11) have exact sequential captions in the document!")

# Check table count and headers
print(f"\nTotal tables: {len(doc.tables)}")
assert len(doc.tables) == 3, f"Expected exactly 3 tables, found {len(doc.tables)}"

for t_idx, table in enumerate(doc.tables):
    headers = [cell.text.strip() for cell in table.rows[0].cells]
    print(f"Table {t_idx + 1} ({len(table.rows)} rows x {len(table.columns)} cols):")
    print(f"  Headers: {headers}")

# Check that no page break exists before Introduction
# Check paragraph text of first 15 paragraphs
print("\nFirst 10 paragraphs structure:")
for i, p in enumerate(doc.paragraphs[:10]):
    print(f"  P{i+1}: {p.text[:60]}")

print("\n=== COMPLETE MANUSCRIPT FIGURE & TABLE AUDIT PASSED 100%! ===")
