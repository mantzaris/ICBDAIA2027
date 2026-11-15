"""Check anonymity, template geometry, evidence tables, and rendered page count.

Visual inspection of the rendered pages is a separate recorded review, not inferred
from these automated checks.
"""
import hashlib
import json
import re
import subprocess
from pathlib import Path
from zipfile import ZipFile

from docx import Document
from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manuscript"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    word_path = OUT / "underwater_feasibility_anonymous.docx"
    pdf_path = word_path.with_suffix(".pdf")
    document = Document(word_path)
    template = Document(OUT / "template/conference_template_macro_free.docx")
    assert document.core_properties.author == document.core_properties.last_modified_by == "Anonymous"
    for attribute in ("page_width", "page_height", "top_margin", "bottom_margin", "left_margin", "right_margin"):
        assert getattr(document.sections[0], attribute) == getattr(template.sections[0], attribute), attribute
    with ZipFile(word_path) as archive:
        assert "docProps/custom.xml" not in archive.namelist()
        assert not any("vba" in name.lower() or "comments" in name.lower() for name in archive.namelist())
        assert b"macroEnabled" not in archive.read("[Content_Types].xml")
        assert b"<Company>" not in archive.read("docProps/app.xml")
        assert b"GrammarlyDocumentId" not in b"".join(archive.read(name) for name in archive.namelist() if name.endswith(".xml"))
        external = []
        for name in archive.namelist():
            if name.endswith(".rels"):
                for relationship in etree.fromstring(archive.read(name)):
                    if relationship.get("TargetMode") == "External":
                        external.append(relationship.get("Target"))
        assert not external, external
    metadata = subprocess.check_output(["pdfinfo", str(pdf_path)], text=True)
    pages = int(re.search(r"^Pages:\s+(\d+)", metadata, re.MULTILINE).group(1))
    assert pages == 8, pages
    assert re.search(r"^Author:\s+Anonymous\s*$", metadata, re.MULTILINE)
    text = subprocess.check_output(["pdftotext", "-layout", str(pdf_path), "-"], text=True)
    assert "LAYOUT PREVIEW" not in text and "[PENDING" not in text and "\ufffd" not in text
    assert "Anonymous submission" in text and "References" in text
    assert "main study" in text.lower() and "untested" in text.lower()
    plan = json.loads((OUT / "render_plan.json").read_text())
    assert not plan["metadata"]["layout_preview"] and not plan["metadata"]["submission_ready"]
    expected_tables = [block for page in plan["pages"] for block in page if block["kind"] == "table"]
    assert len(document.tables) == len(expected_tables) == 3
    for table, expected in zip(document.tables, expected_tables):
        assert [[cell.text for cell in row.cells] for row in table.rows] == [expected["headers"], *expected["rows"]]
    report = {"status": "passed", "pages": pages, "author": "Anonymous",
              "template_geometry_preserved": True, "no_macros_or_external_links": True,
              "sample_custom_and_company_metadata_removed": True, "table_cells_match_evidence_render_plan": True,
              "abstract_words": plan["metadata"]["abstract_words"], "tables": 3,
              "reference_count": len(json.loads((OUT / "bibliography.json").read_text())),
              "docx_sha256": sha(word_path), "pdf_sha256": sha(pdf_path),
              "scope": "Development draft; automated checks do not establish scientific completeness or replace page-by-page visual inspection"}
    (OUT / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
