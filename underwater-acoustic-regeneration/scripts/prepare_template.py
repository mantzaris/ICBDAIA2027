"""Make an editable macro-free copy of the exact conference template.

This manipulates ZIP/XML only. It does not execute VBA or change the original.
"""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import hashlib
import json
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "provenance/downloads/conference_template.docm"
    output = ROOT / "manuscript/template/conference_template_macro_free.docx"
    output.parent.mkdir(parents=True, exist_ok=True)
    removed = []
    removed_external_links = []
    with ZipFile(source) as original, ZipFile(output, "w", ZIP_DEFLATED) as clean:
        excluded = {name for name in original.namelist() if "vba" in name.lower() or name.startswith("customUI/")}
        for entry in original.infolist():
            if entry.filename in excluded:
                removed.append(entry.filename)
                continue
            payload = original.read(entry.filename)
            if entry.filename == "[Content_Types].xml":
                tree = ET.fromstring(payload)
                for child in list(tree):
                    content_type = child.get("ContentType", "")
                    if "vba" in content_type.lower() or child.get("PartName", "").lstrip("/") in excluded:
                        tree.remove(child)
                    elif content_type == "application/vnd.ms-word.document.macroEnabled.main+xml":
                        child.set("ContentType", "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml")
                payload = ET.tostring(tree, encoding="utf-8", xml_declaration=True)
            elif entry.filename.endswith(".rels"):
                tree = ET.fromstring(payload)
                for child in list(tree):
                    if any(term in (child.get("Type", "") + child.get("Target", "")).lower() for term in ("vba", "customui")):
                        tree.remove(child)
                    elif child.get("TargetMode") == "External" and "oleObject" in child.get("Type", ""):
                        removed_external_links.append({"part": entry.filename, "type": "external sample chart workbook"})
                        tree.remove(child)
                payload = ET.tostring(tree, encoding="utf-8", xml_declaration=True)
            elif entry.filename.startswith("word/charts/") and entry.filename.endswith(".xml"):
                tree = ET.fromstring(payload)
                for parent in tree.iter():
                    for child in list(parent):
                        if child.tag.endswith("}externalData"):
                            parent.remove(child)
                payload = ET.tostring(tree, encoding="utf-8", xml_declaration=True)
            clean.writestr(entry.filename, payload)
    with ZipFile(output) as check:
        assert not any("vba" in name.lower() for name in check.namelist())
        assert b"macroEnabled" not in check.read("[Content_Types].xml")
    report = {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "removed_parts": removed,
              "removed_external_links": removed_external_links,
              "macros_executed": False, "status": "template preparation only; not a manuscript"}
    (output.parent / "conversion.json").write_text(json.dumps(report, indent=2) + "\n")
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
