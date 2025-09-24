import ocrmypdf
import os

def ocr_book_pdf(input_pdf: str, output_pdf: str, sidecar_txt: str | None = None):
    """
    OCR a (possibly scanned) PDF in Spanish.
    - Disables post-OCR optimization to avoid 'invalid pdf' from bad JPEG streams.
    - Writes a sidecar .txt with the recognized text.
    """
    ocrmypdf.ocr(
        input_pdf,
        output_pdf,
        language="spa",          # Spanish OCR
        skip_text=True,          # don't re-OCR pages that already have text
        deskew=True,             # straighten pages
        # clean=True,              # light cleanup
        optimize=0,              # <-- important: skip image re-encoding
        output_type="pdf",       # don’t force PDF/A; keep it simple
        sidecar=sidecar_txt
    )
    print(f"OCR done: {output_pdf}")
    if sidecar_txt:
        print(f"Text output to: {sidecar_txt}")


if __name__ == "__main__":
    inp = r".\docs"
    #Iterate over the folder, avoiding pdf with the suffix _searchable
    for file in os.listdir(inp):
        if file.endswith(".pdf") and not file.endswith("_searchable.pdf"):
            inp = os.path.join(inp, file)
            out = os.path.join(inp, file.replace(".pdf", "_searchable.pdf"))
            txt = os.path.join(inp, file.replace(".pdf", "_text.txt"))
            ocr_book_pdf(inp, out, sidecar_txt=txt)
    
    