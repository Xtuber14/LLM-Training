import sys
import os
from pathlib import Path
import pytest

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from dataset import extract_text_from_pdf

def _find_test_pdf():
    base_dir = Path(__file__).parent.parent
    preferred = base_dir / "training_data/List_of_That_Time_I_Got_Reincarnated_as_a_Slime_volumes.pdf"
    if preferred.exists():
        return preferred
    pdfs = sorted((base_dir / "training_data").glob("*.pdf"))
    return pdfs[0] if pdfs else None


def test_pdf_extraction(tmp_path):
    pdf_path = _find_test_pdf()
    if pdf_path is None:
        pytest.skip("No PDF files found in training_data/")

    text = extract_text_from_pdf(str(pdf_path))
    assert isinstance(text, str)
    assert text.strip(), f"Expected extracted text from {pdf_path}"

    output_file = tmp_path / "extracted_content_test.txt"
    output_file.write_text(text, encoding="utf-8")
    assert output_file.exists()

if __name__ == "__main__":
    # Test on a likely candidate for tables
    base_dir = Path(__file__).parent.parent
    test_pdf = base_dir / "training_data/List_of_That_Time_I_Got_Reincarnated_as_a_Slime_volumes.pdf"
    
    if test_pdf.exists():
        print(extract_text_from_pdf(str(test_pdf))[:500])
    else:
        # Fallback to any PDF
        pdfs = list((base_dir / "training_data").glob("*.pdf"))
        if pdfs:
            print(extract_text_from_pdf(str(pdfs[0]))[:500])
        else:
            print(f"No PDF files found in {base_dir}/training_data/")
