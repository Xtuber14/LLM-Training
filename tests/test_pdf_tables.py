import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from dataset import extract_text_from_pdf

def test_pdf_extraction(pdf_path):
    print(f"Testing extraction on: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)
    
    if not text:
        print("Failed to extract text.")
        return
        
    print(f"Extracted {len(text)} characters.")
    
    # Save full output to a file for inspection
    output_file = "extracted_content_test.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Full extracted content saved to: {output_file}")
    
    # Look for table markers
    if "### Extracted Tables:" in text:
        print("Success! Found extracted tables.")
        # Count how many tables were found
        count = text.count("### Extracted Tables:")
        print(f"Total table sections found: {count}")
        
        # Print a larger snippet of the FIRST table
        idx = text.find("### Extracted Tables:")
        print("\nSnippet of extracted tables (first 2000 characters):")
        print("-" * 40)
        print(text[idx:idx+2000] + "...")
        print("-" * 40)
    else:
        print("No tables found in the Markdown format.")
        # Print first 500 chars to see if text extraction worked
        print("\nSnippet of text:")
        print(text[:500] + "...")

if __name__ == "__main__":
    # Test on a likely candidate for tables
    base_dir = Path(__file__).parent.parent
    test_pdf = base_dir / "training_data/List_of_That_Time_I_Got_Reincarnated_as_a_Slime_volumes.pdf"
    
    if test_pdf.exists():
        test_pdf_extraction(str(test_pdf))
    else:
        # Fallback to any PDF
        pdfs = list((base_dir / "training_data").glob("*.pdf"))
        if pdfs:
            test_pdf_extraction(str(pdfs[0]))
        else:
            print(f"No PDF files found in {base_dir}/training_data/")
