import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tokenizer import Tokenizer
import re

def strip_markdown(text):
    """Strip markdown formatting more robustly."""
    # Remove headers
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    # Remove bold/italic
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    # Remove links [text](url) -> text
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    # Remove images ![]()
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    # Remove horizontal rules
    text = re.sub(r'^---+\s*$', '', text, flags=re.MULTILINE)
    # Remove blockquotes
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
    # Keep code blocks but remove fences if desired, for now keeping them
    return text

def extract_text_from_epub(epub_path):
    """Extract clean text from an EPUB file."""
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup
    import warnings
    
    # Ignore ebooklib warnings about non-standard epubs
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    try:
        book = epub.read_epub(epub_path)
        chapters = []
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'lxml')
                # Remove script and style elements
                for script_or_style in soup(['script', 'style']):
                    script_or_style.decompose()
                # Get text, using a space to join elements to avoid merging words
                text = soup.get_text(separator=' ')
                # Clean up whitespace
                text = re.sub(r'\s+', ' ', text).strip()
                if text:
                    chapters.append(text)
        return "\n\n".join(chapters)
    except Exception as e:
        print(f"Error reading {epub_path}: {e}")
        return ""

def extract_text_from_pdf(pdf_path):
    """Extract clean text from a PDF file using PyMuPDF."""
    import fitz  # pymupdf
    
    try:
        doc = fitz.open(pdf_path)
        pages = []
        for page in doc:
            text = page.get_text("text")
            if text.strip():
                pages.append(text.strip())
        doc.close()
        
        full_text = "\n\n".join(pages)
        # Clean up common PDF artifacts
        full_text = re.sub(r'\n{3,}', '\n\n', full_text)  # Collapse excessive newlines
        full_text = re.sub(r'[ \t]+', ' ', full_text)      # Collapse spaces/tabs
        return full_text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

def prepare_data(data_dir, tokenizer_model="tokenizer.model", out_file="data.bin"):
    print(f"Preparing data from {data_dir}...")
    tokenizer = Tokenizer(tokenizer_model)
    
    all_tokens = []
    # Search for .md, .epub, and .pdf files
    files = []
    for ext in ["*.md", "*.epub", "*.pdf"]:
        files.extend(list(Path(data_dir).rglob(ext)))
    files = sorted(files)
    
    for file_path in files:
        if file_path.suffix.lower() == ".md":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                text = strip_markdown(text)
        elif file_path.suffix.lower() == ".epub":
            text = extract_text_from_epub(str(file_path))
        elif file_path.suffix.lower() == ".pdf":
            text = extract_text_from_pdf(str(file_path))
        else:
            continue
            
        if not text:
            continue
            
        # Encode with EOS token between documents
        tokens = tokenizer.encode(text, eos=True)
        all_tokens.extend(tokens)
            
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    
    # Save to directory of data_dir
    data_path = Path(data_dir)
    train_out = data_path / "data_train.bin"
    val_out = data_path / "data_val.bin"
    full_out = data_path / "data.bin"

    # Save to memmap
    memmap = np.memmap(full_out, dtype=np.uint16, mode='w+', shape=all_tokens.shape)
    memmap[:] = all_tokens[:]
    memmap.flush()
    print(f"Saved {len(all_tokens)} tokens to {full_out}")
    
    # Split for train/val
    split_idx = int(len(all_tokens) * 0.9)
    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:]
    
    train_memmap = np.memmap(train_out, dtype=np.uint16, mode='w+', shape=train_tokens.shape)
    train_memmap[:] = train_tokens[:]
    train_memmap.flush()
    
    val_memmap = np.memmap(val_out, dtype=np.uint16, mode='w+', shape=val_tokens.shape)
    val_memmap[:] = val_tokens[:]
    val_memmap.flush()
    
    return str(train_out), str(val_out)

class MemmapDataset(Dataset):
    def __init__(self, memmap_file, ctx_len, pad_id=3):
        self.data = np.memmap(memmap_file, dtype=np.uint16, mode='r')
        self.ctx_len = ctx_len
        self.pad_id = pad_id
        
        # Non-overlapping chunks: stride = ctx_len
        self.num_chunks = len(self.data) // self.ctx_len

    def __len__(self):
        return max(1, self.num_chunks)

    def __getitem__(self, idx):
        # Start index based on non-overlapping chunks
        start_idx = idx * self.ctx_len
        
        # We need ctx_len + 1 tokens for x and y
        chunk_data = self.data[start_idx : start_idx + self.ctx_len + 1].astype(np.int64)
        
        # Handle end of file where we might not have a full chunk + 1
        if len(chunk_data) < self.ctx_len + 1:
            full_chunk = np.zeros(self.ctx_len + 1, dtype=np.int64) + self.pad_id
            full_chunk[:len(chunk_data)] = chunk_data
            chunk_data = full_chunk
            
        x = torch.tensor(chunk_data[:-1], dtype=torch.long)
        y = torch.tensor(chunk_data[1:], dtype=torch.long)
        
        # Mask padding in loss
        y[y == self.pad_id] = -100
        
        return x, y

def get_dataloader(memmap_file, batch_size, ctx_len, pad_id=3, num_workers=4):
    dataset = MemmapDataset(memmap_file, ctx_len, pad_id)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False # Allow small datasets to return at least one batch
    )

if __name__ == "__main__":
    prepare_data("training_data/")

if __name__ == "__main__":
    prepare_data("training_data/")
