import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tokenizer import Tokenizer
import re
import unicodedata
from html import unescape


def _recommended_num_workers():
    cpu_count = os.cpu_count() or 1
    return min(4, max(0, cpu_count - 1))

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


def normalize_text(text):
    """Normalize text and collapse noisy whitespace while preserving paragraphs."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\ufeff", "").replace("\u00a0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_txt(txt_path):
    """Extract text from .txt files with robust encoding fallback."""
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    for enc in encodings:
        try:
            with open(txt_path, "r", encoding=enc) as f:
                return normalize_text(f.read())
        except UnicodeDecodeError:
            continue
        except Exception:
            break

    # Last resort: best-effort decode
    with open(txt_path, "rb") as f:
        raw = f.read()
    return normalize_text(raw.decode("utf-8", errors="ignore"))


def _clean_subtitle_line(line):
    line = line.strip()
    if not line:
        return ""
    # Remove SSA/ASS style formatting codes and HTML-like tags
    line = re.sub(r"\{\\.*?\}", "", line)
    line = re.sub(r"<[^>]+>", "", line)
    line = line.replace("\\N", "\n").replace("\\n", "\n")
    return normalize_text(unescape(line))


def extract_text_from_subtitles(sub_path):
    """Extract text from subtitles: .srt, .vtt, .ass, .ssa."""
    ext = Path(sub_path).suffix.lower()
    raw = extract_text_from_txt(sub_path)
    if not raw:
        return ""

    lines = []
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        if ext in {".srt", ".vtt"}:
            if s.upper() == "WEBVTT":
                continue
            if re.fullmatch(r"\d+", s):
                continue
            # srt/vtt timestamps
            if "-->" in s:
                continue
        if ext in {".ass", ".ssa"}:
            # Skip non-dialogue sections/metadata
            if s.startswith(("[", ";", "Format:", "Style:")):
                continue
            if s.startswith("Dialogue:"):
                # Dialogue: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
                parts = s.split(",", 9)
                if len(parts) == 10:
                    s = parts[-1]
                else:
                    s = s[len("Dialogue:") :].strip()
            else:
                continue

        cleaned = _clean_subtitle_line(s)
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines)

def extract_text_from_epub(epub_path):
    """Extract clean text from an EPUB file in reading order when possible."""
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

        # Prefer spine order for better semantic continuity.
        spine_ids = [item_id for item_id, _ in getattr(book, "spine", []) if item_id and item_id != "nav"]
        id_to_item = {}
        for item in book.get_items():
            try:
                id_to_item[item.get_id()] = item
            except Exception:
                continue

        ordered_items = []
        for item_id in spine_ids:
            if item_id in id_to_item:
                ordered_items.append(id_to_item[item_id])

        # Fallback to all document items if spine is missing/broken.
        if not ordered_items:
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    ordered_items.append(item)

        for item in ordered_items:
            if item.get_type() != ebooklib.ITEM_DOCUMENT:
                continue
            content = item.get_content()
            soup = None
            for parser in ("lxml", "html.parser"):
                try:
                    soup = BeautifulSoup(content, parser)
                    break
                except Exception:
                    continue
            if soup is None:
                continue

            for script_or_style in soup(["script", "style", "noscript"]):
                script_or_style.decompose()
            text = soup.get_text(separator="\n")
            text = normalize_text(text)
            if text:
                chapters.append(text)

        return "\n\n".join(chapters)
    except Exception as e:
        print(f"Error reading {epub_path}: {e}")
        return ""

def table_to_markdown(table):
    """Convert a 2D list (table) to a Markdown table."""
    from tabulate import tabulate
    if not table or not any(table):
        return ""
    # Clean up cells: remove None and strip whitespace
    cleaned_table = []
    for row in table:
        cleaned_row = [str(cell).strip() if cell is not None else "" for cell in row]
        # Skip completely empty rows
        if any(cleaned_row):
            cleaned_table.append(cleaned_row)
    
    if not cleaned_table:
        return ""
        
    # Use the first row as header if it looks like one, or generic headers
    try:
        return tabulate(cleaned_table, tablefmt="github", headers="firstrow")
    except:
        return tabulate(cleaned_table, tablefmt="github")

def extract_text_from_pdf(pdf_path):
    """Extract clean text and tables from a PDF file using fitz (text) and pdfplumber (tables)."""
    import fitz
    import pdfplumber
    import logging
    from tqdm import tqdm
    
    # Suppress noisy warnings from both libraries
    logging.getLogger('pdfminer').setLevel(logging.ERROR)
    try:
        fitz.TOOLS.mupdf_display_errors(False)
    except:
        pass
    
    try:
        pages = []
        doc = fitz.open(pdf_path)
        pdf = pdfplumber.open(pdf_path)
        
        num_pages = len(doc)
        page_indices = range(num_pages)
        
        # Always show progress for PDFs with many pages to reassure the user
        desc = f"Reading {os.path.basename(pdf_path)}"
        page_indices = tqdm(range(num_pages), desc=desc, leave=False)
            
        for i in page_indices:
            # 1. Fast text extraction with fitz
            text = doc[i].get_text()
            
            # 2. Table extraction with pdfplumber (only if needed or for every page)
            markdown_tables = []
            try:
                # pdfplumber can be slow, so we only call it on the specific page
                p_page = pdf.pages[i]
                tables = p_page.extract_tables()
                for table in tables:
                    md = table_to_markdown(table)
                    if md:
                        markdown_tables.append(md)
            except:
                pass # Skip page if table extraction fails
            
            page_content = []
            if text:
                text = re.sub(r'[ \t]+', ' ', text.strip())
                page_content.append(text)
            if markdown_tables:
                page_content.append("\n\n<|start_table|>\n### Extracted Tables:\n" + "\n\n".join(markdown_tables) + "\n<|end_table|>\n")
            
            if page_content:
                pages.append("\n\n".join(page_content))
                
        doc.close()
        pdf.close()
                    
        if not pages:
            return ""
            
        full_text = "\n\n".join(pages)
        return normalize_text(full_text)
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""


def extract_text_from_file(file_path):
    suffix = Path(file_path).suffix.lower()
    if suffix == ".md":
        with open(file_path, "r", encoding="utf-8") as f:
            return normalize_text(strip_markdown(f.read()))
    if suffix == ".txt":
        return extract_text_from_txt(file_path)
    if suffix in {".srt", ".vtt", ".ass", ".ssa"}:
        return extract_text_from_subtitles(file_path)
    if suffix == ".epub":
        return extract_text_from_epub(file_path)
    if suffix == ".pdf":
        return extract_text_from_pdf(file_path)
    return ""

def prepare_data(data_dir, tokenizer_model="tokenizer.model", out_file="data.bin"):
    print(f"Preparing data from {data_dir}...")
    tokenizer = Tokenizer(tokenizer_model)
    
    token_chunks = []
    total_tokens = 0
    # Search for supported text-bearing files.
    files = []
    for ext in ["*.md", "*.txt", "*.srt", "*.vtt", "*.ass", "*.ssa", "*.epub", "*.pdf"]:
        files.extend(list(Path(data_dir).rglob(ext)))
    files = sorted(files)
    
    for file_path in files:
        text = extract_text_from_file(str(file_path))
            
        if not text:
            continue
            
        # Encode with EOS token between documents
        tokens = np.asarray(tokenizer.encode(text, eos=True), dtype=np.uint16)
        if tokens.size == 0:
            continue
        token_chunks.append(tokens)
        total_tokens += tokens.size

    if total_tokens == 0:
        raise ValueError(f"No tokenizable training data found in {data_dir}")
    
    # Save to directory of data_dir
    data_path = Path(data_dir)
    train_out = data_path / "data_train.bin"
    val_out = data_path / "data_val.bin"
    full_out = data_path / "data.bin"

    # Stream into memmap to avoid building one giant Python list in memory.
    memmap = np.memmap(full_out, dtype=np.uint16, mode='w+', shape=(total_tokens,))
    offset = 0
    for chunk in token_chunks:
        next_offset = offset + chunk.size
        memmap[offset:next_offset] = chunk
        offset = next_offset
    memmap.flush()
    print(f"Saved {total_tokens} tokens to {full_out}")
    
    # Split for train/val
    split_idx = int(total_tokens * 0.9)
    
    train_memmap = np.memmap(train_out, dtype=np.uint16, mode='w+', shape=(split_idx,))
    train_memmap[:] = memmap[:split_idx]
    train_memmap.flush()
    
    val_size = total_tokens - split_idx
    val_memmap = np.memmap(val_out, dtype=np.uint16, mode='w+', shape=(val_size,))
    val_memmap[:] = memmap[split_idx:]
    val_memmap.flush()
    
    return str(train_out), str(val_out)

class MemmapDataset(Dataset):
    def __init__(self, memmap_file, ctx_len, pad_id=3):
        self.data = np.memmap(memmap_file, dtype=np.uint16, mode='r')
        self.ctx_len = ctx_len
        self.pad_id = pad_id
        self.sample_span = self.ctx_len + 1
        if len(self.data) <= 1:
            self.num_chunks = 1
        else:
            self.num_chunks = max(1, math.ceil((len(self.data) - 1) / self.ctx_len))

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        # Random windows improve token/context diversity per unit time versus
        # fixed non-overlapping chunks, especially on smaller datasets.
        if len(self.data) > self.sample_span:
            max_start = len(self.data) - self.sample_span
            start_idx = int(torch.randint(0, max_start + 1, (1,)).item())
        else:
            start_idx = 0
        
        # We need ctx_len + 1 tokens for x and y
        chunk_data = np.asarray(self.data[start_idx : start_idx + self.sample_span], dtype=np.int64)
        
        # Handle end of file where we might not have a full chunk + 1
        if len(chunk_data) < self.ctx_len + 1:
            full_chunk = np.full(self.sample_span, self.pad_id, dtype=np.int64)
            full_chunk[:len(chunk_data)] = chunk_data
            chunk_data = full_chunk
            
        x = torch.from_numpy(chunk_data[:-1]).long()
        y = torch.from_numpy(chunk_data[1:]).long()
        
        # Mask padding in loss
        y[y == self.pad_id] = -100
        
        return x, y

def get_dataloader(memmap_file, batch_size, ctx_len, pad_id=3, num_workers=None, prefetch_factor=2):
    dataset = MemmapDataset(memmap_file, ctx_len, pad_id)
    if num_workers is None:
        num_workers = _recommended_num_workers()
    pin_memory = torch.cuda.is_available()
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        drop_last=False, # Allow small datasets to return at least one batch
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(
        **loader_kwargs
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="training_data/", help="Directory containing PDFs/EPUBs")
    args = parser.parse_args()
    
    prepare_data(args.data_dir)
