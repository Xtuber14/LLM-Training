import sentencepiece as spm
import os
import argparse
from pathlib import Path

class Tokenizer:
    def __init__(self, model_path="tokenizer.model"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Tokenizer model not found at {model_path}. "
                "Please run `python tokenizer.py --data_dir <dir>` first."
            )
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
            
    def encode(self, text, bos=False, eos=False):
        ids = self.sp.encode_as_ids(text)
        if bos:
            ids = [self.bos_id] + ids
        if eos:
            ids = ids + [self.eos_id]
        return ids
        
    def decode(self, ids):
        return self.sp.decode_ids(ids)
        
    @property
    def vocab_size(self):
        return self.sp.get_piece_size()
        
    @property
    def pad_id(self):
        return self.sp.pad_id()

    @property
    def bos_id(self):
        return self.sp.bos_id()

    @property
    def eos_id(self):
        return self.sp.eos_id()

def train_tokenizer(data_dir, vocab_size=32000, model_prefix="tokenizer"):
    print(f"Training tokenizer on {data_dir} with vocab size {vocab_size}")
    
    # We need to create a temporary file with all text for sentencepiece
    temp_text_file = "all_text.txt"
    files = []
    for ext in ["*.md", "*.epub", "*.pdf"]:
        files.extend(list(Path(data_dir).rglob(ext)))
    
    with open(temp_text_file, "w", encoding="utf-8") as out:
        for file_path in files:
            if file_path.suffix.lower() == ".md":
                with open(file_path, "r", encoding="utf-8") as f:
                    out.write(f.read() + "\n")
            elif file_path.suffix.lower() == ".epub":
                from dataset import extract_text_from_epub
                out.write(extract_text_from_epub(str(file_path)) + "\n")
            elif file_path.suffix.lower() == ".pdf":
                from dataset import extract_text_from_pdf
                out.write(extract_text_from_pdf(str(file_path)) + "\n")
    
    spm.SentencePieceTrainer.train(
        input=temp_text_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="BPE",
        character_coverage=0.9995,
        pad_id=3,
        bos_id=1,
        eos_id=2,
        unk_id=0,
        user_defined_symbols=["<|start_table|>", "<|end_table|>"],
        unk_piece="<|unk|>",
        bos_piece="<|bos|>",
        eos_piece="<|eos|>",
        pad_piece="<|pad|>",
        byte_fallback=True, # Allow bytes for unknown characters
        max_sentence_length=32768 # Handle long lines in PDFs/Epubs
    )
    
    os.remove(temp_text_file)
    print(f"Tokenizer saved to {model_prefix}.model and {model_prefix}.vocab")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing .md files")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size")
    args = parser.parse_args()
    
    # Fallback to smaller vocab size if training data is too small
    # Sentencepiece requires at least vocab_size sentences/tokens usually
    # For sample data, we might need a much smaller vocab
    try:
        train_tokenizer(args.data_dir, args.vocab_size)
    except Exception as e:
        print(f"\n[!] Tokenizer training failed for vocab_size={args.vocab_size}")
        print(f"Error: {e}")
        print("-" * 50)
        print("POSSIBLE CAUSE: Your training data is too small for a 32,000 word dictionary.")
        print("To fix this, either:")
        print("  1. Add more .pdf / .epub / .md files to your training_data folder.")
        print("  2. Run again with a smaller size, e.g., --vocab_size 2000")
        print("-" * 50)
        
        # Automatic fallback for convenience
        print("\nRetrying with vocab size 1000 for testing purposes...")
        try:
            train_tokenizer(args.data_dir, 1000)
        except:
            print("Still too small. Falling back to vocab size 500.")
            train_tokenizer(args.data_dir, 500)
