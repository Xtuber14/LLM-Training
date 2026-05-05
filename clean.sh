#!/bin/bash

# Script to clean all training artifacts and temporary files

echo "Cleaning training artifacts..."

# Remove checkpoints
if [ -d "checkpoints" ]; then
    echo "Removing checkpoints/..."
    rm -rf checkpoints
fi

if [ -d "checkpoints_sft" ]; then
    echo "Removing checkpoints_sft/..."
    rm -rf checkpoints_sft
fi

# Remove tokenizer files
echo "Removing tokenizer files..."
rm -f tokenizer.model tokenizer.vocab all_text.txt

# Remove binary dataset files
echo "Removing binary dataset files..."
rm -f data.bin data_train.bin data_val.bin
rm -f training_data/data.bin training_data/data_train.bin training_data/data_val.bin

# Remove Python cache
echo "Removing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} +
rm -rf .pytest_cache

echo "Done. Environment is clean."
