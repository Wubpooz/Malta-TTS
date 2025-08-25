# Simple BPE Tokenizer Extension

This module provides a simple, dependency-light solution for extending existing BPE tokenizers with new vocabulary while preserving all existing token mappings and IDs.

## Problem Solved

When working with multilingual TTS systems or adding new languages to existing models, you need to extend the tokenizer vocabulary without breaking the existing model. This solution ensures:

- ✅ **No token ID shifts** - Existing tokens keep their exact same IDs
- ✅ **Preserved language features** - All existing language mappings remain intact  
- ✅ **Safe extension** - Comprehensive validation to detect any corruption
- ✅ **Simple interface** - Matches the user's original code requirements

## Quick Start

### Simple Usage (Character-level extension)

```python
from simple_tokenizer_extension import simple_extend_tokenizer

# This matches the user's original code snippet exactly
tokenizer = simple_extend_tokenizer(
    vocab_path="vocab.json",           # Your existing tokenizer
    csv_path="your_new_metadata.csv", # CSV with new text data  
    output_path="tokenizer.json"      # Output path
)
```

### Comprehensive Usage (Character + Word extension)

```python
from simple_tokenizer_extension import extend_tokenizer_safe

results = extend_tokenizer_safe(
    vocab_path="vocab.json",
    csv_path="your_new_metadata.csv",
    output_path="extended_tokenizer.json",
    min_frequency=2,           # Minimum word frequency
    max_new_tokens=5000,       # Limit new tokens
    add_characters=True,       # Add new characters
    add_words=True            # Add frequent words
)

if results["success"]:
    print(f"Extended vocabulary: {results['original_vocab_size']} → {results['extended_vocab_size']}")
    print(f"Added {results['tokens_added']} new tokens")
```

## Files Overview

- **`simple_tokenizer_extension.py`** - Main module with all functionality
- **`test_tokenizer_extension.py`** - Comprehensive test suite  
- **`example_usage.py`** - Demo script showing all features
- **`README.md`** - This documentation

## Key Features

### 1. Safe Token Extension
- Adds only tokens that don't already exist
- Character-level and word-level token discovery
- Frequency-based filtering for word tokens
- Configurable limits to prevent vocabulary explosion

### 2. Integrity Validation
- Detects any token ID shifts (critical corruption)
- Validates existing token mappings are preserved
- Tests tokenization consistency for original content
- Comprehensive reporting of any issues found

### 3. Multiple Interfaces
- **Simple**: Matches user's original code snippet exactly
- **Safe**: Full validation with comprehensive options
- **Command-line**: Direct CLI usage for automation

## CSV Format

Your CSV file should have a `text` column with pipe (|) separation:

```csv
text
"Hello world, this is sample text."
"Text with émojis: 😀 and special chars: àáâã"
"Foreign language: Bongu! Kif int?"
```

## Validation Process

The extension process includes comprehensive validation:

1. **Token ID Preservation**: Ensures existing tokens keep exact same IDs
2. **Mapping Integrity**: Validates all original token→ID mappings are preserved  
3. **Tokenization Consistency**: Tests that original content tokenizes identically
4. **Corruption Detection**: Detects and prevents any corruption

## Example Output

```
🚀 Starting safe tokenizer extension...
✅ Successfully loaded tokenizer from vocab.json
   Original vocabulary size: 2000
✅ Successfully loaded 500 texts from metadata.csv
🔍 Analyzing texts to identify new tokens...
   📊 Analysis results:
      - Total unique characters: 150
      - New characters to add: 45
      - Total unique words: 1200  
      - New words to add: 180
📝 Adding 45 new character tokens
📝 Adding 180 new word tokens
✅ Successfully added 225 new tokens
🔍 Validating tokenizer integrity...
   ✅ Tokenizer integrity maintained
      - No token ID shifts detected
      - Tokenization consistency preserved for original content
🎉 Tokenizer extension completed successfully!
   📁 Saved to: extended_tokenizer.json
   📊 Vocabulary: 2000 → 2225 (+225)
```

## Command Line Usage

```bash
# Simple extension (character tokens only)
python simple_tokenizer_extension.py \
  --vocab_path vocab.json \
  --csv_path your_new_metadata.csv \
  --output_path tokenizer.json \
  --simple

# Full extension with options
python simple_tokenizer_extension.py \
  --vocab_path vocab.json \
  --csv_path your_new_metadata.csv \
  --output_path extended_tokenizer.json \
  --min_frequency 2 \
  --max_new_tokens 5000
```

## Testing

Run the comprehensive test suite:

```bash
python test_tokenizer_extension.py
```

See the full demonstration:

```bash
python example_usage.py
```

## Key Benefits

1. **Zero Token ID Shifts** - Critical for maintaining model compatibility
2. **Comprehensive Validation** - Detects any potential corruption 
3. **Flexible Options** - Character-only or word+character extension
4. **Safety First** - Aborts on any detected corruption
5. **Simple Interface** - Easy to integrate into existing workflows
6. **Well Tested** - Comprehensive test suite validates all functionality

## Error Handling

The module includes robust error handling:

- File validation (checks if files exist and are readable)
- CSV format validation (ensures required columns exist)
- Tokenizer integrity validation (detects corruption)
- Safe failure mode (aborts rather than corrupting data)

## Dependencies

Minimal dependencies for maximum compatibility:

- `tokenizers` - Core BPE tokenizer library
- `pandas` - CSV file processing  
- Standard library modules only

## License

This code is part of the Malta-TTS project and follows the same license terms.