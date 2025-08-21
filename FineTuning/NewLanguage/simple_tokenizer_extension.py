#!/usr/bin/env python3
"""
Simple BPE Tokenizer Extension without Dependencies

This module provides a simple, dependency-light implementation for extending
an existing BPE tokenizer with new tokens while preserving existing token IDs.

Key Features:
- Load existing tokenizer from vocab.json
- Add new tokens from CSV data without shifting existing token IDs
- Preserve all existing token mappings and language features
- Comprehensive validation to ensure no corruption occurs
"""

import json
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from collections import Counter
import os


def load_existing_tokenizer(vocab_path):
    """
    Load existing BPE tokenizer from vocab.json file.
    
    Args:
        vocab_path (str): Path to the vocab.json file
        
    Returns:
        Tokenizer: Loaded tokenizer with whitespace pre-tokenizer
        
    Raises:
        FileNotFoundError: If vocab.json file doesn't exist
        Exception: If tokenizer loading fails
    """
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Tokenizer file not found: {vocab_path}")
    
    try:
        tokenizer = Tokenizer.from_file(vocab_path)
        tokenizer.pre_tokenizer = Whitespace()
        print(f"✅ Successfully loaded tokenizer from {vocab_path}")
        print(f"   Original vocabulary size: {tokenizer.get_vocab_size()}")
        return tokenizer
    except Exception as e:
        raise Exception(f"Failed to load tokenizer from {vocab_path}: {str(e)}")


def extract_texts_from_csv(csv_path, text_column="text", separator="|"):
    """
    Extract text data from CSV file.
    
    Args:
        csv_path (str): Path to the CSV file
        text_column (str): Name of the text column (default: "text")
        separator (str): CSV separator (default: "|")
        
    Returns:
        list: List of text strings
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        KeyError: If text column doesn't exist
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path, sep=separator, usecols=[text_column])
        texts = df[text_column].astype(str).tolist()
        print(f"✅ Successfully loaded {len(texts)} texts from {csv_path}")
        return texts
    except KeyError:
        raise KeyError(f"Text column '{text_column}' not found in CSV file")
    except Exception as e:
        raise Exception(f"Failed to load CSV file {csv_path}: {str(e)}")


def identify_new_tokens(tokenizer, texts, min_frequency=1, token_type="characters"):
    """
    Identify new tokens that need to be added to the tokenizer.
    
    Args:
        tokenizer (Tokenizer): Existing tokenizer
        texts (list): List of text strings
        min_frequency (int): Minimum frequency threshold for tokens
        token_type (str): Type of tokens to extract ("characters" or "words")
        
    Returns:
        tuple: (new_character_tokens, new_word_tokens, statistics)
    """
    print(f"🔍 Analyzing texts to identify new tokens...")
    
    # Character-level analysis
    all_chars = set()
    for text in texts:
        all_chars.update(list(text))
    
    # Find new characters not in tokenizer
    new_chars = []
    for char in all_chars:
        if tokenizer.token_to_id(char) is None and not char.isspace():
            new_chars.append(char)
    
    # Word-level analysis with frequency
    word_freq = Counter()
    for text in texts:
        for word in text.strip().split():
            word = word.strip()
            if word:
                word_freq[word] += 1
    
    # Find new words not effectively present in tokenizer
    new_words = []
    for word, count in word_freq.most_common():
        if count < min_frequency:
            break  # Below threshold
            
        # Check if word is already effectively in tokenizer
        try:
            existing_id = tokenizer.token_to_id(word)
        except Exception:
            existing_id = None
            
        if existing_id is not None:
            continue  # Already exists
            
        # Check if word encodes to a single token that matches
        enc = tokenizer.encode(word)
        if len(enc.ids) == 1 and enc.tokens and enc.tokens[0] == word:
            continue  # Effectively present
            
        new_words.append(word)
    
    stats = {
        "total_characters": len(all_chars),
        "new_characters": len(new_chars),
        "total_unique_words": len(word_freq),
        "new_words": len(new_words),
        "words_above_threshold": sum(1 for count in word_freq.values() if count >= min_frequency)
    }
    
    print(f"   📊 Analysis results:")
    print(f"      - Total unique characters: {stats['total_characters']}")
    print(f"      - New characters to add: {stats['new_characters']}")
    print(f"      - Total unique words: {stats['total_unique_words']}")
    print(f"      - Words above frequency threshold: {stats['words_above_threshold']}")
    print(f"      - New words to add: {stats['new_words']}")
    
    return new_chars, new_words, stats


def validate_tokenizer_integrity(original_tokenizer, extended_tokenizer, test_texts=None):
    """
    Validate that tokenizer extension didn't corrupt existing mappings.
    
    Note: Tokenization changes for text containing new tokens are expected and normal.
    This function focuses on validating that existing token IDs are preserved.
    
    Args:
        original_tokenizer (Tokenizer): Original tokenizer
        extended_tokenizer (Tokenizer): Extended tokenizer
        test_texts (list): Optional test texts for validation (should be texts that existed in original training)
        
    Returns:
        dict: Validation results
    """
    print("🔍 Validating tokenizer integrity...")
    
    results = {
        "token_id_shifts": 0,
        "tokenization_changes_in_old_text": 0,
        "corruption_detected": False,
        "details": []
    }
    
    # Get vocabularies
    orig_vocab = original_tokenizer.get_vocab()
    ext_vocab = extended_tokenizer.get_vocab()
    
    # Check for token ID shifts in existing tokens (THIS IS THE CRITICAL CHECK)
    common_tokens = set(orig_vocab.keys()) & set(ext_vocab.keys())
    
    for token in common_tokens:
        if orig_vocab[token] != ext_vocab[token]:
            results["token_id_shifts"] += 1
            results["details"].append(f"CRITICAL: Token '{token}' ID changed: {orig_vocab[token]} → {ext_vocab[token]}")
    
    # Test common English words/characters that should be stable
    stable_test_tokens = ["hello", "world", "the", "and", "is", "that", "you", "for", "a", ".", ","]
    
    for token in stable_test_tokens:
        orig_id = orig_vocab.get(token)
        ext_id = ext_vocab.get(token)
        
        if orig_id is not None and orig_id != ext_id:
            results["token_id_shifts"] += 1
            results["details"].append(f"CRITICAL: Stable token '{token}' ID changed: {orig_id} → {ext_id}")
    
    # Test tokenization consistency for ORIGINAL training-like texts only
    # (texts that don't contain new characters should tokenize identically)
    if test_texts:
        original_like_texts = []
        for text in test_texts[:5]:
            # Only test texts that likely don't contain new characters
            if all(c.isascii() and (c.isalnum() or c.isspace() or c in ".,!?'-") for c in text):
                original_like_texts.append(text)
        
        for text in original_like_texts:
            orig_tokens = original_tokenizer.encode(text).ids
            ext_tokens = extended_tokenizer.encode(text).ids
            
            if orig_tokens != ext_tokens:
                results["tokenization_changes_in_old_text"] += 1
                results["details"].append(f"WARNING: Original-like text tokenization changed: '{text[:50]}...'")
    
    # Only consider token ID shifts as corruption (tokenization changes for new content are normal)
    results["corruption_detected"] = (results["token_id_shifts"] > 0)
    
    if results["corruption_detected"]:
        print(f"   ❌ CORRUPTION DETECTED!")
        print(f"      - Token ID shifts: {results['token_id_shifts']}")
        if results["tokenization_changes_in_old_text"] > 0:
            print(f"      - Tokenization changes in old text: {results['tokenization_changes_in_old_text']}")
        for detail in results["details"][:5]:  # Show first 5 issues
            print(f"      - {detail}")
    else:
        print(f"   ✅ Tokenizer integrity maintained")
        print(f"      - No token ID shifts detected")
        if results["tokenization_changes_in_old_text"] > 0:
            print(f"      - Warning: {results['tokenization_changes_in_old_text']} original-like texts changed tokenization")
        else:
            print(f"      - Tokenization consistency preserved for original content")
    
    return results


def extend_tokenizer_safe(vocab_path, csv_path, output_path=None, 
                         text_column="text", separator="|", 
                         min_frequency=1, max_new_tokens=10000,
                         add_characters=True, add_words=True):
    """
    Safely extend BPE tokenizer with new tokens while preserving existing mappings.
    
    Args:
        vocab_path (str): Path to existing vocab.json file
        csv_path (str): Path to CSV file with new text data
        output_path (str): Path to save extended tokenizer (default: tokenizer.json)
        text_column (str): Name of text column in CSV (default: "text")
        separator (str): CSV separator (default: "|")
        min_frequency (int): Minimum frequency for word tokens (default: 1)
        max_new_tokens (int): Maximum number of new tokens to add (default: 10000)
        add_characters (bool): Whether to add new character tokens (default: True)
        add_words (bool): Whether to add new word tokens (default: True)
        
    Returns:
        dict: Results including tokenizer, statistics, and validation
    """
    print("🚀 Starting safe tokenizer extension...")
    
    # Load existing tokenizer
    original_tokenizer = load_existing_tokenizer(vocab_path)
    
    # Create a copy for extension
    extended_tokenizer = Tokenizer.from_file(vocab_path)
    extended_tokenizer.pre_tokenizer = Whitespace()
    
    # Extract texts from CSV
    texts = extract_texts_from_csv(csv_path, text_column, separator)
    
    # Identify new tokens
    new_chars, new_words, stats = identify_new_tokens(extended_tokenizer, texts, min_frequency)
    
    # Add new tokens
    tokens_to_add = []
    
    if add_characters and new_chars:
        tokens_to_add.extend(new_chars)
        print(f"📝 Adding {len(new_chars)} new character tokens")
    
    if add_words and new_words:
        # Limit word tokens to prevent explosion
        word_tokens = new_words[:max_new_tokens - len(tokens_to_add)]
        tokens_to_add.extend(word_tokens)
        print(f"📝 Adding {len(word_tokens)} new word tokens")
    
    if tokens_to_add:
        # Limit total additions
        if len(tokens_to_add) > max_new_tokens:
            tokens_to_add = tokens_to_add[:max_new_tokens]
            print(f"⚠️  Limited to {max_new_tokens} new tokens")
        
        extended_tokenizer.add_tokens(tokens_to_add)
        print(f"✅ Successfully added {len(tokens_to_add)} new tokens")
    else:
        print("ℹ️  No new tokens to add")
    
    # Validate integrity
    validation = validate_tokenizer_integrity(original_tokenizer, extended_tokenizer, texts[:5])
    
    if validation["corruption_detected"]:
        print("❌ ABORTING: Tokenizer corruption detected!")
        return {
            "success": False,
            "error": "Tokenizer corruption detected",
            "validation": validation
        }
    
    # Save extended tokenizer
    if output_path is None:
        output_path = "tokenizer.json"
    
    extended_tokenizer.save(output_path)
    
    results = {
        "success": True,
        "original_vocab_size": original_tokenizer.get_vocab_size(),
        "extended_vocab_size": extended_tokenizer.get_vocab_size(),
        "tokens_added": len(tokens_to_add),
        "new_characters": len(new_chars) if add_characters else 0,
        "new_words": len(new_words) if add_words else 0,
        "output_path": output_path,
        "statistics": stats,
        "validation": validation,
        "tokenizer": extended_tokenizer
    }
    
    print(f"🎉 Tokenizer extension completed successfully!")
    print(f"   📁 Saved to: {output_path}")
    print(f"   📊 Vocabulary: {results['original_vocab_size']} → {results['extended_vocab_size']} (+{results['tokens_added']})")
    
    return results


def simple_extend_tokenizer(vocab_path, csv_path, output_path="tokenizer.json"):
    """
    Simple interface that matches the user's original code snippet.
    
    Args:
        vocab_path (str): Path to vocab.json file
        csv_path (str): Path to CSV file with text column
        output_path (str): Output path for extended tokenizer
        
    Returns:
        Tokenizer: Extended tokenizer
    """
    print("🚀 Simple tokenizer extension (matching user's code snippet)...")
    
    # Load existing tokenizer
    tokenizer = Tokenizer.from_file(vocab_path)
    tokenizer.pre_tokenizer = Whitespace()
    
    # Load new texts
    df = pd.read_csv(csv_path, sep="|", usecols=["text"])
    texts = df["text"].astype(str).tolist()
    
    # Collect new tokens (character-level)
    all_chars = set()
    for t in texts:
        all_chars.update(list(t))
    
    # Only add tokens not already present
    to_add = [c for c in all_chars if tokenizer.token_to_id(c) is None and not c.isspace()]
    if to_add:
        tokenizer.add_tokens(to_add)
        print(f"Added {len(to_add)} new tokens.")
    else:
        print("No new tokens to add.")
    
    # Save updated tokenizer
    tokenizer.save(output_path)
    print(f"Tokenizer saved to {output_path}")
    
    return tokenizer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extend BPE tokenizer with new tokens")
    parser.add_argument("--vocab_path", required=True, help="Path to existing vocab.json file")
    parser.add_argument("--csv_path", required=True, help="Path to CSV file with new text data")
    parser.add_argument("--output_path", default="tokenizer.json", help="Output path for extended tokenizer")
    parser.add_argument("--text_column", default="text", help="Name of text column in CSV")
    parser.add_argument("--separator", default="|", help="CSV separator")
    parser.add_argument("--min_frequency", type=int, default=1, help="Minimum frequency for word tokens")
    parser.add_argument("--max_new_tokens", type=int, default=10000, help="Maximum new tokens to add")
    parser.add_argument("--simple", action="store_true", help="Use simple mode (character tokens only)")
    
    args = parser.parse_args()
    
    if args.simple:
        simple_extend_tokenizer(args.vocab_path, args.csv_path, args.output_path)
    else:
        extend_tokenizer_safe(
            vocab_path=args.vocab_path,
            csv_path=args.csv_path,
            output_path=args.output_path,
            text_column=args.text_column,
            separator=args.separator,
            min_frequency=args.min_frequency,
            max_new_tokens=args.max_new_tokens
        )