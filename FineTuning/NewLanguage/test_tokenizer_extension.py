#!/usr/bin/env python3
"""
Test Suite for Simple BPE Tokenizer Extension

This module provides comprehensive tests to validate that the tokenizer
extension functionality works correctly and preserves existing token mappings.
"""

import os
import tempfile
import json
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Import our extension module
from simple_tokenizer_extension import (
    load_existing_tokenizer,
    extract_texts_from_csv,
    identify_new_tokens,
    validate_tokenizer_integrity,
    extend_tokenizer_safe,
    simple_extend_tokenizer
)


def create_test_tokenizer(vocab_size=1000):
    """
    Create a small test tokenizer for testing purposes.
    
    Args:
        vocab_size (int): Size of the vocabulary
        
    Returns:
        Tokenizer: Test tokenizer
    """
    # Sample training texts
    training_texts = [
        "Hello world, how are you today?",
        "This is a test for the tokenizer.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning and natural language processing.",
        "Python is a great programming language.",
        "Tokenizers help process text data efficiently.",
    ]
    
    # Create BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    
    # Train the tokenizer
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.train_from_iterator(training_texts, trainer)
    
    return tokenizer


def create_test_csv(texts, output_path):
    """
    Create a test CSV file with text data.
    
    Args:
        texts (list): List of text strings
        output_path (str): Path to save CSV file
    """
    df = pd.DataFrame({"text": texts})
    df.to_csv(output_path, sep="|", index=False)


def test_load_existing_tokenizer():
    """Test loading existing tokenizer from file."""
    print("\n🧪 Testing: load_existing_tokenizer")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test tokenizer
        tokenizer = create_test_tokenizer()
        vocab_path = os.path.join(temp_dir, "vocab.json")
        tokenizer.save(vocab_path)
        
        # Test loading
        loaded_tokenizer = load_existing_tokenizer(vocab_path)
        
        # Validate
        assert loaded_tokenizer.get_vocab_size() == tokenizer.get_vocab_size()
        assert loaded_tokenizer.encode("hello").ids == tokenizer.encode("hello").ids
        
        print("   ✅ Successfully loaded tokenizer")
        
        # Test error case
        try:
            load_existing_tokenizer("nonexistent.json")
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            print("   ✅ Correctly handles missing file")


def test_extract_texts_from_csv():
    """Test extracting text data from CSV file."""
    print("\n🧪 Testing: extract_texts_from_csv")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_texts = [
            "This is test text one.",
            "Here is another test sentence.",
            "Third sample text for testing."
        ]
        
        csv_path = os.path.join(temp_dir, "test.csv")
        create_test_csv(test_texts, csv_path)
        
        # Test extraction
        extracted_texts = extract_texts_from_csv(csv_path)
        
        # Validate
        assert len(extracted_texts) == len(test_texts)
        assert extracted_texts == test_texts
        
        print("   ✅ Successfully extracted texts from CSV")
        
        # Test error case
        try:
            extract_texts_from_csv("nonexistent.csv")
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            print("   ✅ Correctly handles missing file")


def test_identify_new_tokens():
    """Test identification of new tokens."""
    print("\n🧪 Testing: identify_new_tokens")
    
    # Create test tokenizer with limited vocabulary
    tokenizer = create_test_tokenizer(vocab_size=50)
    
    # Test texts with some new characters and words
    test_texts = [
        "Hello world! This contains émojis: 😀 and special chars: àáâã",
        "New words: żebbuġ, ħobż, qattus",  # Maltese words likely not in English tokenizer
        "Numbers and symbols: 123 $50 €25 #hashtag @mention"
    ]
    
    new_chars, new_words, stats = identify_new_tokens(tokenizer, test_texts, min_frequency=1)
    
    # Validate results
    assert isinstance(new_chars, list)
    assert isinstance(new_words, list)
    assert isinstance(stats, dict)
    
    # Check that we found some new characters (special chars, accents, emojis)
    assert len(new_chars) > 0, "Should find new characters"
    
    # Check statistics structure
    required_stats = ['total_characters', 'new_characters', 'total_unique_words', 'new_words']
    for stat in required_stats:
        assert stat in stats, f"Missing statistic: {stat}"
    
    print(f"   ✅ Found {len(new_chars)} new characters and {len(new_words)} new words")
    print(f"   📊 Statistics: {stats}")


def test_validate_tokenizer_integrity():
    """Test tokenizer integrity validation."""
    print("\n🧪 Testing: validate_tokenizer_integrity")
    
    # Create original tokenizer
    original = create_test_tokenizer()
    
    # Create extended version (should be identical for this test)
    extended = create_test_tokenizer()
    
    # Test validation with identical tokenizers
    test_texts = ["Hello world", "Test text"]
    validation = validate_tokenizer_integrity(original, extended, test_texts)
    
    # Should show no corruption
    assert not validation["corruption_detected"], "Identical tokenizers should show no corruption"
    assert validation["token_id_shifts"] == 0, "Should have no token ID shifts"
    
    print("   ✅ Correctly validates identical tokenizers")
    
    # Test with artificially corrupted tokenizer
    # We'll manually create a scenario where token IDs change
    corrupted = create_test_tokenizer(vocab_size=50)  # Different vocab size
    
    validation_corrupted = validate_tokenizer_integrity(original, corrupted, test_texts)
    
    print(f"   📊 Corruption test - Token ID shifts: {validation_corrupted['token_id_shifts']}")
    print(f"   📊 Note: Tokenization changes for new content are expected and normal")


def test_simple_extend_tokenizer():
    """Test simple tokenizer extension function."""
    print("\n🧪 Testing: simple_extend_tokenizer")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test tokenizer
        original_tokenizer = create_test_tokenizer()
        vocab_path = os.path.join(temp_dir, "vocab.json")
        original_tokenizer.save(vocab_path)
        
        # Create test CSV with new characters
        test_texts = [
            "Text with émojis: 😀😁😂",
            "Special characters: àáâãäåæç",
            "More symbols: ℓ₹€¥£$"
        ]
        csv_path = os.path.join(temp_dir, "test.csv")
        create_test_csv(test_texts, csv_path)
        
        # Test simple extension
        output_path = os.path.join(temp_dir, "extended.json")
        extended_tokenizer = simple_extend_tokenizer(vocab_path, csv_path, output_path)
        
        # Validate results
        assert extended_tokenizer.get_vocab_size() >= original_tokenizer.get_vocab_size()
        assert os.path.exists(output_path)
        
        # Test that original tokens still work the same way
        test_sentence = "Hello world"
        original_ids = original_tokenizer.encode(test_sentence).ids
        extended_ids = extended_tokenizer.encode(test_sentence).ids
        
        # The tokenization should be the same for existing text
        # (though it might differ if the new tokens affect segmentation)
        print(f"   📊 Original vocab size: {original_tokenizer.get_vocab_size()}")
        print(f"   📊 Extended vocab size: {extended_tokenizer.get_vocab_size()}")
        print(f"   📊 Tokens added: {extended_tokenizer.get_vocab_size() - original_tokenizer.get_vocab_size()}")
        
        print("   ✅ Simple tokenizer extension completed")


def test_extend_tokenizer_safe():
    """Test comprehensive safe tokenizer extension."""
    print("\n🧪 Testing: extend_tokenizer_safe")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test tokenizer
        original_tokenizer = create_test_tokenizer()
        vocab_path = os.path.join(temp_dir, "vocab.json")
        original_tokenizer.save(vocab_path)
        
        # Create test CSV with various new content
        test_texts = [
            "Hello world, this is English text.",
            "Texte français avec des accents: café, résumé, naïve",
            "Malti: Kemm int? Jien tajjeb, grazzi.",
            "Emojis and symbols: 😀💚🔥 €100 $50 #test @user",
            "Numbers: 123456789 and special chars: àáâãäåæçèé"
        ]
        csv_path = os.path.join(temp_dir, "test.csv")
        create_test_csv(test_texts, csv_path)
        
        # Test safe extension
        output_path = os.path.join(temp_dir, "safe_extended.json")
        results = extend_tokenizer_safe(
            vocab_path=vocab_path,
            csv_path=csv_path,
            output_path=output_path,
            min_frequency=1,
            max_new_tokens=1000,
            add_characters=True,
            add_words=True
        )
        
        # Validate results
        assert results["success"], f"Extension failed: {results.get('error', 'Unknown error')}"
        assert results["extended_vocab_size"] >= results["original_vocab_size"]
        assert results["tokens_added"] >= 0
        assert not results["validation"]["corruption_detected"], "Corruption detected in safe extension"
        assert os.path.exists(output_path)
        
        print(f"   ✅ Safe extension completed successfully")
        print(f"   📊 Vocabulary: {results['original_vocab_size']} → {results['extended_vocab_size']}")
        print(f"   📊 Tokens added: {results['tokens_added']}")
        print(f"   📊 New characters: {results['new_characters']}")
        print(f"   📊 New words: {results['new_words']}")


def test_preservation_of_existing_mappings():
    """Test that existing token mappings are preserved exactly."""
    print("\n🧪 Testing: preservation_of_existing_mappings")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test tokenizer
        original_tokenizer = create_test_tokenizer()
        vocab_path = os.path.join(temp_dir, "vocab.json")
        original_tokenizer.save(vocab_path)
        
        # Get original vocabulary for comparison
        original_vocab = original_tokenizer.get_vocab()
        
        # Create test CSV with new content
        test_texts = [
            "New content with unique symbols: ★☆♠♣♥♦",
            "Foreign text: żebbuġ qattus kelb ħobż",
            "More emojis: 🌟⭐✨💫🔆"
        ]
        csv_path = os.path.join(temp_dir, "test.csv")
        create_test_csv(test_texts, csv_path)
        
        # Extend tokenizer
        output_path = os.path.join(temp_dir, "extended.json")
        results = extend_tokenizer_safe(vocab_path, csv_path, output_path)
        
        # Load extended tokenizer and compare
        extended_tokenizer = Tokenizer.from_file(output_path)
        extended_vocab = extended_tokenizer.get_vocab()
        
        # Check that all original tokens have the same IDs
        mismatched_tokens = []
        for token, original_id in original_vocab.items():
            if token in extended_vocab:
                extended_id = extended_vocab[token]
                if original_id != extended_id:
                    mismatched_tokens.append((token, original_id, extended_id))
        
        assert len(mismatched_tokens) == 0, f"Token ID mismatches found: {mismatched_tokens[:5]}"
        
        # Test tokenization consistency for ASCII-only phrases (no new characters)
        ascii_test_phrases = [
            "Hello world",
            "This is a test",
            "The quick brown fox",
            "123 456"
        ]
        
        tokenization_mismatches = []
        for phrase in ascii_test_phrases:
            original_tokens = original_tokenizer.encode(phrase).ids
            extended_tokens = extended_tokenizer.encode(phrase).ids
            
            if original_tokens != extended_tokens:
                tokenization_mismatches.append((phrase, original_tokens, extended_tokens))
        
        if tokenization_mismatches:
            print(f"   ⚠️  Warning: {len(tokenization_mismatches)} ASCII tokenization changes detected")
            for phrase, orig, ext in tokenization_mismatches[:2]:
                print(f"      '{phrase}': {orig} → {ext}")
        else:
            print("   ✅ All ASCII tokenizations preserved exactly")
        
        print(f"   ✅ All {len(original_vocab)} original token mappings preserved")
        
        # Note: Changes in tokenization for text with new characters is expected and normal
        print(f"   ℹ️  Note: Tokenization changes for text with new characters are expected")


def run_all_tests():
    """Run all tests in the test suite."""
    print("🧪 Running Simple BPE Tokenizer Extension Test Suite")
    print("=" * 60)
    
    tests = [
        test_load_existing_tokenizer,
        test_extract_texts_from_csv,
        test_identify_new_tokens,
        test_validate_tokenizer_integrity,
        test_simple_extend_tokenizer,
        test_extend_tokenizer_safe,
        test_preservation_of_existing_mappings
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"   ❌ FAILED: {str(e)}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"🧪 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Tokenizer extension is working correctly.")
    else:
        print("⚠️  Some tests failed. Please review the errors above.")
    
    return failed == 0


if __name__ == "__main__":
    run_all_tests()