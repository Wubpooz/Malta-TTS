#!/usr/bin/env python3
"""
Example Usage of Simple BPE Tokenizer Extension

This script demonstrates how to use the simple tokenizer extension functionality
to extend an existing BPE tokenizer with new vocabulary from CSV data while
preserving existing token mappings.
"""

import os
import tempfile
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from simple_tokenizer_extension import (
    simple_extend_tokenizer,
    extend_tokenizer_safe,
    validate_tokenizer_integrity
)


def create_demo_tokenizer():
    """Create a demo tokenizer for the example."""
    print("🔨 Creating demo tokenizer...")
    
    # Sample training texts (English)
    training_texts = [
        "Hello world, how are you today?",
        "This is a test for the tokenizer.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning and natural language processing are fascinating.",
        "Python is a great programming language for data science.",
        "Tokenizers help us process text data efficiently.",
        "Natural language understanding requires good tokenization.",
        "Deep learning models work better with proper text preprocessing."
    ]
    
    # Create BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    
    # Train the tokenizer with a reasonable vocabulary size
    trainer = BpeTrainer(
        vocab_size=2000,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )
    tokenizer.train_from_iterator(training_texts, trainer)
    
    print(f"   ✅ Created tokenizer with {tokenizer.get_vocab_size()} tokens")
    return tokenizer


def create_demo_csv(output_path):
    """Create a demo CSV file with new text data."""
    print("📝 Creating demo CSV with new text data...")
    
    # Sample texts with new vocabulary (including Maltese, emojis, special characters)
    new_texts = [
        "Bongu! Kif int illum? Jien tajjeb ħafna, grazzi.",  # Maltese greeting
        "Il-qattus qiegħed jiekol il-ħobż fuq il-mejda.",    # Maltese sentence
        "Hello! 😀 This text contains emojis: 🌟✨💫🔥",      # Emojis
        "Café résumé naïve piñata jalapeño façade",          # Accented characters
        "Price: €50, £30, $40, ¥500, ₹1000",                # Currency symbols
        "Email: test@example.com, Website: https://test.org", # URLs/emails
        "Symbols: ★☆♠♣♥♦◆◇○●△▲□■",                          # Special symbols
        "Math: α+β=γ, ∑∆∏∫√∞±≤≥≠≈",                          # Mathematical symbols
        "Għandi ħamest klieb u tliet qtates fil-ġnien.",     # More Maltese
        "Temperatures: 25°C, 77°F, 298K",                    # Temperature units
        "Fractions: ½ ¼ ¾ ⅓ ⅔ ⅛ ⅜ ⅝ ⅞",                      # Fractions
        "Arrows: ←→↑↓↔↕⇒⇔⇑⇓",                               # Arrows
        "Programming: def function(): return 'hello'",       # Code
        "Music: ♪♫♬♩♭♮♯𝄞𝄢",                                  # Musical symbols
        "Weather: ☀️☁️🌧️❄️⛈️🌩️⛅",                           # Weather emojis
    ]
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame({"text": new_texts})
    df.to_csv(output_path, sep="|", index=False)
    
    print(f"   ✅ Created CSV with {len(new_texts)} text samples")
    print(f"   📁 Saved to: {output_path}")
    
    return new_texts


def demonstrate_simple_usage():
    """Demonstrate the simple usage matching the user's code snippet."""
    print("\n" + "="*60)
    print("🚀 DEMO 1: Simple Usage (matches user's code snippet)")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create demo files
        tokenizer = create_demo_tokenizer()
        vocab_path = os.path.join(temp_dir, "vocab.json")
        tokenizer.save(vocab_path)
        
        csv_path = os.path.join(temp_dir, "your_new_metadata.csv")
        new_texts = create_demo_csv(csv_path)
        
        output_path = os.path.join(temp_dir, "tokenizer.json")
        
        print(f"\n📊 Original tokenizer stats:")
        print(f"   Vocabulary size: {tokenizer.get_vocab_size()}")
        
        # Test a few original phrases
        test_phrases = ["Hello world", "This is a test", "Python programming"]
        print(f"   Original tokenizations:")
        for phrase in test_phrases:
            tokens = tokenizer.encode(phrase)
            print(f"     '{phrase}' → {tokens.ids}")
        
        # Demonstrate the simple extension (matching user's code)
        print(f"\n🔧 Extending tokenizer with simple method...")
        extended_tokenizer = simple_extend_tokenizer(vocab_path, csv_path, output_path)
        
        print(f"\n📊 Extended tokenizer stats:")
        print(f"   Vocabulary size: {extended_tokenizer.get_vocab_size()}")
        print(f"   Tokens added: {extended_tokenizer.get_vocab_size() - tokenizer.get_vocab_size()}")
        
        # Test that original phrases still tokenize the same way
        print(f"   Extended tokenizations (should be identical):")
        for phrase in test_phrases:
            tokens = extended_tokenizer.encode(phrase)
            print(f"     '{phrase}' → {tokens.ids}")
        
        # Test new content
        print(f"\n🆕 Testing new content:")
        new_test_phrases = ["Bongu! 😀", "Café résumé", "€50 $40"]
        for phrase in new_test_phrases:
            original_tokens = tokenizer.encode(phrase)
            extended_tokens = extended_tokenizer.encode(phrase)
            print(f"     '{phrase}':")
            print(f"       Original: {original_tokens.ids} (tokens: {original_tokens.tokens})")
            print(f"       Extended: {extended_tokens.ids} (tokens: {extended_tokens.tokens})")


def demonstrate_safe_usage():
    """Demonstrate the comprehensive safe usage."""
    print("\n" + "="*60)
    print("🛡️  DEMO 2: Safe Usage (comprehensive with validation)")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create demo files
        tokenizer = create_demo_tokenizer()
        vocab_path = os.path.join(temp_dir, "vocab.json")
        tokenizer.save(vocab_path)
        
        csv_path = os.path.join(temp_dir, "your_new_metadata.csv")
        new_texts = create_demo_csv(csv_path)
        
        output_path = os.path.join(temp_dir, "safe_tokenizer.json")
        
        print(f"\n📊 Original tokenizer stats:")
        print(f"   Vocabulary size: {tokenizer.get_vocab_size()}")
        
        # Demonstrate safe extension with full validation
        print(f"\n🔧 Extending tokenizer with safe method...")
        results = extend_tokenizer_safe(
            vocab_path=vocab_path,
            csv_path=csv_path,
            output_path=output_path,
            text_column="text",
            separator="|",
            min_frequency=1,
            max_new_tokens=5000,
            add_characters=True,
            add_words=True
        )
        
        if results["success"]:
            print(f"\n🎉 Extension completed successfully!")
            print(f"   📊 Results summary:")
            print(f"      Original vocabulary: {results['original_vocab_size']}")
            print(f"      Extended vocabulary: {results['extended_vocab_size']}")
            print(f"      Total tokens added: {results['tokens_added']}")
            print(f"      New characters: {results['new_characters']}")
            print(f"      New words: {results['new_words']}")
            print(f"      Output saved to: {results['output_path']}")
            
            validation = results["validation"]
            if not validation["corruption_detected"]:
                print(f"   ✅ Tokenizer integrity validated - no corruption detected")
            else:
                print(f"   ⚠️  Validation warnings:")
                print(f"      Token ID shifts: {validation['token_id_shifts']}")
                print(f"      Tokenization changes: {validation['tokenization_changes']}")
        else:
            print(f"   ❌ Extension failed: {results.get('error', 'Unknown error')}")


def demonstrate_validation():
    """Demonstrate tokenizer integrity validation."""
    print("\n" + "="*60)
    print("🔍 DEMO 3: Tokenizer Integrity Validation")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create demo files
        original_tokenizer = create_demo_tokenizer()
        vocab_path = os.path.join(temp_dir, "vocab.json")
        original_tokenizer.save(vocab_path)
        
        csv_path = os.path.join(temp_dir, "your_new_metadata.csv")
        new_texts = create_demo_csv(csv_path)
        
        # Extend tokenizer
        extended_tokenizer = simple_extend_tokenizer(vocab_path, csv_path)
        
        # Validate integrity
        print(f"\n🔍 Validating tokenizer integrity...")
        validation_results = validate_tokenizer_integrity(
            original_tokenizer, 
            extended_tokenizer, 
            test_texts=["Hello world", "This is a test", "Programming in Python"]
        )
        
        print(f"\n📋 Validation Results:")
        print(f"   Corruption detected: {validation_results['corruption_detected']}")
        print(f"   Token ID shifts: {validation_results['token_id_shifts']}")
        print(f"   Tokenization changes in old text: {validation_results['tokenization_changes_in_old_text']}")
        
        if validation_results["details"]:
            print(f"   Issues found:")
            for detail in validation_results["details"][:5]:
                print(f"     - {detail}")
        else:
            print(f"   ✅ No issues found - tokenizer integrity maintained")


def main():
    """Run all demonstrations."""
    print("🌟 Simple BPE Tokenizer Extension - Usage Examples")
    print("="*60)
    print("This script demonstrates how to extend an existing BPE tokenizer")
    print("with new vocabulary while preserving existing token mappings.")
    print()
    
    try:
        # Run demonstrations
        demonstrate_simple_usage()
        demonstrate_safe_usage()
        demonstrate_validation()
        
        print("\n" + "="*60)
        print("🎉 All demonstrations completed successfully!")
        print("="*60)
        print()
        print("📋 Summary:")
        print("   ✅ Simple extension method works as expected")
        print("   ✅ Safe extension method provides comprehensive validation")
        print("   ✅ Tokenizer integrity validation detects any issues")
        print()
        print("🚀 You can now use these methods with your own data:")
        print("   1. Use simple_extend_tokenizer() for basic character-level extension")
        print("   2. Use extend_tokenizer_safe() for comprehensive word+character extension")
        print("   3. Use validate_tokenizer_integrity() to check for any corruption")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()