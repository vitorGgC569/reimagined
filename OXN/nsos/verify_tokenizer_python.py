import sys
import os

# Add build/Release to path to find nsos_ext
sys.path.append(os.path.join(os.getcwd(), "build", "Release"))

try:
    import nsos_ext
    print("nsos_ext loaded successfully.")
except ImportError as e:
    print(f"Error loading nsos_ext: {e}")
    sys.exit(1)

def test_tokenizer():
    tok = nsos_ext.Tokenizer()
    
    # Test special tokens
    tok.add_special_tokens(["<|endoftext|>", "<|system|>"])
    
    # Add manual base vocabulary if needed (constructor does it for 0-255)
    # The BPE ranks need to be loaded or manually set if bound
    # Since we can't easily set BPE ranks from Python (not bound in bindings.cpp),
    # we'll test the special token splitting which IS bound.
    
    text = "<|system|>Hello World<|endoftext|>"
    ids = tok.encode(text)
    
    print(f"Encoded IDs: {ids}")
    decoded = tok.decode(ids)
    print(f"Decoded Text: {decoded}")
    
    assert decoded == text, f"Mismatch: {decoded} != {text}"
    print("Special token test passed!")

    # Test character encoding (base vocab)
    text2 = "abc"
    ids2 = tok.encode(text2)
    # a=97, b=98, c=99
    assert ids2 == [97, 98, 99], f"Base vocab mismatch: {ids2}"
    assert tok.decode(ids2) == text2
    print("Base vocab test passed!")

if __name__ == "__main__":
    test_tokenizer()
    print("\nAll Python verification tests passed!")
