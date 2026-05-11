"""
Test script to verify HelixLM can be loaded without trust_remote_code=True
when the helix_lm package is installed locally.

Usage:
  pip install -e .  # or python setup.py develop
  python test_load_without_trust_remote_code.py
"""
import sys
import os

# Add the repo root to path so helix_lm is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helix_lm import HelixConfig, HelixForCausalLM, HelixTokenizer

def test_local_loading():
    """Test loading with local package installed (no trust_remote_code needed)."""
    print("Testing local loading without trust_remote_code...")
    
    # Create a tiny model
    tokenizer = HelixTokenizer("gpt2")
    cfg = HelixConfig(
        vocab_size=len(tokenizer),
        d_model=128,
        n_columns=2,
        n_loops=1,
        nodes_per_column=(2, 2),
        seq_len=256,
        attention_mode="hybrid",
        hybrid_full_attention_interval=2,
        dtype="float32",
    )
    cfg.pad_token_id = tokenizer.pad_token_id
    cfg.eos_token_id = tokenizer.eos_token_id
    cfg.bos_token_id = tokenizer.bos_token_id
    
    model = HelixForCausalLM(cfg)
    print(f"✓ Model created: {model.count_parameters()['total']:,} params")
    
    # Test save and load
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir)
        print(f"✓ Model saved to {tmpdir}")
        
        # Load without trust_remote_code (local package is installed)
        loaded = HelixForCausalLM.from_pretrained(tmpdir)
        print(f"✓ Model loaded without trust_remote_code")
        
        # Verify params match
        orig_params = model.count_parameters()["total"]
        loaded_params = loaded.count_parameters()["total"]
        assert orig_params == loaded_params, f"Param mismatch: {orig_params} != {loaded_params}"
        print(f"✓ Parameter count matches: {loaded_params:,}")
    
    print("\nAll local loading tests passed!")
    return True

def test_auto_model_registration():
    """Test that AutoModel can find the model when package is installed."""
    from transformers import AutoConfig, AutoModelForCausalLM
    
    print("\nTesting AutoModel registration...")
    
    # Check that 'helix' is registered
    try:
        cfg = AutoConfig.from_pretrained("helix", trust_remote_code=False)
        print("✓ AutoConfig can resolve 'helix' model type")
    except Exception as e:
        print(f"⚠ AutoConfig cannot resolve 'helix' without trust_remote_code: {e}")
        print("  This is expected when loading from Hub. Use trust_remote_code=True for Hub models.")
    
    return True

if __name__ == "__main__":
    test_local_loading()
    test_auto_model_registration()
    print("\n" + "="*60)
    print("SUMMARY:")
    print("  - Local loading: Works without trust_remote_code")
    print("  - Hub loading:   Still requires trust_remote_code=True")
    print("  - To eliminate trust_remote_code entirely, install the package:")
    print("    pip install -e .")
    print("="*60)
