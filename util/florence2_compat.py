"""Patches Florence2 cached remote code for transformers 5.x compatibility.

Florence2 uses trust_remote_code=True which downloads custom Python files from
HuggingFace. These files were written for transformers ~4.37 and break on 5.x.

Known issues fixed:
1. processing_florence2.py: tokenizer.additional_special_tokens not available on TokenizersBackend
2. configuration_florence2.py: self.forced_bos_token_id raises AttributeError on new PretrainedConfig
3. modeling_florence2.py: _supports_sdpa/_supports_flash_attn_2 properties access self.language_model
   before it's initialized in __init__
"""

import os


def _find_cached_dirs():
    """Find Florence2 cached remote code directories in HuggingFace cache."""
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "modules", "transformers_modules", "microsoft")
    dirs = {}
    if not os.path.isdir(cache_dir):
        return dirs
    for name in os.listdir(cache_dir):
        full = os.path.join(cache_dir, name)
        if not os.path.isdir(full):
            continue
        for sub in os.listdir(full):
            sub_path = os.path.join(full, sub)
            if os.path.isdir(sub_path) and sub != "__pycache__":
                dirs[name] = sub_path
    return dirs


def _patch_file(filepath, patches):
    """Apply text patches to a file if they haven't been applied yet."""
    if not os.path.isfile(filepath):
        return False
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    changed = False
    for old, new in patches:
        if old in content and new not in content:
            content = content.replace(old, new)
            changed = True
    if changed:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        pycache = os.path.join(os.path.dirname(filepath), "__pycache__")
        if os.path.isdir(pycache):
            import shutil
            shutil.rmtree(pycache, ignore_errors=True)
    return changed


def patch_florence2_remote_code():
    """Patch all known Florence2 remote code compatibility issues."""
    dirs = _find_cached_dirs()

    for dir_name, dir_path in dirs.items():
        # Fix 1: processing_florence2.py - additional_special_tokens
        _patch_file(os.path.join(dir_path, "processing_florence2.py"), [
            (
                "tokenizer.additional_special_tokens + \\",
                "(tokenizer.additional_special_tokens if hasattr(tokenizer, 'additional_special_tokens') else []) + \\"
            ),
        ])

        # Fix 2: configuration_florence2.py - forced_bos_token_id
        _patch_file(os.path.join(dir_path, "configuration_florence2.py"), [
            (
                "if self.forced_bos_token_id is None",
                "if getattr(self, 'forced_bos_token_id', None) is None"
            ),
        ])

        # Fix 3: modeling_florence2.py - _supports_sdpa and _supports_flash_attn_2
        # These are @property methods that access self.language_model, but
        # PreTrainedModel.__init__ in transformers 5.x accesses them BEFORE
        # Florence2's __init__ creates self.language_model. Fix by guarding
        # with hasattr so they return False during construction.
        #
        # Fix 4: modeling_florence2.py - torch.linspace().item() on meta tensors
        # PyTorch 2.8+ intercepts torch.linspace with device hooks that produce
        # meta tensors. .item() fails on meta tensors. Use .tolist() instead.
        _patch_file(os.path.join(dir_path, "modeling_florence2.py"), [
            (
                "return self.language_model._supports_flash_attn_2",
                "return getattr(self.language_model, '_supports_flash_attn_2', False) if hasattr(self, 'language_model') else False"
            ),
            (
                "return self.language_model._supports_sdpa",
                "return getattr(self.language_model, '_supports_sdpa', False) if hasattr(self, 'language_model') else False"
            ),
            (
                "dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths)*2)]",
                "_n = sum(depths) * 2; dpr = [float(i) * drop_path_rate / max(_n - 1, 1) for i in range(_n)]"
            ),
            (
                "dpr = torch.linspace(0, drop_path_rate, sum(depths)*2).cpu().tolist()",
                "_n = sum(depths) * 2; dpr = [float(i) * drop_path_rate / max(_n - 1, 1) for i in range(_n)]"
            ),
        ])
