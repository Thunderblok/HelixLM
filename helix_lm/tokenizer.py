"""
HelixLM Tokenizer Abstraction.

Supports multiple backends:
  - character:  character-level (for smoke tests, tiny models)
  - gpt2:       GPT-2 BPE via transformers
  - qwen:       Qwen3 family tokenizer via transformers
  - lengthmax:  Experimental byte vocabulary with leftmost-longest matching
  - custom:     Any AutoTokenizer from HF

All backends expose a unified interface for encode / decode / vocab_size / pad_id / eos_id / bos_id.
"""
import hashlib
import json
import shutil
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import torch


_LENGTHMAX_PREFIX = "lengthmax:"
_LENGTHMAX_ALGORITHM = "iterative-byte-bpe-vocab-leftmost-longest-v0"
_LENGTHMAX_SPECIAL = "<|endoftext|>"
_LENGTHMAX_VOCAB_SIZE = 50_257


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _LengthMaxBackend:
    """Validated byte vocabulary decoded with deterministic longest matching."""

    def __init__(self, artifact_path: Path):
        if not artifact_path.is_absolute():
            raise ValueError(
                "LengthMAX tokenizer_name must use an absolute artifact path: "
                "lengthmax:/abs/path/to/iterative-hybrid-tokenizer.json"
            )
        if not artifact_path.is_file():
            raise FileNotFoundError(f"LengthMAX tokenizer artifact not found: {artifact_path}")

        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        if artifact.get("algorithm") != _LENGTHMAX_ALGORITHM:
            raise ValueError(f"Unsupported LengthMAX algorithm: {artifact.get('algorithm')!r}")
        if artifact.get("vocab_size") != _LENGTHMAX_VOCAB_SIZE:
            raise ValueError(f"LengthMAX vocab_size must be {_LENGTHMAX_VOCAB_SIZE}")
        if artifact.get("special_tokens") != [_LENGTHMAX_SPECIAL]:
            raise ValueError("LengthMAX special token contract mismatch")

        vocab = artifact.get("vocab")
        if not isinstance(vocab, dict) or len(vocab) != _LENGTHMAX_VOCAB_SIZE:
            raise ValueError("LengthMAX vocabulary must contain exactly 50,257 entries")
        if vocab.get(_LENGTHMAX_SPECIAL) != _LENGTHMAX_VOCAB_SIZE - 1:
            raise ValueError("LengthMAX end-of-text token must have id 50,256")

        id_to_bytes: List[Optional[bytes]] = [None] * (_LENGTHMAX_VOCAB_SIZE - 1)
        trie = [{"children": {}, "token_id": None}]
        max_token_len = 0
        for text, token_id in vocab.items():
            if text == _LENGTHMAX_SPECIAL:
                continue
            if not isinstance(text, str) or not isinstance(token_id, int):
                raise ValueError("LengthMAX vocabulary entries must map strings to integer ids")
            if not 0 <= token_id < _LENGTHMAX_VOCAB_SIZE - 1:
                raise ValueError(f"LengthMAX token id out of range: {token_id}")
            try:
                payload = text.encode("latin-1")
            except UnicodeEncodeError as exc:
                raise ValueError("LengthMAX vocabulary contains a non-byte symbol") from exc
            if not payload:
                raise ValueError("LengthMAX vocabulary contains an empty token")
            if id_to_bytes[token_id] is not None:
                raise ValueError(f"Duplicate LengthMAX token id: {token_id}")
            id_to_bytes[token_id] = payload
            max_token_len = max(max_token_len, len(payload))

            node = 0
            for value in payload:
                children = trie[node]["children"]
                node = children.setdefault(value, len(trie))
                if node == len(trie):
                    trie.append({"children": {}, "token_id": None})
            if trie[node]["token_id"] is not None:
                raise ValueError("Duplicate LengthMAX token bytes")
            trie[node]["token_id"] = token_id

        if any(payload is None for payload in id_to_bytes):
            raise ValueError("LengthMAX token ids must be contiguous from 0 through 50,255")
        for value in range(256):
            if id_to_bytes[value] != bytes([value]):
                raise ValueError("LengthMAX ids 0 through 255 must be the exact byte alphabet")
        if artifact.get("max_token_len") != max_token_len:
            raise ValueError("LengthMAX max_token_len does not match the vocabulary")

        self.artifact_path = artifact_path
        self.artifact_sha256 = _sha256_file(artifact_path)
        self.id_to_bytes = id_to_bytes
        self.trie = trie
        self.max_token_len = max_token_len

    def encode(self, text: str) -> List[int]:
        payload = text.encode("utf-8")
        token_ids: List[int] = []
        position = 0
        while position < len(payload):
            node = 0
            cursor = position
            best_id = None
            best_end = position
            while cursor < min(len(payload), position + self.max_token_len):
                child = self.trie[node]["children"].get(payload[cursor])
                if child is None:
                    break
                node = child
                cursor += 1
                candidate = self.trie[node]["token_id"]
                if candidate is not None:
                    best_id = candidate
                    best_end = cursor
            if best_id is None:
                raise RuntimeError(f"LengthMAX byte fallback missing at offset {position}")
            token_ids.append(best_id)
            position = best_end
        return token_ids

    def decode(self, ids: List[int], *, skip_special_tokens: bool, errors: str = "replace") -> str:
        payload = bytearray()
        for raw_id in ids:
            token_id = int(raw_id)
            if token_id == _LENGTHMAX_VOCAB_SIZE - 1:
                if skip_special_tokens:
                    continue
                payload.extend(_LENGTHMAX_SPECIAL.encode("utf-8"))
                continue
            if not 0 <= token_id < len(self.id_to_bytes):
                raise ValueError(f"LengthMAX token id out of range: {token_id}")
            payload.extend(self.id_to_bytes[token_id])
        return bytes(payload).decode("utf-8", errors=errors)


class HelixTokenizer:
    """
    Unified tokenizer wrapper for HelixLM.
    Automatically loads the correct backend based on tokenizer_name.
    """
    def __init__(self, tokenizer_name: str = "gpt2", **kwargs):
        self.tokenizer_name = tokenizer_name
        self._backend = None
        self._lengthmax_artifact_path: Optional[Path] = None
        self._char_to_id: Optional[Dict[str, int]] = None
        self._id_to_char: Optional[Dict[int, str]] = None

        if tokenizer_name == "char":
            # Character-level: must call build_vocab(texts) before use
            pass
        elif tokenizer_name.startswith(_LENGTHMAX_PREFIX):
            artifact_path = Path(tokenizer_name[len(_LENGTHMAX_PREFIX):])
            self._backend = _LengthMaxBackend(artifact_path)
            self._lengthmax_artifact_path = artifact_path
        elif tokenizer_name.startswith("gpt2") or tokenizer_name.startswith("openai"):
            from transformers import AutoTokenizer
            self._backend = AutoTokenizer.from_pretrained("gpt2", **kwargs)
            self._backend.pad_token = self._backend.eos_token
        elif tokenizer_name.startswith("qwen"):
            from transformers import AutoTokenizer
            self._backend = AutoTokenizer.from_pretrained(
                tokenizer_name if "/" in tokenizer_name else "Qwen/Qwen2.5-0.5B",
                trust_remote_code=True,
                **kwargs,
            )
            if self._backend.pad_token is None:
                self._backend.pad_token = self._backend.eos_token
        else:
            # Custom HF tokenizer
            from transformers import AutoTokenizer
            self._backend = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True, **kwargs)
            if self._backend.pad_token is None:
                self._backend.pad_token = self._backend.eos_token

    # ------------------------------------------------------------------
    # Character-level vocab builder
    # ------------------------------------------------------------------
    def build_char_vocab(self, texts: Union[str, List[str]], special_tokens: Optional[List[str]] = None):
        """
        Build character vocabulary from text(s).
        Must be called before encode/decode when tokenizer_name == 'char'.
        """
        if isinstance(texts, str):
            texts = [texts]
        all_chars = set()
        for t in texts:
            all_chars.update(t)
        chars = sorted(all_chars)

        # Reserve 0 for pad, 1 for eos, 2 for bos
        offset = 3
        if special_tokens:
            for i, tok in enumerate(special_tokens):
                chars = [tok] + chars if tok not in chars else chars

        self._char_to_id = {c: i + offset for i, c in enumerate(chars)}
        self._id_to_char = {i + offset: c for i, c in enumerate(chars)}
        self._char_to_id["<pad>"] = 0
        self._char_to_id["<eos>"] = 1
        self._char_to_id["<bos>"] = 2
        self._id_to_char[0] = "<pad>"
        self._id_to_char[1] = "<eos>"
        self._id_to_char[2] = "<bos>"

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------
    @property
    def _is_lengthmax(self) -> bool:
        return self.tokenizer_name.startswith(_LENGTHMAX_PREFIX)

    def encode(self, text: str, add_special_tokens: bool = False, **kwargs) -> List[int]:
        if self.tokenizer_name == "char":
            if self._char_to_id is None:
                raise RuntimeError("Call build_char_vocab() before encode()")
            ids = [self._char_to_id.get(c, 0) for c in text]
            if add_special_tokens:
                ids = [2] + ids + [1]
            return ids
        if self._is_lengthmax:
            if kwargs:
                raise TypeError(f"Unsupported LengthMAX encode options: {sorted(kwargs)}")
            # GPT-2 does not add BOS/EOS through encode(add_special_tokens=True).
            return self._backend.encode(text)
        return self._backend.encode(text, add_special_tokens=add_special_tokens, **kwargs)

    def decode(self, ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True, **kwargs) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if self.tokenizer_name == "char":
            if self._id_to_char is None:
                raise RuntimeError("Call build_char_vocab() before decode()")
            return "".join(self._id_to_char.get(i, "") for i in ids if not (skip_special_tokens and i in (0, 1, 2)))
        if self._is_lengthmax:
            errors = kwargs.pop("errors", "replace")
            if kwargs:
                raise TypeError(f"Unsupported LengthMAX decode options: {sorted(kwargs)}")
            return self._backend.decode(ids, skip_special_tokens=skip_special_tokens, errors=errors)
        return self._backend.decode(ids, skip_special_tokens=skip_special_tokens, **kwargs)

    def __call__(self, text: Union[str, List[str]], return_tensors: Optional[str] = None, padding: bool = False,
                 truncation: bool = False, max_length: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """Batch tokenization returning dict with input_ids, attention_mask."""
        if self.tokenizer_name == "char" or self._is_lengthmax:
            if isinstance(text, str):
                text = [text]
            input_ids = [self.encode(t) for t in text]
            max_len = max(len(ids) for ids in input_ids) if not max_length else max_length
            attention_mask = []
            padded_ids = []
            for ids in input_ids:
                if truncation and max_length and len(ids) > max_length:
                    ids = ids[:max_length]
                mask = [1] * len(ids) + [0] * (max_len - len(ids))
                ids = ids + [0] * (max_len - len(ids))
                padded_ids.append(ids)
                attention_mask.append(mask)
            result = {"input_ids": padded_ids, "attention_mask": attention_mask}
            if return_tensors == "pt":
                result["input_ids"] = torch.tensor(result["input_ids"], dtype=torch.long)
                result["attention_mask"] = torch.tensor(result["attention_mask"], dtype=torch.long)
            return result
        return self._backend(text, return_tensors=return_tensors, padding=padding, truncation=truncation,
                             max_length=max_length, **kwargs)

    def batch_encode(self, texts: List[str], **kwargs) -> Dict[str, Any]:
        """Alias for __call__ with list input."""
        return self(texts, **kwargs)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        if self.tokenizer_name == "char":
            return len(self._char_to_id) if self._char_to_id else 0
        if self._is_lengthmax:
            return _LENGTHMAX_VOCAB_SIZE
        return len(self._backend)

    def __len__(self) -> int:
        return self.vocab_size

    @property
    def pad_token_id(self) -> int:
        if self.tokenizer_name == "char":
            return 0
        if self._is_lengthmax:
            return _LENGTHMAX_VOCAB_SIZE - 1
        return self._backend.pad_token_id

    @property
    def eos_token_id(self) -> int:
        if self.tokenizer_name == "char":
            return 1
        if self._is_lengthmax:
            return _LENGTHMAX_VOCAB_SIZE - 1
        return self._backend.eos_token_id

    @property
    def bos_token_id(self) -> int:
        if self.tokenizer_name == "char":
            return 2
        if self._is_lengthmax:
            return _LENGTHMAX_VOCAB_SIZE - 1
        return getattr(self._backend, "bos_token_id", self.eos_token_id)

    @property
    def unk_token_id(self) -> int:
        if self.tokenizer_name == "char":
            return 0
        if self._is_lengthmax:
            return _LENGTHMAX_VOCAB_SIZE - 1
        return getattr(self._backend, "unk_token_id", self.pad_token_id)

    @property
    def pad_token(self) -> str:
        if self.tokenizer_name == "char":
            return "<pad>"
        if self._is_lengthmax:
            return _LENGTHMAX_SPECIAL
        return self._backend.pad_token

    @property
    def eos_token(self) -> str:
        if self.tokenizer_name == "char":
            return "<eos>"
        if self._is_lengthmax:
            return _LENGTHMAX_SPECIAL
        return self._backend.eos_token

    @property
    def bos_token(self) -> str:
        if self.tokenizer_name == "char":
            return "<bos>"
        if self._is_lengthmax:
            return _LENGTHMAX_SPECIAL
        return getattr(self._backend, "bos_token", self.eos_token)

    @property
    def special_tokens_map(self) -> Dict[str, Any]:
        if self.tokenizer_name == "char":
            return {"pad_token": "<pad>", "eos_token": "<eos>", "bos_token": "<bos>"}
        if self._is_lengthmax:
            return {
                "bos_token": _LENGTHMAX_SPECIAL,
                "eos_token": _LENGTHMAX_SPECIAL,
                "unk_token": _LENGTHMAX_SPECIAL,
                "pad_token": _LENGTHMAX_SPECIAL,
            }
        return self._backend.special_tokens_map

    def apply_chat_template(self, messages: List[Dict[str, Any]], tokenize: bool = True,
                            add_generation_prompt: bool = False, return_dict: bool = False,
                            return_tensors: Optional[str] = None, **kwargs) -> Any:
        """
        Apply chat template to messages.
        Falls back to manual formatting for char tokenizer or missing template.
        """
        if self.tokenizer_name == "char" or self._is_lengthmax:
            # Simple manual format
            formatted = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if isinstance(content, list):
                    text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
                    content = " ".join(text_parts)
                formatted += f"[{role}]: {content}\n"
            if add_generation_prompt:
                formatted += "[assistant]: "
            if not tokenize:
                return formatted
            result = self(formatted, return_tensors=return_tensors)
            if return_dict:
                return result
            return result["input_ids"]

        if hasattr(self._backend, "apply_chat_template") and self._backend.chat_template is not None:
            return self._backend.apply_chat_template(
                messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt,
                return_dict=return_dict, return_tensors=return_tensors, **kwargs,
            )

        # Fallback: manual Qwen-style format
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
                content = " ".join(text_parts)
            if role == "system":
                formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        if add_generation_prompt:
            formatted += "<|im_start|>assistant\n"

        if not tokenize:
            return formatted
        result = self._backend(formatted, return_tensors=return_tensors, padding=True, truncation=True)
        if return_dict:
            return result
        return result["input_ids"]

    def save_pretrained(self, path):
        if self._is_lengthmax:
            output_path = Path(path)
            output_path.mkdir(parents=True, exist_ok=True)
            artifact_path = output_path / "lengthmax-tokenizer.json"
            shutil.copy2(self._lengthmax_artifact_path, artifact_path)
            artifact_sha256 = _sha256_file(artifact_path)
            config = {
                "schema": "helix.lengthmax-tokenizer-checkpoint.v0",
                "tokenizer_class": "HelixTokenizer",
                "tokenizer_name": f"{_LENGTHMAX_PREFIX}{artifact_path.name}",
                "artifact_file": artifact_path.name,
                "artifact_sha256": artifact_sha256,
                "algorithm": _LENGTHMAX_ALGORITHM,
                "vocab_size": self.vocab_size,
                "pad_token_id": self.pad_token_id,
                "eos_token_id": self.eos_token_id,
                "bos_token_id": self.bos_token_id,
                "unk_token_id": self.unk_token_id,
                "special_tokens_map": self.special_tokens_map,
            }
            (output_path / "helix_tokenizer_config.json").write_text(
                json.dumps(config, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return (str(output_path),)
        return self._backend.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, path):
        input_path = Path(path)
        config_path = input_path / "helix_tokenizer_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("schema") != "helix.lengthmax-tokenizer-checkpoint.v0":
            raise ValueError("Unsupported Helix tokenizer checkpoint schema")
        artifact_path = input_path / config["artifact_file"]
        actual_sha256 = _sha256_file(artifact_path)
        if actual_sha256 != config.get("artifact_sha256"):
            raise ValueError(
                "LengthMAX tokenizer artifact hash mismatch: "
                f"expected {config.get('artifact_sha256')}, observed {actual_sha256}"
            )
        tokenizer = cls(f"{_LENGTHMAX_PREFIX}{artifact_path.resolve()}")
        expected = {
            "algorithm": _LENGTHMAX_ALGORITHM,
            "vocab_size": tokenizer.vocab_size,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "unk_token_id": tokenizer.unk_token_id,
            "special_tokens_map": tokenizer.special_tokens_map,
        }
        for key, observed in expected.items():
            if config.get(key) != observed:
                raise ValueError(f"LengthMAX checkpoint {key} mismatch")
        return tokenizer

    def push_to_hub(self, repo_id, **kwargs):
        if self._is_lengthmax:
            raise NotImplementedError(
                "Experimental LengthMAX tokenizer checkpoints must be uploaded with their model directory."
            )
        return self._backend.push_to_hub(repo_id, **kwargs)
