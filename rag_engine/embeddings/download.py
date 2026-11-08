"""Image build step (Step 95): `python -m rag_engine.embeddings.download`.

Fetches the sparse model's files into `FASTEMBED_CACHE_DIR` so workers start with
`HF_HUB_OFFLINE=1` and never download at runtime. (The tiktoken encoding is baked the same
way via `TIKTOKEN_CACHE_DIR`.)
"""

from rag_engine.embeddings.sparse import download_model
from rag_engine.embeddings.tokenizer import tokenizer_for


def main() -> None:
    print(f"cached sparse model {download_model()}")
    print(f"cached tokenizer {tokenizer_for('text-embedding-3-small').name}")


if __name__ == "__main__":
    main()
