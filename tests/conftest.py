"""Root test configuration.

Required secrets must exist before `rag_engine.config.settings` is imported, so
deterministic test values are injected into the environment at collection time.
"""

import os

TEST_ENV: dict[str, str] = {
    "ENVIRONMENT": "test",
    "ADMIN_API_KEY": "test-admin-key-0123456789abcdef0123456789",
    "API_KEY_PEPPER": "test-pepper-0123456789abcdef0123456789abcd",
    "OPENAI_API_KEY": "sk-test-not-a-real-key",
}

for _key, _value in TEST_ENV.items():
    os.environ.setdefault(_key, _value)
