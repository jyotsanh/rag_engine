"""Webhook signing (service side) and verification (receiver side, stdlib only), plus the
`SecretBox` that keeps tenants' webhook secrets encrypted at rest."""

import ast
import sys
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import SecretStr

from rag_engine.config import Settings
from rag_engine.security import SecretBox, SecretUnreadableError
from rag_engine.webhooks import (
    SECRET_PREFIX,
    SIGNATURE_HEADER,
    MalformedSignatureError,
    SignatureMismatchError,
    TimestampOutOfToleranceError,
    WebhookVerificationError,
    display_prefix,
    generate_secret,
    sign,
    signature_headers,
    verify_webhook,
)
from rag_engine.webhooks import verify as verify_module

_BODY = b'{"event":"job.succeeded"}'
_T = 1731150000
# hex(HMAC-SHA256(b"whsec_test", b"1731150000." + _BODY)), computed independently
_KNOWN = "fa7d33074bfdc42463fb87357c515065c423c8a31f269ba5648352aa25e9da92"


def _box(*keys: str) -> SecretBox:
    return SecretBox([SecretStr(k) for k in keys])


_KEY_A = "a" * 32 + "-key-one"
_KEY_B = "b" * 32 + "-key-two"


# ---- #
# Tests
# ---- #


class TestWebhookSigner_S82:
    """S82 — HMAC-SHA256 over `"{t}." + body`, `X-RagEngine-Signature: t=…,v1=…`, and the
    receiver's verification helper."""

    # ---- signing ---- #

    def test_known_vector(self) -> None:
        assert sign(_BODY, ["whsec_test"], timestamp=_T) == f"t={_T},v1={_KNOWN}"

    def test_one_signature_per_active_secret_current_first(self) -> None:
        header = sign(_BODY, ["whsec_new", "whsec_old"], timestamp=_T)

        parts = header.split(",")
        assert parts[0] == f"t={_T}" and len(parts) == 3
        assert all(p.startswith("v1=") and len(p) == 3 + 64 for p in parts[1:])
        assert parts[1] != parts[2]

    def test_signature_headers_uses_the_current_time(self) -> None:
        headers = signature_headers(_BODY, ["whsec_test"], now=_T + 0.9)

        assert headers == {SIGNATURE_HEADER: f"t={_T},v1={_KNOWN}"}
        assert SIGNATURE_HEADER == "X-RagEngine-Signature"

    @pytest.mark.parametrize(
        ("secrets", "timestamp"),
        [([], _T), ([""], _T), (["s"] * 9, _T), (["s"], -1)],
    )
    def test_invalid_signing_input_is_refused(self, secrets: list[str], timestamp: int) -> None:
        with pytest.raises(ValueError):
            sign(_BODY, secrets, timestamp=timestamp)

    def test_generated_secrets(self) -> None:
        secrets = {generate_secret() for _ in range(100)}

        assert len(secrets) == 100
        for secret in secrets:
            assert secret.startswith(SECRET_PREFIX) and len(secret) == len(SECRET_PREFIX) + 43
            assert display_prefix(secret) == secret[:12]

    # ---- verification ---- #

    def test_a_fresh_signed_request_verifies(self) -> None:
        header = sign(_BODY, ["whsec_test"], timestamp=_T)

        assert verify_webhook(_BODY, header, "whsec_test", now=_T + 10) == _T

    def test_during_a_rotation_either_secret_verifies(self) -> None:
        header = sign(_BODY, ["whsec_new", "whsec_old"], timestamp=_T)

        assert verify_webhook(_BODY, header, "whsec_old", now=_T)  # receiver not switched yet
        assert verify_webhook(_BODY, header, "whsec_new", now=_T)  # receiver switched
        # a receiver rotating its own config accepts several secrets
        single = sign(_BODY, ["whsec_new"], timestamp=_T)
        assert verify_webhook(_BODY, single, ["whsec_old", "whsec_new"], now=_T)

    def test_a_tampered_body_or_wrong_secret_is_rejected(self) -> None:
        header = sign(_BODY, ["whsec_test"], timestamp=_T)

        with pytest.raises(SignatureMismatchError):
            verify_webhook(_BODY + b" ", header, "whsec_test", now=_T)
        with pytest.raises(SignatureMismatchError):
            verify_webhook(_BODY, header, "whsec_other", now=_T)
        # the timestamp is signed too: moving it breaks the signature
        moved = header.replace(f"t={_T}", f"t={_T + 1}")
        with pytest.raises(SignatureMismatchError):
            verify_webhook(_BODY, moved, "whsec_test", now=_T)

    @pytest.mark.parametrize(("offset", "ok"), [(300, True), (301, False), (-300, True),
                                                (-301, False)])  # fmt: skip
    def test_timestamp_tolerance_both_ways(self, offset: int, ok: bool) -> None:
        header = sign(_BODY, ["whsec_test"], timestamp=_T)

        if ok:
            assert verify_webhook(_BODY, header, "whsec_test", now=_T + offset) == _T
        else:
            with pytest.raises(TimestampOutOfToleranceError):
                verify_webhook(_BODY, header, "whsec_test", now=_T + offset)

    def test_a_custom_tolerance(self) -> None:
        header = sign(_BODY, ["whsec_test"], timestamp=_T)

        with pytest.raises(TimestampOutOfToleranceError):
            verify_webhook(_BODY, header, "whsec_test", now=_T + 61, tolerance_seconds=60)

    @pytest.mark.parametrize(
        "header",
        [
            None,
            "",
            f"v1={_KNOWN}",  # no timestamp
            f"t={_T}",  # no signature
            f"t={_T},v2={_KNOWN}",  # only an unknown scheme
            f"t=abc,v1={_KNOWN}",
            f"t=-5,v1={_KNOWN}",
            f"t={_T},t={_T},v1={_KNOWN}",
            f"t={_T},v1",
            f"t={_T},v1=",
            f"t={_T}," + ",".join([f"v1={_KNOWN}"] * 9),
            f"t={_T},v1=" + "a" * 2000,
        ],
    )
    def test_malformed_headers_are_rejected(self, header: str | None) -> None:
        with pytest.raises(MalformedSignatureError):
            verify_webhook(_BODY, header, "whsec_test", now=_T)

    def test_unknown_schemes_are_ignored_and_case_and_spaces_tolerated(self) -> None:
        header = f"t={_T}, v2=deadbeef, v1={_KNOWN.upper()}"

        assert verify_webhook(_BODY, header, "whsec_test", now=_T) == _T

    def test_every_failure_is_a_verification_error(self) -> None:
        for error in (MalformedSignatureError, TimestampOutOfToleranceError,
                      SignatureMismatchError):  # fmt: skip
            assert issubclass(error, WebhookVerificationError)
        with pytest.raises(ValueError):
            verify_webhook(_BODY, f"t={_T},v1={_KNOWN}", [], now=_T)

    def test_the_receiver_helper_uses_only_the_standard_library(self) -> None:
        tree = ast.parse(Path(verify_module.__file__).read_text())
        imported = {
            ((node.module or "") if isinstance(node, ast.ImportFrom) else alias.name).split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }

        assert imported and imported <= set(sys.stdlib_module_names)

    @given(body=st.binary(max_size=2048), secret=st.text(min_size=1, max_size=64),
           timestamp=st.integers(min_value=0, max_value=2**40))  # fmt: skip
    def test_property_round_trip(self, body: bytes, secret: str, timestamp: int) -> None:
        header = sign(body, [secret], timestamp=timestamp)

        assert verify_webhook(body, header, secret, now=timestamp) == timestamp

    @given(body=st.binary(min_size=1, max_size=512), index=st.integers(min_value=0))
    def test_property_any_flipped_byte_fails(self, body: bytes, index: int) -> None:
        header = sign(body, ["whsec_test"], timestamp=_T)
        position = index % len(body)
        tampered = body[:position] + bytes([body[position] ^ 0x01]) + body[position + 1 :]

        with pytest.raises(SignatureMismatchError):
            verify_webhook(tampered, header, "whsec_test", now=_T)


class TestSecretBox_S82:
    """S82 — `SecretBox`: AES-256-GCM at rest, bound to its record, with key rotation."""

    def test_round_trip_and_format(self) -> None:
        box = _box(_KEY_A)

        token = box.seal("whsec_secret", context="ten_1")

        assert token.startswith("sb1.") and len(token.split(".")) == 3
        assert "whsec_secret" not in token
        assert box.open(token, context="ten_1") == "whsec_secret"

    def test_every_seal_uses_a_fresh_nonce(self) -> None:
        box = _box(_KEY_A)

        assert box.seal("same", context="c") != box.seal("same", context="c")

    def test_a_sealed_value_is_bound_to_its_record(self) -> None:
        box = _box(_KEY_A)
        token = box.seal("whsec_secret", context="ten_1")

        with pytest.raises(SecretUnreadableError):
            box.open(token, context="ten_2")  # copied onto another tenant

    def test_tampered_or_malformed_tokens_are_refused(self) -> None:
        box = _box(_KEY_A)
        token = box.seal("whsec_secret", context="c")
        version, key_id, payload = token.split(".")
        flipped = payload[:-2] + ("A" if payload[-2] != "A" else "B") + payload[-1]

        for bad in (f"{version}.{key_id}.{flipped}", "garbage", f"sb9.{key_id}.{payload}",
                    f"{version}.{key_id}.!!!"):  # fmt: skip
            with pytest.raises(SecretUnreadableError):
                box.open(bad, context="c")

    def test_a_retired_key_still_opens_and_asks_for_a_reseal(self) -> None:
        old_token = _box(_KEY_A).seal("whsec_secret", context="c")
        rotated = _box(_KEY_B, _KEY_A)  # new key current, old one retired

        assert rotated.open(old_token, context="c") == "whsec_secret"
        assert rotated.needs_reseal(old_token) is True
        new_token = rotated.seal("whsec_secret", context="c")
        assert rotated.needs_reseal(new_token) is False
        with pytest.raises(SecretUnreadableError):
            _box(_KEY_B).open(old_token, context="c")  # the old key dropped entirely

    def test_settings_require_strong_keys(self) -> None:
        box = SecretBox.from_settings(Settings(_env_file=None))
        assert box.open(box.seal("x", context="c"), context="c") == "x"

        with pytest.raises(ValueError, match="SECRET_ENCRYPTION_KEY"):
            Settings(_env_file=None, SECRET_ENCRYPTION_KEY="short")
        with pytest.raises(ValueError, match="SECRET_ENCRYPTION_KEYS_PREVIOUS"):
            Settings(_env_file=None, SECRET_ENCRYPTION_KEYS_PREVIOUS=["short"])
        with pytest.raises(ValueError):
            SecretBox([])
