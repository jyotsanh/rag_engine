"""URL normalization: explicit rules plus Hypothesis properties."""

import random
from urllib.parse import quote, urlsplit

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from rag_engine.crawling import UrlNormalizationError, normalize_and_hash, normalize_url, url_hash
from rag_engine.domain import Source, Tenant

TEN = Tenant(name="t").id

# ---- strategies ---- #

_label = st.from_regex(r"[a-zA-Z0-9]([a-zA-Z0-9-]{0,10}[a-zA-Z0-9])?", fullmatch=True)
_unicode_label = st.text(alphabet="abcdéüßñ漢字", min_size=1, max_size=8)
_host = st.lists(st.one_of(_label, _unicode_label), min_size=1, max_size=3).map(
    lambda labels: ".".join(labels) + ".example"
)
_segment = st.text(
    alphabet=st.characters(blacklist_categories=("Cs", "Cc"), blacklist_characters="/?#%\\"),
    max_size=8,
)
_path = st.lists(st.one_of(_segment, st.sampled_from([".", ".."])), max_size=5).map(
    lambda segs: "/" + "/".join(segs)
)
_qtext = st.text(alphabet="abcXYZ019 -_.~!$'()*,;:@/é", max_size=6)
_query = st.lists(st.tuples(_qtext.filter(bool), _qtext), max_size=4)
_url = st.builds(
    lambda scheme, host, port, path, query, frag: (
        f"{scheme}://{host}{port}{path}"
        + ("?" + "&".join(f"{quote(k)}={quote(v)}" for k, v in query) if query else "")
        + (f"#{frag}" if frag else "")
    ),
    st.sampled_from(["http", "https", "HTTP", "HttPs"]),
    _host,
    st.sampled_from(["", ":80", ":443", ":8080"]),
    _path,
    _query,
    st.text(alphabet="abc", max_size=3),
)

_PROPERTY = settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])


# ---- #
# Tests
# ---- #


class TestUrlNormalization_S23:
    """S23 — URL normalization: canonical form, tracking params, url_hash."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("HTTPS://Acme.Example/Docs", "https://acme.example/Docs"),
            ("https://acme.example", "https://acme.example/"),
            ("https://acme.example:443/a", "https://acme.example/a"),
            ("http://acme.example:80/a", "http://acme.example/a"),
            ("https://acme.example:8443/a", "https://acme.example:8443/a"),
            ("http://acme.example:443/a", "http://acme.example:443/a"),
            ("https://acme.example./a", "https://acme.example/a"),
            ("https://acme.example/a/./b/../c", "https://acme.example/a/c"),
            ("https://acme.example/../../x", "https://acme.example/x"),
            ("https://acme.example/a/b/..", "https://acme.example/a/"),
            ("https://acme.example/docs/", "https://acme.example/docs/"),
            ("https://acme.example/a//b", "https://acme.example/a//b"),
            ("https://acme.example/a#section", "https://acme.example/a"),
            ("https://acme.example/%7euser", "https://acme.example/~user"),
            ("https://acme.example/%2f%3a", "https://acme.example/%2F%3A"),
            ("https://acme.example/a b", "https://acme.example/a%20b"),
            ("https://acme.example/€", "https://acme.example/%E2%82%AC"),
            ("https://Bücher.Example/", "https://xn--bcher-kva.example/"),
            ("http://[2001:DB8::1]:8080/x", "http://[2001:db8::1]:8080/x"),
            ("http://192.168.1.1/x", "http://192.168.1.1/x"),
            ("  https://acme.example/a\n\tb  ", "https://acme.example/ab"),
        ],
    )
    def test_canonical_forms(self, raw: str, expected: str) -> None:
        assert normalize_url(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("https://a.example/?b=2&a=1", "https://a.example/?a=1&b=2"),
            ("https://a.example/?a=2&a=1", "https://a.example/?a=1&a=2"),
            ("https://a.example/?utm_source=x&utm_medium=y", "https://a.example/"),
            ("https://a.example/?gclid=1&fbclid=2&MSCLKID=3&id=7", "https://a.example/?id=7"),
            ("https://a.example/?pk_campaign=1&mtm_kwd=2&q=w", "https://a.example/?q=w"),
            ("https://a.example/?ref=home", "https://a.example/?ref=home"),
            ("https://a.example/?empty=&q=1", "https://a.example/?empty=&q=1"),
            ("https://a.example/?q=a+b", "https://a.example/?q=a%20b"),
            ("https://a.example/?q=a%2Bb", "https://a.example/?q=a%2Bb"),
            ("https://a.example/?q=x%26y", "https://a.example/?q=x%26y"),
            ("https://a.example/?", "https://a.example/"),
        ],
    )
    def test_query_rules(self, raw: str, expected: str) -> None:
        assert normalize_url(raw) == expected

    def test_fixture_site_tracking_link_keeps_only_ref(self) -> None:
        tracked = "/docs/getting-started.html?utm_source=newsletter&utm_medium=email&ref=home"

        assert normalize_url(tracked, base="http://127.0.0.1:8000/") == (
            "http://127.0.0.1:8000/docs/getting-started.html?ref=home"
        )

    @pytest.mark.parametrize(
        ("relative", "expected"),
        [
            ("c", "https://acme.example/a/b/c"),
            ("../c", "https://acme.example/a/c"),
            ("/root", "https://acme.example/root"),
            ("?q=1", "https://acme.example/a/b/page?q=1"),
            ("//other.example/x", "https://other.example/x"),
            ("#frag", "https://acme.example/a/b/page"),
            ("https://abs.example/", "https://abs.example/"),
        ],
    )
    def test_resolves_against_base(self, relative: str, expected: str) -> None:
        assert normalize_url(relative, base="https://acme.example/a/b/page") == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "",
            "   ",
            "/relative/only",
            "ftp://acme.example/",
            "mailto:support@acme.example",
            "javascript:void(0)",
            "tel:+15550100",
            "https://user:pw@acme.example/",
            "https://user@acme.example/",
            "https:///no-host",
            "https://acme.example:99999/",
            "https://acme.example:abc/",
            "https://bad_host!.example/",
            "http://[::1/",
        ],
    )
    def test_rejects_uncrawlable(self, raw: str) -> None:
        with pytest.raises(UrlNormalizationError):
            normalize_url(raw)

    def test_hash_is_sha256_of_normalized_form(self) -> None:
        normalized, digest = normalize_and_hash("HTTPS://Acme.Example/a?b=1&a=2#x")

        assert normalized == "https://acme.example/a?a=2&b=1"
        assert digest == url_hash(normalized)
        assert len(digest) == 64
        assert digest == normalize_and_hash("https://acme.example:443/a?utm_x=1&a=2&b=1")[1]

    @_PROPERTY
    @given(_url)
    def test_idempotent(self, url: str) -> None:
        try:
            once = normalize_url(url)
        except UrlNormalizationError:
            return  # some generated hosts are not valid IDNA; rejection is fine
        assert normalize_url(once) == once

    @_PROPERTY
    @given(_url)
    def test_output_is_a_valid_storable_url(self, url: str) -> None:
        try:
            normalized = normalize_url(url)
        except UrlNormalizationError:
            return
        parts = urlsplit(normalized)

        assert parts.scheme in {"http", "https"}
        assert parts.hostname and parts.hostname == parts.hostname.lower()
        assert normalized.isascii()
        assert not parts.fragment
        assert parts.path.startswith("/")
        # every normalized URL is acceptable to the domain model
        Source(tenant_id=TEN, kb_id="kb_" + "0" * 32, seed_url=normalized)

    @_PROPERTY
    @given(_url, st.randoms(use_true_random=False))
    def test_invariant_under_noise(self, url: str, rnd: random.Random) -> None:
        try:
            base = normalize_url(url)
        except UrlNormalizationError:
            return
        parts = urlsplit(base)
        pairs = [p for p in parts.query.split("&") if p]
        rnd.shuffle(pairs)
        tracking = ["utm_source=news", "gclid=abc", "fbclid=x"]
        noisy_query = "&".join(pairs + tracking)
        host = parts.netloc.upper()
        noisy = f"{parts.scheme.upper()}://{host}{parts.path}?{noisy_query}#fragment"

        assert normalize_url(noisy) == base
