"""Tests for urc.validate — manifest capability detection."""

import textwrap

import pytest

from urc.validate import validate_manifest_capabilities


# ── Helper to build minimal manifests ──

def _minimal_manifest(streams_yaml: str = "") -> str:
    """Return a valid minimal manifest with optional stream content."""
    return textwrap.dedent(f"""\
        type: DeclarativeSource
        check:
          type: CheckStream
          stream_names: ["events"]
        streams:
          - type: DeclarativeStream
            name: events
            retriever:
              type: SimpleRetriever
              requester:
                type: HttpRequester
                url_base: "https://api.example.com"
                path: "/events"
              record_selector:
                type: RecordSelector
                extractor:
                  type: DpathExtractor
                  field_path: ["data"]
        {textwrap.indent(streams_yaml, '    ')}
    """)


# ── Unknown type → error ──

class TestUnknownType:
    def test_unknown_type_produces_error(self):
        manifest = textwrap.dedent("""\
            type: DeclarativeSource
            streams:
              - type: DeclarativeStream
                name: events
                retriever:
                  type: SimpleRetriever
                  requester:
                    type: HttpRequester
                    url_base: "https://api.example.com"
                    path: "/events"
                  record_selector:
                    type: RecordSelector
                    extractor:
                      type: DpathExtractor
                      field_path: ["data"]
                  paginator:
                    type: TotallyMadeUpPaginator
        """)
        issues = validate_manifest_capabilities(manifest)
        errors = [i for i in issues if i["severity"] == "error"]
        assert len(errors) >= 1
        bogus = [e for e in errors if e["type"] == "TotallyMadeUpPaginator"]
        assert len(bogus) == 1
        assert "Unknown component type" in bogus[0]["message"]

    def test_unknown_type_at_check_level(self):
        manifest = textwrap.dedent("""\
            type: DeclarativeSource
            check:
              type: CheckStream
              stream_names: ["events"]
            streams: []
        """)
        issues = validate_manifest_capabilities(manifest)
        # CheckStream is not in REGISTRY → error
        errors = [i for i in issues if i["severity"] == "error"]
        assert any(i["type"] == "CheckStream" for i in errors)


# ── JWT with RSA/EC algorithm → warning ──

class TestJwtWarning:
    @pytest.mark.parametrize("algorithm", ["RS256", "RS384", "RS512", "ES256", "ES384", "ES512"])
    def test_jwt_rsa_ec_algorithm_warns(self, algorithm):
        manifest = textwrap.dedent(f"""\
            type: DeclarativeSource
            streams:
              - type: DeclarativeStream
                name: events
                retriever:
                  type: SimpleRetriever
                  requester:
                    type: HttpRequester
                    url_base: "https://api.example.com"
                    path: "/events"
                    authenticator:
                      type: JwtAuthenticator
                      secret_key: "secret"
                      algorithm: {algorithm}
                      token_duration: 3600
                  record_selector:
                    type: RecordSelector
                    extractor:
                      type: DpathExtractor
                      field_path: ["data"]
        """)
        issues = validate_manifest_capabilities(manifest)
        warnings = [i for i in issues if i["severity"] == "warning"]
        assert len(warnings) >= 1
        assert any(algorithm in w["message"] and "cryptography" in w["message"] for w in warnings)

    def test_jwt_hs256_no_warning(self):
        manifest = textwrap.dedent("""\
            type: DeclarativeSource
            streams:
              - type: DeclarativeStream
                name: events
                retriever:
                  type: SimpleRetriever
                  requester:
                    type: HttpRequester
                    url_base: "https://api.example.com"
                    path: "/events"
                    authenticator:
                      type: JwtAuthenticator
                      secret_key: "secret"
                      algorithm: HS256
                      token_duration: 3600
                  record_selector:
                    type: RecordSelector
                    extractor:
                      type: DpathExtractor
                      field_path: ["data"]
        """)
        issues = validate_manifest_capabilities(manifest)
        warnings = [i for i in issues if i["severity"] == "warning"]
        assert len(warnings) == 0


# ── OAuth2 PKCE → warning ──

class TestOAuthPkce:
    def test_oauth_with_code_challenge_method_warns(self):
        manifest = textwrap.dedent("""\
            type: DeclarativeSource
            streams:
              - type: DeclarativeStream
                name: events
                retriever:
                  type: SimpleRetriever
                  requester:
                    type: HttpRequester
                    url_base: "https://api.example.com"
                    path: "/events"
                    authenticator:
                      type: OAuthAuthenticator
                      client_id: "id"
                      client_secret: "secret"
                      token_refresh_endpoint: "https://auth.example.com/token"
                      code_challenge_method: S256
                  record_selector:
                    type: RecordSelector
                    extractor:
                      type: DpathExtractor
                      field_path: ["data"]
        """)
        issues = validate_manifest_capabilities(manifest)
        warnings = [i for i in issues if i["severity"] == "warning"]
        assert len(warnings) >= 1
        assert any("PKCE" in w["message"] for w in warnings)

    def test_oauth_with_pkce_enabled_warns(self):
        manifest = textwrap.dedent("""\
            type: DeclarativeSource
            streams:
              - type: DeclarativeStream
                name: events
                retriever:
                  type: SimpleRetriever
                  requester:
                    type: HttpRequester
                    url_base: "https://api.example.com"
                    path: "/events"
                    authenticator:
                      type: OAuthAuthenticator
                      client_id: "id"
                      client_secret: "secret"
                      token_refresh_endpoint: "https://auth.example.com/token"
                      pkce_enabled: true
                  record_selector:
                    type: RecordSelector
                    extractor:
                      type: DpathExtractor
                      field_path: ["data"]
        """)
        issues = validate_manifest_capabilities(manifest)
        warnings = [i for i in issues if i["severity"] == "warning"]
        assert len(warnings) >= 1
        assert any("PKCE" in w["message"] for w in warnings)

    def test_oauth_without_pkce_no_warning(self):
        manifest = textwrap.dedent("""\
            type: DeclarativeSource
            streams:
              - type: DeclarativeStream
                name: events
                retriever:
                  type: SimpleRetriever
                  requester:
                    type: HttpRequester
                    url_base: "https://api.example.com"
                    path: "/events"
                    authenticator:
                      type: OAuthAuthenticator
                      client_id: "id"
                      client_secret: "secret"
                      token_refresh_endpoint: "https://auth.example.com/token"
                  record_selector:
                    type: RecordSelector
                    extractor:
                      type: DpathExtractor
                      field_path: ["data"]
        """)
        issues = validate_manifest_capabilities(manifest)
        warnings = [i for i in issues if i["severity"] == "warning"]
        assert len(warnings) == 0


# ── Valid manifest → empty list ──

class TestValidManifest:
    def test_valid_manifest_returns_empty(self):
        manifest = textwrap.dedent("""\
            type: DeclarativeSource
            streams:
              - type: DeclarativeStream
                name: events
                retriever:
                  type: SimpleRetriever
                  requester:
                    type: HttpRequester
                    url_base: "https://api.example.com"
                    path: "/events"
                    authenticator:
                      type: BearerAuthenticator
                      api_token: "tok123"
                  record_selector:
                    type: RecordSelector
                    extractor:
                      type: DpathExtractor
                      field_path: ["data"]
        """)
        issues = validate_manifest_capabilities(manifest)
        assert issues == []


# ── Multiple issues in one manifest ──

class TestMultipleIssues:
    def test_multiple_issues_collected(self):
        manifest = textwrap.dedent("""\
            type: DeclarativeSource
            streams:
              - type: DeclarativeStream
                name: events
                retriever:
                  type: SimpleRetriever
                  requester:
                    type: HttpRequester
                    url_base: "https://api.example.com"
                    path: "/events"
                    authenticator:
                      type: JwtAuthenticator
                      secret_key: "secret"
                      algorithm: RS256
                      token_duration: 3600
                  record_selector:
                    type: RecordSelector
                    extractor:
                      type: DpathExtractor
                      field_path: ["data"]
                  paginator:
                    type: BogusCustomPaginator
              - type: DeclarativeStream
                name: users
                retriever:
                  type: SimpleRetriever
                  requester:
                    type: HttpRequester
                    url_base: "https://api.example.com"
                    path: "/users"
                    authenticator:
                      type: OAuthAuthenticator
                      client_id: "id"
                      client_secret: "secret"
                      token_refresh_endpoint: "https://auth.example.com/token"
                      code_challenge_method: S256
                  record_selector:
                    type: RecordSelector
                    extractor:
                      type: DpathExtractor
                      field_path: ["data"]
        """)
        issues = validate_manifest_capabilities(manifest)
        errors = [i for i in issues if i["severity"] == "error"]
        warnings = [i for i in issues if i["severity"] == "warning"]
        # At least one error (BogusCustomPaginator) and two warnings (JWT RS256 + PKCE)
        assert len(errors) >= 1
        assert len(warnings) >= 2
        assert any(i["type"] == "BogusCustomPaginator" for i in errors)
        assert any("cryptography" in i["message"] for i in warnings)
        assert any("PKCE" in i["message"] for i in warnings)


# ── Structural types are skipped ──

class TestStructuralTypesSkipped:
    def test_declarative_source_not_flagged(self):
        manifest = textwrap.dedent("""\
            type: DeclarativeSource
            streams: []
        """)
        issues = validate_manifest_capabilities(manifest)
        assert not any(i["type"] == "DeclarativeSource" for i in issues)

    def test_declarative_stream_not_flagged(self):
        manifest = textwrap.dedent("""\
            type: DeclarativeSource
            streams:
              - type: DeclarativeStream
                name: events
                retriever:
                  type: SimpleRetriever
                  requester:
                    type: HttpRequester
                    url_base: "https://api.example.com"
                    path: "/events"
                  record_selector:
                    type: RecordSelector
                    extractor:
                      type: DpathExtractor
                      field_path: ["data"]
        """)
        issues = validate_manifest_capabilities(manifest)
        assert not any(i["type"] == "DeclarativeStream" for i in issues)
