"""Tests for urc.components.auth — authentication strategies."""

import base64

import requests
import responses

from urc.components.auth import (
    NoAuth,
    BearerAuth,
    BasicAuth,
    ApiKeyAuth,
    OAuth2Auth,
    DigestAuth,
    JwtAuth,
    SessionTokenAuth,
    SelectiveAuth,
)


def _make_session():
    return requests.Session()


# ── NoAuth ──


def test_no_auth_leaves_session_unchanged(mock_config):
    session = _make_session()
    original_headers = dict(session.headers)
    auth = NoAuth({}, mock_config)
    auth.apply(session, mock_config)
    assert dict(session.headers) == original_headers


# ── BearerAuth ──


def test_bearer_auth_sets_header(mock_config):
    session = _make_session()
    auth = BearerAuth(
        {"api_token": "{{ config['api_key'] }}"},
        mock_config,
    )
    auth.apply(session, mock_config)
    assert session.headers["Authorization"] == "Bearer test-key-123"


def test_bearer_auth_static_token(mock_config):
    session = _make_session()
    auth = BearerAuth({"api_token": "static-token"}, mock_config)
    auth.apply(session, mock_config)
    assert session.headers["Authorization"] == "Bearer static-token"


# ── BasicAuth ──


def test_basic_auth_sets_header(mock_config):
    session = _make_session()
    auth = BasicAuth(
        {
            "username": "{{ config['username'] }}",
            "password": "{{ config['password'] }}",
        },
        mock_config,
    )
    auth.apply(session, mock_config)
    expected = base64.b64encode(b"testuser:testpass").decode()
    assert session.headers["Authorization"] == f"Basic {expected}"


# ── ApiKeyAuth ──


def test_api_key_auth_header_injection(mock_config):
    session = _make_session()
    auth = ApiKeyAuth(
        {
            "api_token": "{{ config['api_key'] }}",
            "header": "X-Api-Key",
        },
        mock_config,
    )
    auth.apply(session, mock_config)
    assert session.headers["X-Api-Key"] == "test-key-123"


def test_api_key_auth_request_parameter_injection(mock_config):
    session = _make_session()
    auth = ApiKeyAuth(
        {
            "api_token": "{{ config['api_key'] }}",
            "inject_into": {
                "inject_into": "request_parameter",
                "field_name": "api_key",
            },
        },
        mock_config,
    )
    result = auth.apply(session, mock_config)
    assert result == {"params": {"api_key": "test-key-123"}}


# ── OAuth2Auth ──


def test_oauth2_auth_fetches_token(mock_config, activate_responses):
    activate_responses.add(
        responses.POST,
        "https://auth.test/token",
        json={"access_token": "oauth-token-xyz", "expires_in": 3600},
        status=200,
    )

    session = _make_session()
    auth = OAuth2Auth(
        {
            "client_id": "my-client",
            "client_secret": "my-secret",
            "token_refresh_endpoint": "https://auth.test/token",
            "grant_type": "client_credentials",
        },
        mock_config,
    )
    auth.apply(session, mock_config)
    assert session.headers["Authorization"] == "Bearer oauth-token-xyz"


def test_oauth2_auth_caches_token(mock_config, activate_responses):
    activate_responses.add(
        responses.POST,
        "https://auth.test/token",
        json={"access_token": "cached-token", "expires_in": 3600},
        status=200,
    )

    session = _make_session()
    auth = OAuth2Auth(
        {
            "client_id": "c",
            "client_secret": "s",
            "token_refresh_endpoint": "https://auth.test/token",
        },
        mock_config,
    )
    auth.apply(session, mock_config)
    # Second call should use cached token (only 1 POST registered)
    session2 = _make_session()
    auth.apply(session2, mock_config)
    assert session2.headers["Authorization"] == "Bearer cached-token"
    assert len(activate_responses.calls) == 1  # only one token request


# ── DigestAuth ──


def test_digest_auth_sets_session_auth(mock_config):
    session = _make_session()
    auth = DigestAuth(
        {
            "username": "{{ config['username'] }}",
            "password": "{{ config['password'] }}",
        },
        mock_config,
    )
    auth.apply(session, mock_config)
    assert session.auth is not None
    from requests.auth import HTTPDigestAuth
    assert isinstance(session.auth, HTTPDigestAuth)


# ── JwtAuth ──


def test_jwt_auth_hs256(mock_config):
    session = _make_session()
    auth = JwtAuth(
        {
            "secret_key": "my-secret",
            "algorithm": "HS256",
            "token_duration": 600,
            "header_prefix": "Bearer",
        },
        mock_config,
    )
    auth.apply(session, mock_config)
    assert "Authorization" in session.headers
    assert session.headers["Authorization"].startswith("Bearer ")
    # Decode and verify
    import jwt as pyjwt
    token_str = session.headers["Authorization"].split(" ", 1)[1]
    payload = pyjwt.decode(token_str, "my-secret", algorithms=["HS256"])
    assert "iat" in payload
    assert "exp" in payload
    assert payload["exp"] - payload["iat"] == 600


# ── SessionTokenAuth ──


def test_session_token_auth_login_and_inject(mock_config, activate_responses):
    # Register login endpoint
    activate_responses.add(
        responses.POST,
        "https://mock-api.test/login",
        json={"session": {"token": "sess-abc"}},
        status=200,
    )

    session = _make_session()
    auth = SessionTokenAuth(
        {
            "login_requester": {
                "type": "HttpRequester",
                "url": "https://mock-api.test/login",
                "http_method": "POST",
            },
            "session_token_path": ["session", "token"],
            "request_authentication": {
                "type": "BearerAuthenticator",
            },
        },
        mock_config,
    )
    auth.apply(session, mock_config)
    assert session.headers["Authorization"] == "Bearer sess-abc"


# ── SelectiveAuth ──


def test_selective_auth_routes_based_on_config():
    config = {
        "auth_type": "bearer",
        "api_key": "sel-key",
    }
    auth = SelectiveAuth(
        {
            "authenticator_selection_path": ["auth_type"],
            "authenticators": {
                "bearer": {
                    "type": "BearerAuthenticator",
                    "api_token": "{{ config['api_key'] }}",
                },
                "basic": {
                    "type": "BasicHttpAuthenticator",
                    "username": "user",
                    "password": "pass",
                },
            },
        },
        config,
    )
    session = _make_session()
    auth.apply(session, config)
    assert session.headers["Authorization"] == "Bearer sel-key"


def test_selective_auth_fallback_to_noauth():
    config = {"auth_type": "unknown"}
    auth = SelectiveAuth(
        {
            "authenticator_selection_path": ["auth_type"],
            "authenticators": {
                "bearer": {
                    "type": "BearerAuthenticator",
                    "api_token": "tok",
                },
            },
        },
        config,
    )
    session = _make_session()
    original_headers = dict(session.headers)
    auth.apply(session, config)
    # NoAuth fallback — no Authorization header added
    assert "Authorization" not in session.headers
