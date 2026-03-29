# Authentication components — apply credentials to HTTP requests.

import base64
import logging
import time
from typing import Any, Dict, Optional

import requests

from urc.interpolation import eval_string
from urc.registry import component

logger = logging.getLogger(__name__)


class BaseAuth:
    """Base class for authentication strategies."""

    def apply(self, session: requests.Session, config: dict) -> None:
        """Apply authentication to the session. Override in subclasses."""
        pass


@component("NoAuth")
class NoAuth(BaseAuth):
    def __init__(self, definition: dict, config: dict, **kwargs):
        pass


@component("ApiKeyAuthenticator")
class ApiKeyAuth(BaseAuth):
    def __init__(self, definition: dict, config: dict, **kwargs):
        self._api_token = eval_string(definition.get("api_token", ""), config)
        self._header = eval_string(
            definition.get("header", ""),
            config,
        )
        # Support inject_into for non-header injection
        self._inject_into = definition.get("inject_into")

    def apply(self, session: requests.Session, config: dict) -> Optional[Dict[str, Any]]:
        """Returns extra request kwargs (headers, params) for API key injection."""
        token = eval_string(self._api_token, config) if "{{" in str(self._api_token) else self._api_token

        if self._inject_into:
            inject_type = self._inject_into.get("inject_into", "header")
            field_name = eval_string(self._inject_into.get("field_name", ""), config)
            if inject_type == "header":
                session.headers[field_name] = str(token)
            elif inject_type == "request_parameter":
                return {"params": {field_name: str(token)}}
        elif self._header:
            session.headers[self._header] = str(token)

        return None


@component("BearerAuthenticator")
class BearerAuth(BaseAuth):
    def __init__(self, definition: dict, config: dict, **kwargs):
        self._api_token_template = definition.get("api_token", "")

    def apply(self, session: requests.Session, config: dict) -> None:
        token = eval_string(self._api_token_template, config)
        session.headers["Authorization"] = f"Bearer {token}"


@component("BasicHttpAuthenticator")
class BasicAuth(BaseAuth):
    def __init__(self, definition: dict, config: dict, **kwargs):
        self._username_template = definition.get("username", "")
        self._password_template = definition.get("password", "")

    def apply(self, session: requests.Session, config: dict) -> None:
        username = eval_string(self._username_template, config)
        password = eval_string(self._password_template, config)
        credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
        session.headers["Authorization"] = f"Basic {credentials}"


@component("OAuthAuthenticator")
class OAuth2Auth(BaseAuth):
    def __init__(self, definition: dict, config: dict, **kwargs):
        self._client_id_template = definition.get("client_id", "")
        self._client_secret_template = definition.get("client_secret", "")
        self._token_url_template = definition.get("token_refresh_endpoint", "")
        self._grant_type = definition.get("grant_type", "client_credentials")
        self._refresh_token_template = definition.get("refresh_token", "")
        self._scopes = definition.get("scopes", [])
        self._access_token_name = definition.get("access_token_name", "access_token")
        self._expires_in_name = definition.get("expires_in_name", "expires_in")
        self._refresh_request_body = definition.get("refresh_request_body", {})
        self._refresh_request_headers = definition.get("refresh_request_headers", {})
        self._access_token = None
        self._token_expiry = 0

    def apply(self, session: requests.Session, config: dict) -> None:
        if self._access_token is None or time.time() >= self._token_expiry:
            self._refresh_token(config)
        if self._access_token:
            session.headers["Authorization"] = f"Bearer {self._access_token}"

    def _refresh_token(self, config: dict) -> None:
        token_url = eval_string(self._token_url_template, config)
        client_id = eval_string(self._client_id_template, config)
        client_secret = eval_string(self._client_secret_template, config)

        body = {
            "grant_type": self._grant_type,
            "client_id": client_id,
            "client_secret": client_secret,
        }
        if self._grant_type == "refresh_token" and self._refresh_token_template:
            body["refresh_token"] = eval_string(self._refresh_token_template, config)
        if self._scopes:
            body["scope"] = " ".join(self._scopes)

        # Merge any custom request body fields
        for k, v in self._refresh_request_body.items():
            body[k] = eval_string(v, config)

        headers = {}
        for k, v in self._refresh_request_headers.items():
            headers[k] = eval_string(v, config)

        try:
            resp = requests.post(token_url, data=body, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            self._access_token = data.get(self._access_token_name)
            expires_in = data.get(self._expires_in_name, 3600)
            self._token_expiry = time.time() + int(expires_in) - 60  # refresh 60s early
        except Exception as e:
            logger.error(f"OAuth2 token refresh failed: {e}")
            raise


@component("SelectiveAuthenticator")
class SelectiveAuth(BaseAuth):
    """Selects auth strategy based on a config value."""

    def __init__(self, definition: dict, config: dict, **kwargs):
        from urc.registry import create as create_component

        selection_path = definition.get("authenticator_selection_path", [])
        # Navigate config to find selected auth type
        selected = config
        for key in selection_path:
            selected = selected.get(key, "") if isinstance(selected, dict) else ""

        authenticators = definition.get("authenticators", {})
        auth_def = authenticators.get(str(selected))
        if auth_def:
            self._delegate = create_component(auth_def, config)
        else:
            self._delegate = NoAuth({}, config)

    def apply(self, session: requests.Session, config: dict) -> Optional[Dict[str, Any]]:
        return self._delegate.apply(session, config)


@component("DigestHttpAuthenticator")
class DigestAuth(BaseAuth):
    """HTTP Digest authentication using requests.auth.HTTPDigestAuth."""

    def __init__(self, definition: dict, config: dict, **kwargs):
        self._username_template = definition.get("username", "")
        self._password_template = definition.get("password", "")

    def apply(self, session: requests.Session, config: dict) -> None:
        from requests.auth import HTTPDigestAuth

        username = eval_string(self._username_template, config)
        password = eval_string(self._password_template, config)
        session.auth = HTTPDigestAuth(username, password)


@component("JwtAuthenticator")
class JwtAuth(BaseAuth):
    """Signs a JWT token for each request using PyJWT."""

    def __init__(self, definition: dict, config: dict, **kwargs):
        self._secret_key_template = definition.get("secret_key", "")
        self._algorithm = definition.get("algorithm", "HS256")
        self._token_duration = int(definition.get("token_duration", 1200))
        self._header_prefix = definition.get("header_prefix", "Bearer")
        self._jwt_headers = definition.get("jwt_headers", {})
        self._jwt_payload = definition.get("jwt_payload", {})
        self._additional_jwt_headers = definition.get("additional_jwt_headers", {})
        self._additional_jwt_payload = definition.get("additional_jwt_payload", {})
        self._cached_token: Optional[str] = None
        self._token_expiry: float = 0

    def apply(self, session: requests.Session, config: dict) -> None:
        now = time.time()
        if self._cached_token is not None and now < self._token_expiry:
            session.headers["Authorization"] = f"{self._header_prefix} {self._cached_token}"
            return

        try:
            import jwt as pyjwt
        except ImportError:
            raise ImportError(
                "PyJWT is required for JwtAuthenticator. "
                "Install it with: pip install PyJWT"
            )

        secret_key = eval_string(self._secret_key_template, config)
        algorithm = eval_string(str(self._algorithm), config)
        header_prefix = eval_string(str(self._header_prefix), config)

        # Build JWT headers
        token_headers: Dict[str, Any] = {}
        for k, v in self._jwt_headers.items():
            token_headers[k] = eval_string(str(v), config) if isinstance(v, str) else v
        for k, v in self._additional_jwt_headers.items():
            token_headers[k] = eval_string(str(v), config) if isinstance(v, str) else v

        # Build JWT payload
        payload: Dict[str, Any] = {}
        for k, v in self._jwt_payload.items():
            payload[k] = eval_string(str(v), config) if isinstance(v, str) else v
        for k, v in self._additional_jwt_payload.items():
            payload[k] = eval_string(str(v), config) if isinstance(v, str) else v

        # Set standard time claims if not already present
        payload.setdefault("iat", int(now))
        payload.setdefault("exp", int(now) + self._token_duration)

        token = pyjwt.encode(
            payload,
            secret_key,
            algorithm=algorithm,
            headers=token_headers if token_headers else None,
        )
        # PyJWT >= 2.0 returns str; older versions return bytes
        if isinstance(token, bytes):
            token = token.decode("utf-8")

        self._cached_token = token
        self._token_expiry = now + self._token_duration
        session.headers["Authorization"] = f"{header_prefix} {token}"


@component("SessionTokenAuthenticator")
class SessionTokenAuth(BaseAuth):
    """Authenticates by first making a login request to obtain a session token,
    then injecting it into subsequent requests."""

    def __init__(self, definition: dict, config: dict, **kwargs):
        from urc.registry import create as create_component

        login_requester_def = definition.get("login_requester", {})
        if login_requester_def:
            self._login_requester = create_component(login_requester_def, config)
        else:
            self._login_requester = None

        self._session_token_path: list = definition.get("session_token_path", [])
        self._request_authentication = definition.get("request_authentication", {})
        self._expiration_duration_str = definition.get("expiration_duration")
        self._config = config

        # Parse expiration duration
        self._expiration_seconds: Optional[float] = None
        if self._expiration_duration_str:
            try:
                import isodate
                td = isodate.parse_duration(self._expiration_duration_str)
                self._expiration_seconds = td.total_seconds()
            except ImportError:
                logger.warning(
                    "isodate not available; expiration_duration will be ignored. "
                    "Install it with: pip install isodate"
                )
            except Exception as e:
                logger.warning(f"Failed to parse expiration_duration: {e}")

        self._session_token: Optional[str] = None
        self._token_obtained_at: float = 0

    def _is_token_expired(self) -> bool:
        if self._session_token is None:
            return True
        if self._expiration_seconds is not None:
            return time.time() >= (self._token_obtained_at + self._expiration_seconds)
        return False

    def _fetch_token(self, config: dict) -> str:
        """Call the login requester and extract the session token via dpath."""
        if self._login_requester is None:
            raise ValueError("SessionTokenAuthenticator: login_requester is not configured")

        # Build a temporary session for the login request
        login_session = requests.Session()
        response = self._login_requester.send(login_session, config)
        response.raise_for_status()
        data = response.json()

        # Walk the token path through the response JSON
        token = data
        for key in self._session_token_path:
            if isinstance(token, dict):
                token = token[key]
            elif isinstance(token, (list, tuple)):
                token = token[int(key)]
            else:
                raise ValueError(
                    f"SessionTokenAuthenticator: cannot traverse path "
                    f"{self._session_token_path} in response"
                )
        return str(token)

    def apply(self, session: requests.Session, config: dict) -> Optional[Dict[str, Any]]:
        if self._is_token_expired():
            try:
                self._session_token = self._fetch_token(config)
                self._token_obtained_at = time.time()
            except Exception as e:
                logger.error(f"SessionTokenAuthenticator login failed: {e}")
                raise

        # Inject the token based on request_authentication config
        auth_type = self._request_authentication.get("type", "")
        token = self._session_token or ""

        if auth_type == "ApiKeyAuthenticator":
            header = eval_string(
                self._request_authentication.get("header", ""),
                config,
            )
            api_token_template = self._request_authentication.get("api_token", "")
            # If the api_token template references the token, inject it;
            # otherwise use the session token directly.
            if api_token_template:
                # Make token available for interpolation
                token_config = {**config, "_session_token": token}
                resolved = eval_string(api_token_template, token_config)
            else:
                resolved = token

            inject_into = self._request_authentication.get("inject_into")
            if inject_into:
                inject_type = inject_into.get("inject_into", "header")
                field_name = eval_string(inject_into.get("field_name", ""), config)
                if inject_type == "header":
                    session.headers[field_name] = str(resolved)
                elif inject_type == "request_parameter":
                    return {"params": {field_name: str(resolved)}}
            elif header:
                session.headers[header] = str(resolved)

        elif auth_type == "BearerAuthenticator":
            session.headers["Authorization"] = f"Bearer {token}"

        else:
            # Default: inject as Bearer
            session.headers["Authorization"] = f"Bearer {token}"

        return None
