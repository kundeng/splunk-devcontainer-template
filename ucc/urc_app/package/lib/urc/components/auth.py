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
