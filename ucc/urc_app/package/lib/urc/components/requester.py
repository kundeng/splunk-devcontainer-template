# HTTP Requester — makes authenticated, paginated HTTP requests.

import logging
from typing import Any, Dict, List, Optional

import requests

from urc.interpolation import eval_string, eval_dict
from urc.registry import component

logger = logging.getLogger(__name__)


# ── Request body type constants ──

REQUEST_BODY_PLAIN_TEXT = "RequestBodyPlainText"
REQUEST_BODY_URL_ENCODED_FORM = "RequestBodyUrlEncodedForm"
REQUEST_BODY_JSON_OBJECT = "RequestBodyJsonObject"
REQUEST_BODY_GRAPHQL = "RequestBodyGraphQL"


@component("HttpRequester")
class HttpRequester:
    """Makes HTTP requests with auth, headers, params, and error handling."""

    def __init__(self, definition: dict, config: dict, **kwargs):
        from urc.registry import create as create_component

        # URL: prefer 'url' over deprecated 'url_base' + 'path'
        self._url_template = definition.get("url", "")
        if not self._url_template:
            base = definition.get("url_base", "")
            path = definition.get("path", "")
            self._url_template = f"{base}{path}" if base else path

        self._method = definition.get("http_method", "GET").upper()
        self._headers = definition.get("request_headers", {})
        self._params = definition.get("request_parameters", {})
        self._config = config

        # Request body — structured type or legacy dict fallback
        self._request_body = definition.get("request_body")
        self._body_json = definition.get("request_body_json") or {}

        # Proxy support: read from definition or config
        proxy_url = definition.get("proxy_url") or config.get("proxy_url")
        self._proxy: Optional[str] = proxy_url or None

        # SSL certificate verification
        ca_bundle = definition.get("ca_bundle_path")
        if ca_bundle:
            self._verify_ssl: Any = ca_bundle
        else:
            self._verify_ssl = definition.get("verify_ssl", True)

        # Configurable timeout (seconds)
        self._timeout = float(definition.get("timeout", 60))

        # Auth
        auth_def = definition.get("authenticator")
        self._auth = None
        if auth_def and auth_def.get("type"):
            self._auth = create_component(auth_def, config)

        # Error handler
        error_def = definition.get("error_handler")
        if error_def and error_def.get("type"):
            self._error_handler = create_component(error_def, config)
        else:
            from urc.components.error_handler import DefaultErrorHandler
            self._error_handler = DefaultErrorHandler({"type": "DefaultErrorHandler"}, config)

        # Rate limiter (optional)
        rate_limiter_def = definition.get("rate_limiter")
        if rate_limiter_def and rate_limiter_def.get("type"):
            self._rate_limiter = create_component(rate_limiter_def, config)
        else:
            self._rate_limiter = None

        # Session is reused across pages within a collection cycle, which
        # automatically persists cookies between requests (via requests.Session).
        self._session = requests.Session()
        self._session.headers["Accept"] = "application/json"
        self._session.headers["User-Agent"] = "URC-Splunk/0.1"

    def _resolve_body(self, config: dict, context: dict):
        """Resolve the request body based on type.

        Returns:
            Tuple of (data, json_body) — one will be set, the other None.
            data is used for plain text and form-encoded bodies.
            json_body is used for JSON and GraphQL bodies.
        """
        # Structured request_body with a "type" key
        if isinstance(self._request_body, dict) and self._request_body.get("type"):
            body_type = self._request_body["type"]
            value = self._request_body.get("value", self._request_body.get("values", {}))

            if body_type == REQUEST_BODY_PLAIN_TEXT:
                resolved = eval_string(value, config, **context) if isinstance(value, str) else value
                return resolved, None

            elif body_type == REQUEST_BODY_URL_ENCODED_FORM:
                resolved = eval_dict(value, config, **context) if isinstance(value, dict) else value
                return resolved, None

            elif body_type == REQUEST_BODY_JSON_OBJECT:
                resolved = eval_dict(value, config, **context) if isinstance(value, dict) else value
                return None, resolved

            elif body_type == REQUEST_BODY_GRAPHQL:
                query = eval_string(value, config, **context) if isinstance(value, str) else value
                return None, {"query": query}

        # Legacy: request_body_json or request_body as plain dict
        body_def = self._request_body if isinstance(self._request_body, dict) else self._body_json
        if body_def:
            return None, eval_dict(body_def, config, **context)

        return None, None

    def send(
        self,
        config: dict,
        extra_params: Optional[dict] = None,
        extra_headers: Optional[dict] = None,
        stream_slice: Optional[dict] = None,
    ) -> requests.Response:
        """Send an HTTP request with retries and backoff."""
        context = {
            "stream_interval": stream_slice or {},
            "stream_partition": stream_slice or {},
            "stream_slice": stream_slice or {},
        }

        # Resolve URL
        url = eval_string(self._url_template, config, **context)

        # Resolve headers
        headers = {}
        if isinstance(self._headers, dict):
            for k, v in self._headers.items():
                headers[k] = eval_string(v, config, **context)
        if extra_headers:
            headers.update(extra_headers)

        # Resolve params
        params = {}
        if isinstance(self._params, dict):
            for k, v in self._params.items():
                if isinstance(v, str):
                    params[k] = eval_string(v, config, **context)
                else:
                    params[k] = v
        if extra_params:
            # Separate special keys
            path_override = extra_params.pop("_path_override", None)
            extra_hdrs = extra_params.pop("_headers", None)
            params.update(extra_params)
            if path_override:
                url = path_override
            if extra_hdrs:
                headers.update(extra_hdrs)

        # Resolve body (only for methods that support a body)
        data = None
        json_body = None
        if self._method in ("POST", "PUT", "PATCH", "DELETE"):
            data, json_body = self._resolve_body(config, context)

        # Apply auth
        if self._auth:
            auth_extras = self._auth.apply(self._session, config)
            if auth_extras and "params" in auth_extras:
                params.update(auth_extras["params"])

        # Proxy configuration
        proxies = None
        if self._proxy:
            proxies = {"http": self._proxy, "https": self._proxy}

        # Execute with retries
        max_retries = self._error_handler.max_retries
        for attempt in range(max_retries + 1):
            # Apply rate limiting before each request
            if self._rate_limiter:
                wait = self._rate_limiter.acquire(url)
                if wait > 0:
                    logger.debug(f"Rate limiter waited {wait:.2f}s for {url}")

            try:
                response = self._session.request(
                    method=self._method,
                    url=url,
                    params=params,
                    headers=headers,
                    data=data,
                    json=json_body,
                    timeout=self._timeout,
                    verify=self._verify_ssl,
                    proxies=proxies,
                )

                if response.ok:
                    return response

                if self._error_handler.should_retry(response) and attempt < max_retries:
                    wait = self._error_handler.get_backoff_time(response, attempt)
                    logger.warning(
                        f"Request to {url} returned {response.status_code}, "
                        f"retrying in {wait:.1f}s (attempt {attempt + 1}/{max_retries})"
                    )
                    import time
                    time.sleep(wait)
                    continue

                # Non-retryable error
                response.raise_for_status()

            except requests.ConnectionError as e:
                if attempt < max_retries:
                    wait = self._error_handler.get_backoff_time(
                        requests.Response(), attempt
                    )
                    logger.warning(f"Connection error: {e}, retrying in {wait:.1f}s")
                    import time
                    time.sleep(wait)
                    continue
                raise

        # Should not reach here
        raise RuntimeError(f"Max retries ({max_retries}) exceeded for {url}")
