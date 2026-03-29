#
# Copyright (c) 2023 Airbyte, Inc., all rights reserved.
#

def __getattr__(name):
    _imports = {
        "BasicHttpAuthenticator": ".token",
        "MultipleTokenAuthenticator": ".token",
        "Oauth2Authenticator": ".oauth",
        "SingleUseRefreshTokenOauth2Authenticator": ".oauth",
        "TokenAuthenticator": ".token",
    }
    if name in _imports:
        import importlib
        mod = importlib.import_module(_imports[name], package=__name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
