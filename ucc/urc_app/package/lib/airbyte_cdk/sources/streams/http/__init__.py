#
# Copyright (c) 2023 Airbyte, Inc., all rights reserved.
#

# Lazy imports to avoid circular dependency chain
def __getattr__(name):
    if name == "HttpStream" or name == "HttpSubStream":
        from .http import HttpStream, HttpSubStream
        if name == "HttpStream":
            return HttpStream
        return HttpSubStream
    elif name == "HttpClient":
        from .http_client import HttpClient
        return HttpClient
    elif name == "UserDefinedBackoffException":
        from .exceptions import UserDefinedBackoffException
        return UserDefinedBackoffException
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["HttpClient", "HttpStream", "HttpSubStream", "UserDefinedBackoffException"]
