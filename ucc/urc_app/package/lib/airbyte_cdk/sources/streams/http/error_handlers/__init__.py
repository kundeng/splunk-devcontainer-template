#
# Copyright (c) 2023 Airbyte, Inc., all rights reserved.
#

def __getattr__(name):
    _imports = {
        "BackoffStrategy": ".backoff_strategy",
        "DefaultBackoffStrategy": ".default_backoff_strategy",
        "ErrorHandler": ".error_handler",
        "ErrorMessageParser": ".error_message_parser",
        "ErrorResolution": ".response_models",
        "HttpStatusErrorHandler": ".http_status_error_handler",
        "JsonErrorMessageParser": ".json_error_message_parser",
        "ResponseAction": ".response_models",
    }
    if name in _imports:
        import importlib
        mod = importlib.import_module(_imports[name], package=__name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
