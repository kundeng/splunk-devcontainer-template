# Centralized error recovery policy.
#
# A single dict defines skip-vs-raise for every (component, method) pair.
# One place to audit all recovery decisions instead of hunting through 12 files.

from urc.structured_logger import emit


class SkipRecord(Exception):
    """Raised by recovery policy to skip the current record and continue."""
    pass


class SkipComponent(Exception):
    """Raised by lifecycle hooks to skip the current component entirely."""
    pass


# Policy table: (component_type, method_name) → action
# "raise" = re-raise exception (default, let retry/caller handle)
# "skip"  = emit warning, raise SkipRecord (engine skips this record)
RECOVERY_POLICY = {
    # Decoders — skip bad data, continue with what parsed
    ("CsvDecoder", "decode"):              "skip",
    ("JsonlDecoder", "decode"):            "skip",

    # Extractors — missing field → skip record
    ("DpathExtractor", "extract"):         "skip",

    # Transformations — bad template/key → skip record
    ("AddFields", "transform"):            "skip",
    ("RemoveFields", "transform"):         "skip",
    ("KeysReplace", "transform"):          "skip",
    ("SchemaNormalization", "transform"):   "skip",
    ("DpathFlattenFields", "transform"):   "skip",
    ("KeysToLower", "transform"):          "skip",
    ("KeysToSnakeCase", "transform"):      "skip",
    ("FlattenFields", "transform"):        "skip",

    # Timestamp — bad parse → skip (FetchTimestamp fallback in component itself)
    ("FieldBasedTimestamp", "resolve"):     "skip",
    ("CursorBasedTimestamp", "resolve"):    "skip",

    # HTTP/Auth — always re-raise, retry loop handles
    ("HttpRequester", "send"):             "raise",
    ("AsyncRetriever", "read_records"):    "raise",
    ("OAuth2Auth", "apply"):               "raise",
    ("SessionTokenAuth", "apply"):         "raise",
}


def recovery_error_hook(component_type: str, method_name: str, exc: Exception,
                        elapsed: float, context: dict) -> None:
    """Error hook that applies the centralized recovery policy.

    If policy is "skip", raises SkipRecord (caught by engine's record loop).
    If policy is "raise" (or unknown), does nothing (exception re-raised by wrapper).
    """
    key = (component_type, method_name)
    action = RECOVERY_POLICY.get(key, "raise")

    if action == "skip":
        emit(
            action="skip",
            component=component_type,
            method=method_name,
            error_type=type(exc).__name__,
            error=str(exc)[:200],
        )
        raise SkipRecord(str(exc)) from exc
