# Minimal stub for exception_handler (trimmed from full CDK)
from typing import List, Mapping


def generate_failed_streams_error_message(stream_failures: Mapping[str, List[Exception]]) -> str:
    failures = "\n".join(
        [
            f"{stream}: {exception!r}"
            for stream, exceptions in stream_failures.items()
            for exception in exceptions
        ]
    )
    return f"During the sync, the following streams did not sync successfully: {failures}"
