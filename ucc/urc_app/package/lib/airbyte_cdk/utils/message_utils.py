# Copyright (c) 2024 Airbyte, Inc., all rights reserved.

from airbyte_cdk.models import AirbyteMessage, Type
from airbyte_cdk.sources.connector_state_manager import HashableStreamDescriptor


def get_stream_descriptor(message: AirbyteMessage) -> HashableStreamDescriptor:
    if message.type == Type.RECORD:
        return HashableStreamDescriptor(
            name=message.record.stream,  # type: ignore[union-attr]
            namespace=message.record.namespace,  # type: ignore[union-attr]
        )
    elif message.type == Type.STATE:
        if not message.state.stream or not message.state.stream.stream_descriptor:  # type: ignore[union-attr]
            raise ValueError(
                "State message was not in per-stream state format, which is required for record counts."
            )
        return HashableStreamDescriptor(
            name=message.state.stream.stream_descriptor.name,  # type: ignore[union-attr]
            namespace=message.state.stream.stream_descriptor.namespace,  # type: ignore[union-attr]
        )
    else:
        raise NotImplementedError(
            f"get_stream_descriptor is not implemented for message type '{message.type}'."
        )
