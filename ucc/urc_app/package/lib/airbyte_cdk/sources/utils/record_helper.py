#
# Copyright (c) 2023 Airbyte, Inc., all rights reserved.
#
import time
from collections.abc import Mapping as ABCMapping
from typing import Any, Mapping, Optional

from airbyte_cdk.models import (
    AirbyteLogMessage,
    AirbyteMessage,
    AirbyteRecordMessage,
    AirbyteRecordMessageFileReference,
    AirbyteTraceMessage,
)
from airbyte_cdk.models import Type as MessageType
from airbyte_cdk.sources.streams.core import StreamData
from airbyte_cdk.sources.utils.transform import TransformConfig, TypeTransformer


def stream_data_to_airbyte_message(
    stream_name: str,
    data_or_message: StreamData,
    transformer: TypeTransformer = TypeTransformer(TransformConfig.NoTransform),
    schema: Optional[Mapping[str, Any]] = None,
    file_reference: Optional[AirbyteRecordMessageFileReference] = None,
) -> AirbyteMessage:
    if schema is None:
        schema = {}

    if isinstance(data_or_message, ABCMapping):
        data = dict(data_or_message)
        now_millis = time.time_ns() // 1_000_000
        transformer.transform(data, schema)
        message = AirbyteRecordMessage(
            stream=stream_name,
            data=data,
            emitted_at=now_millis,
            file_reference=file_reference,
        )
        return AirbyteMessage(type=MessageType.RECORD, record=message)
    elif isinstance(data_or_message, AirbyteTraceMessage):
        return AirbyteMessage(type=MessageType.TRACE, trace=data_or_message)
    elif isinstance(data_or_message, AirbyteLogMessage):
        return AirbyteMessage(type=MessageType.LOG, log=data_or_message)
    else:
        raise ValueError(
            f"Unexpected type for data_or_message: {type(data_or_message)}: {data_or_message}"
        )
