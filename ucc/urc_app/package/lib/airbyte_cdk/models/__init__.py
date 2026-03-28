# The earlier versions of airbyte-cdk (0.28.0<=) had the airbyte_protocol python classes
# declared inline in the airbyte-cdk code. However, somewhere around Feb 2023 the
# Airbyte Protocol moved to its own repo/PyPi package, called airbyte-protocol-models.
# This directory including the airbyte_protocol.py and well_known_types.py files
# are just wrappers on top of that stand-alone package which do some namespacing magic
# to make the airbyte_protocol python classes available to the airbyte-cdk consumer as part
# of airbyte-cdk rather than a standalone package.
from .airbyte_protocol import (
    AdvancedAuth,
    AirbyteAnalyticsTraceMessage,
    AirbyteCatalog,
    AirbyteConnectionStatus,
    AirbyteControlConnectorConfigMessage,
    AirbyteControlMessage,
    AirbyteErrorTraceMessage,
    AirbyteEstimateTraceMessage,
    AirbyteGlobalState,
    AirbyteLogMessage,
    AirbyteMessage,
    AirbyteProtocol,
    AirbyteRecordMessage,
    AirbyteRecordMessageFileReference,
    AirbyteStateBlob,
    AirbyteStateMessage,
    AirbyteStateStats,
    AirbyteStateType,
    AirbyteStream,
    AirbyteStreamState,
    AirbyteStreamStatus,
    AirbyteStreamStatusReason,
    AirbyteStreamStatusReasonType,
    AirbyteStreamStatusTraceMessage,
    AirbyteTraceMessage,
    AuthFlowType,
    ConfiguredAirbyteCatalog,
    ConfiguredAirbyteStream,
    ConnectorSpecification,
    DestinationSyncMode,
    EstimateType,
    FailureType,
    Level,
    OAuthConfigSpecification,
    OauthConnectorInputSpecification,
    OrchestratorType,
    State,
    Status,
    StreamDescriptor,
    SyncMode,
    TraceType,
    Type,
)
try:
    from .airbyte_protocol_serializers import (
        AirbyteMessageSerializer,
        AirbyteStateMessageSerializer,
        AirbyteStreamStateSerializer,
        ConfiguredAirbyteCatalogSerializer,
        ConfiguredAirbyteStreamSerializer,
        ConnectorSpecificationSerializer,
    )
except (ImportError, TypeError):
    # serpyco-rs not available — serializers are not needed for Splunk runtime
    AirbyteMessageSerializer = None  # type: ignore[assignment,misc]
    AirbyteStateMessageSerializer = None  # type: ignore[assignment,misc]
    AirbyteStreamStateSerializer = None  # type: ignore[assignment,misc]
    ConfiguredAirbyteCatalogSerializer = None  # type: ignore[assignment,misc]
    ConfiguredAirbyteStreamSerializer = None  # type: ignore[assignment,misc]
    ConnectorSpecificationSerializer = None  # type: ignore[assignment,misc]
from .well_known_types import (
    BinaryData,
    Boolean,
    Date,
    Integer,
    Model,
    Number,
    String,
    TimestampWithoutTimezone,
    TimestampWithTimezone,
    TimeWithoutTimezone,
    TimeWithTimezone,
)
