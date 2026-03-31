"""Inline YAML manifest strings for testing."""

SIMPLE_GET_MANIFEST = """\
type: DeclarativeSource
streams:
  - type: DeclarativeStream
    name: users
    retriever:
      type: SimpleRetriever
      requester:
        type: HttpRequester
        url: "https://mock-api.test/users"
        http_method: GET
      record_selector:
        type: RecordSelector
        extractor:
          type: DpathExtractor
          field_path: []
"""

PAGINATED_OFFSET_MANIFEST = """\
type: DeclarativeSource
streams:
  - type: DeclarativeStream
    name: items
    retriever:
      type: SimpleRetriever
      requester:
        type: HttpRequester
        url: "https://mock-api.test/items"
        http_method: GET
      record_selector:
        type: RecordSelector
        extractor:
          type: DpathExtractor
          field_path:
            - data
      paginator:
        type: DefaultPaginator
        pagination_strategy:
          type: OffsetIncrement
          page_size: 10
        page_token_option:
          type: RequestOption
          field_name: offset
          inject_into: request_parameter
        page_size_option:
          type: RequestOption
          field_name: limit
          inject_into: request_parameter
"""

BEARER_AUTH_MANIFEST = """\
type: DeclarativeSource
streams:
  - type: DeclarativeStream
    name: protected
    retriever:
      type: SimpleRetriever
      requester:
        type: HttpRequester
        url: "https://mock-api.test/protected"
        http_method: GET
        authenticator:
          type: BearerAuthenticator
          api_token: "{{ config['api_key'] }}"
      record_selector:
        type: RecordSelector
        extractor:
          type: DpathExtractor
          field_path: []
"""

INCREMENTAL_MANIFEST = """\
type: DeclarativeSource
streams:
  - type: DeclarativeStream
    name: events
    retriever:
      type: SimpleRetriever
      requester:
        type: HttpRequester
        url: "https://mock-api.test/events"
        http_method: GET
      record_selector:
        type: RecordSelector
        extractor:
          type: DpathExtractor
          field_path: []
    incremental_sync:
      type: DatetimeBasedCursor
      cursor_field: updated_at
      datetime_format: "%Y-%m-%dT%H:%M:%SZ"
      start_datetime: "2024-01-01T00:00:00Z"
"""

SUBSTREAM_MANIFEST = """\
type: DeclarativeSource
streams:
  - type: DeclarativeStream
    name: comments
    retriever:
      type: SimpleRetriever
      requester:
        type: HttpRequester
        url: "https://mock-api.test/posts/{{ stream_partition['post_id'] }}/comments"
        http_method: GET
      record_selector:
        type: RecordSelector
        extractor:
          type: DpathExtractor
          field_path: []
      partition_router:
        type: SubstreamPartitionRouter
        parent_stream_configs:
          - stream:
              type: DeclarativeStream
              name: posts
              retriever:
                type: SimpleRetriever
                requester:
                  type: HttpRequester
                  url: "https://mock-api.test/posts"
                  http_method: GET
                record_selector:
                  type: RecordSelector
                  extractor:
                    type: DpathExtractor
                    field_path: []
            parent_key: id
            partition_field: post_id
"""

TRANSFORM_MANIFEST = """\
type: DeclarativeSource
streams:
  - type: DeclarativeStream
    name: items
    retriever:
      type: SimpleRetriever
      requester:
        type: HttpRequester
        url: "https://mock-api.test/items"
        http_method: GET
      record_selector:
        type: RecordSelector
        extractor:
          type: DpathExtractor
          field_path: []
    transformations:
      - type: AddFields
        fields:
          - path: ["source"]
            value: "api"
      - type: RemoveFields
        field_pointers:
          - ["internal_id"]
      - type: KeysToSnakeCase
"""
