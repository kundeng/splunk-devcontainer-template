"""Reusable test data for unit tests."""

SAMPLE_RECORDS = [
    {
        "id": i,
        "name": f"Item {i}",
        "created_at": f"2024-01-{i:02d}T00:00:00Z",
        "updated_at": f"2024-02-{i:02d}T12:00:00Z",
    }
    for i in range(1, 11)
]

SAMPLE_CHECKPOINT = {
    "events": {
        "cursor_field": "updated_at",
        "cursor_value": "2024-02-05T12:00:00Z",
    }
}
