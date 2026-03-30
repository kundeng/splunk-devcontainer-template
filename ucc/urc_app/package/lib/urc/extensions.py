"""URC model extensions — subclass overrides for generated models.

Add validators, computed properties, or behavioral overrides here.
Never edit models_generated.py directly — it will be overwritten.

Usage:
    # Import the extended model instead of the generated one:
    from urc.extensions import DeclarativeSourceExt as SourceModel

    # Or if no extension exists, import generated directly:
    from urc.models_generated import DeclarativeStream
"""

# from urc.models_generated import DeclarativeSource1
#
# Example: add a validator or property to a generated model
# class DeclarativeSourceExt(DeclarativeSource1):
#     @validator("streams", pre=True)
#     def validate_streams_not_empty(cls, v):
#         if not v:
#             raise ValueError("At least one stream is required")
#         return v
