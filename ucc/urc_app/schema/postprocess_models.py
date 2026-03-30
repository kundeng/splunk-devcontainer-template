#!/usr/bin/env python3
"""Post-process generated models to add Pydantic v2 discriminated unions.

Only adds discriminators to SAFE unions — those where the type field is always
present in manifest data. Manifest data relies on propagate_types() which only
populates a subset of types. Unsafe unions keep plain Union[] and let pydantic
try each option.

Usage:
    python postprocess_models.py <models_file.py>
"""

import re
import sys

# These are the unions where `type` is ALWAYS present in manifest data
# (either required in the manifest YAML or populated by propagate_types).
# Do NOT add unions where nested objects may omit `type` — those break
# discriminated union validation.
SAFE_DISCRIMINATED_FIELDS = {
    # Top-level source
    "streams",
    "dynamic_streams",
    # Retriever type (SimpleRetriever vs AsyncRetriever)
    "retriever",
    # Auth type
    "authenticator",
    # Paginator type
    "paginator",
    # Pagination strategy
    "pagination_strategy",
    # Error handler
    "error_handler",
    "error_handlers",
    # Record selector parts
    "extractor",
    # Incremental sync
    "incremental_sync",
    # Partition router
    "partition_router",
    # Decoder
    "decoder",
    "download_decoder",
    # Check type
    "check",
}


def postprocess(filepath: str) -> None:
    with open(filepath) as f:
        content = f.read()

    # Ensure Annotated is imported
    if "from typing import" in content and "Annotated" not in content:
        content = content.replace(
            "from typing import ",
            "from typing import Annotated, ",
        )

    # Collect classes with required Literal type field
    literal_type_classes = set()
    for match in re.finditer(
        r'^class\s+(\w+)\(BaseModel\):.*?(?=^class\s|\Z)',
        content,
        re.MULTILINE | re.DOTALL,
    ):
        class_name = match.group(1)
        class_body = match.group(0)
        if re.search(r'\btype:\s*Literal\[', class_body):
            literal_type_classes.add(class_name)

    if not literal_type_classes:
        print("No classes with Literal type fields found. Skipping.")
        return

    discriminated_count = 0

    def replace_union(match):
        nonlocal discriminated_count
        full = match.group(0)
        field_name = match.group(1)  # the field name before the colon
        prefix = match.group(2)      # annotation prefix (Optional[, List[, etc.)
        union_content = match.group(3)
        suffix = match.group(4)

        # Only discriminate if field name is in the safe list
        if field_name.strip() not in SAFE_DISCRIMINATED_FIELDS:
            return full

        members = [m.strip() for m in union_content.split(",")]
        if all(m in literal_type_classes for m in members):
            discriminated_count += 1
            return f'{field_name}{prefix}Annotated[Union[{union_content}], Field(discriminator="type")]{suffix}'
        return full

    # Match: field_name: [Optional[|List[]Union[TypeA, TypeB, ...]...]
    content = re.sub(
        r'(\w+)(:\s*(?:Optional\[)?(?:List\[)?)Union\[([A-Z][\w\s,]+)\](\]?\]?)',
        replace_union,
        content,
    )

    with open(filepath, "w") as f:
        f.write(content)

    print(f"Post-processed: {filepath}")
    print(f"  Classes with Literal type: {len(literal_type_classes)}")
    print(f"  Discriminated unions added: {discriminated_count}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <models_file.py>")
        sys.exit(1)
    postprocess(sys.argv[1])
