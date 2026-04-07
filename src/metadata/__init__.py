# metadata/__init__.py — Image metadata and sidecar JSON subpackage
# Implements: P01 (S19)
#
# Re-export from current dated implementation file per S01 Section 10.
from src.metadata.p01_image_metadata_2026_04_06 import (  # noqa: F401
    AdcsQualityFlags,
    ImageMetadata,
    ingest_real_image,
    build_synthetic_metadata,
    write_sidecar,
    read_sidecar,
)
