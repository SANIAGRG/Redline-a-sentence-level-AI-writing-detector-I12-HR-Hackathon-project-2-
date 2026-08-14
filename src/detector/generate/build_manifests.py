"""Writes the manifest for whichever Module 3 generated sets already
have a checkpoint on disk (polish, modern-gen, adversarial) --
skips a set silently if its checkpoint doesn't exist yet rather than
failing, since these three runs finish at different times.

`make manifests` runs this. Standalone:
`python -m detector.generate.build_manifests`.
"""

from __future__ import annotations

import logging

from detector.config import RedlineConfig, load_config
from detector.generate.manifests import (
    build_adversarial_manifest,
    build_modern_manifest,
    build_polish_manifest,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BUILDERS = {
    "polish": build_polish_manifest,
    "modern": build_modern_manifest,
    "adversarial": build_adversarial_manifest,
}


def run(config: RedlineConfig) -> None:
    for name, builder in BUILDERS.items():
        checkpoint_path = config.paths.generated / name / "checkpoint.parquet"
        if not checkpoint_path.exists():
            logger.warning(
                "Skipping %s manifest -- no checkpoint at %s yet.", name, checkpoint_path
            )
            continue
        manifest = builder(config)
        out_path = config.paths.generated / name / "manifest.parquet"
        manifest.to_parquet(out_path, index=False)
        logger.info("Wrote %d rows to %s", len(manifest), out_path)


def main() -> None:
    run(load_config())


if __name__ == "__main__":
    main()
