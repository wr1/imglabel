= Design Brief: imglabel

== Overview

imglabel is a Python-based command-line tool designed for fast, local image labeling of paint damage patches in images. It leverages HSV (Hue, Saturation, Value) color space for perceptual similarity detection, focusing on hue and saturation to identify similar-colored areas, such as yellowy paint damages, while ignoring brightness variations.

== Key Features

- *Interactive Mode*: Users click on a paint damage spot in a displayed image to select and highlight similar patches with red polygon outlines.
- *Non-Interactive Mode*: Apply pre-saved criteria from a YAML file to label images directly.
- *HSV-Based Similarity*: Uses circular hue distances and saturation thresholds for accurate color matching.
- *Clustering and Outlining*: Groups similar patches, dilates to merge nearby clusters, and draws contours.
- *Criteria Persistence*: Saves and loads selection criteria (HSV values and thresholds) to YAML for reuse.

== Architecture

- *Core Module* (`src/imglabel/core.py`): Handles image loading (RGB and HSV), mask computation, clustering, and criteria I/O.
- *CLI Module* (`src/imglabel/cli.py`): Provides command-line interface with argparse, event handling for interactive clicks, and matplotlib for visualization.
- *Dependencies*: Relies on NumPy, Matplotlib, Pillow, SciPy, scikit-image, and PyYAML for image processing and UI.
- *Packaging*: Configured with Hatchling for building, Ruff for linting, and Pytest for testing.

== Current State

The tool is fully functional with HSV-based detection implemented. It supports coarsening images for performance, vectorized operations for efficiency, and modular code structure. Tests cover core functions, and the CLI is installable via `uv pip install -e .`. An admin script automates formatting, checking, committing, and testing.

== Future Enhancements

- Add adjustable thresholds via CLI options.
- Support for batch processing multiple images.
- Export labeled images or masks to files.
- GUI improvements for better usability.
