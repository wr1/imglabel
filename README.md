# imglabel

A fast local image labeling workflow for paint damages.

## Installation

```bash
uv pip install -e .
```

## Usage

### Command-Line Interface

Interactive mode:

```bash
imglabel path/to/image.jpg
```

Click on a paint damage spot to select and highlight similar areas based on HSV color similarity (focusing on hue and saturation).

Apply saved criteria:

```bash
imglabel path/to/image.jpg --criteria criteria.yaml
```

### Graphical User Interface

```bash
imglabel-gui
```

Select a directory containing images. Choose an image from the left list, click on the image to select a paint damage patch, and view highlighted areas. Use the right panel to load/save criteria.

## License

MIT
