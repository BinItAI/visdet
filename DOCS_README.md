# Documentation Setup

This project uses [MkDocs](https://www.mkdocs.org/) with the [Material theme](https://squidfunk.github.io/mkdocs-material/) for documentation.

## Local Development

### Prerequisites

Install the documentation dependencies:

```bash
uv pip install mkdocs mkdocs-material mkdocstrings[python] mkdocs-git-revision-date-localized-plugin mkdocs-awesome-pages-plugin
```

### Build Documentation

To build the documentation:

```bash
mkdocs build
```

This generates static HTML files in the `site/` directory.

### Serve Documentation Locally

To preview the documentation locally:

```bash
mkdocs serve
```

Then open http://127.0.0.1:8000 in your browser. The documentation will auto-reload when you make changes.

## Documentation Structure

```
docs/
├── index.md                 # Landing page
├── getting-started/
│   ├── installation.md      # Installation instructions
│   └── quick-start.md       # Quick start guide
├── user-guide/
│   ├── training.md          # Training models
│   ├── inference.md         # Running inference
│   └── configuration.md     # Configuration guide
├── api-reference/
│   └── index.md            # Auto-generated API docs
├── development/
│   ├── contributing.md      # Contributing guidelines
│   └── architecture.md      # Architecture overview
└── assets/
    └── images/             # Documentation images
```

## Writing Documentation

### Markdown Basics

MkDocs uses standard Markdown with some extensions. See the [Material theme documentation](https://squidfunk.github.io/mkdocs-material/reference/) for available features.

### Code Blocks

Use fenced code blocks with syntax highlighting:

\`\`\`python
from visdet import Detector

detector = Detector(config='config.py')
\`\`\`

### Admonitions

Create callout boxes:

\`\`\`markdown
!!! note
    This is a note.

!!! warning
    This is a warning.
\`\`\`

### API Documentation

Use mkdocstrings to include API documentation:

\`\`\`markdown
::: visdet.core.Detector
    options:
      show_source: true
\`\`\`

## Deployment

Documentation is automatically deployed to GitHub Pages via GitHub Actions when changes are pushed to the `master` branch.

The workflow is defined in `.github/workflows/docs.yml`.
