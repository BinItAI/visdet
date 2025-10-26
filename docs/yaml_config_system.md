# YAML Configuration System

## Overview

visdet now supports a modern YAML-based configuration system as an alternative to Python dict-based configs. This system provides:

- **Modular component configs** - Reusable YAML files for backbones, necks, datasets, etc.
- **_base_ inheritance** - Compose experiment configs from base configurations
- **$ref resolution** - Reference other YAML files for component composition
- **Pydantic validation** - Optional type-safe validation with clear error messages
- **Backward compatibility** - Python .py configs still work (with deprecation warnings)

## Quick Start

### Loading a YAML Config

```python
from visdet.engine.config import Config

# Load YAML config (with _base_ and $ref support)
cfg = Config.fromfile('configs/experiments/mask_rcnn_swin_tiny_coco.yaml')

# Use the config as before
model = MODELS.build(cfg.model)
```

### Example YAML Config

**Component config** (`configs/components/backbones/swin_tiny.yaml`):
```yaml
type: SwinTransformer
embed_dims: 96
depths: [2, 2, 6, 2]
num_heads: [3, 6, 12, 24]
window_size: 7
```

**Experiment config** (`configs/experiments/mask_rcnn_swin_tiny_coco.yaml`):
```yaml
# Inherit base configurations
_base_:
  - ../components/datasets/coco_instance.yaml
  - ../components/optimizers/adamw_default.yaml
  - ../components/schedules/1x.yaml

# Model with $ref to components
model:
  type: MaskRCNN
  backbone:
    $ref: ../components/backbones/swin_tiny.yaml
  neck:
    $ref: ../components/necks/fpn_256.yaml
  # ... rest of model config

# Override base configs
max_epochs: 24  # Override from 12 (in 1x.yaml)
```

## Key Features

### 1. _base_ Inheritance

Inherit and merge configurations from parent YAML files:

```yaml
_base_:
  - base_config1.yaml
  - base_config2.yaml

# Your overrides here
max_epochs: 24  # Override value from base
```

**How it works:**
1. Load all `_base_` files in order (depth-first)
2. Deep merge all base configs
3. Apply current config on top (overrides base values)

### 2. $ref Resolution

Reference other YAML files for component composition:

```yaml
model:
  backbone:
    $ref: ./backbones/swin_tiny.yaml  # Load from external file
  neck:
    type: FPN  # Or define inline
    out_channels: 256
```

**Rules:**
- `$ref` must be the only key in a dict
- Paths are relative to the current YAML file
- Referenced files can themselves have `_base_` and `$ref`

### 3. Hybrid Approach

You can mix inline definitions and file references:

```yaml
model:
  backbone:
    $ref: ./backbones/swin_tiny.yaml  # Complex component - use $ref
  rpn_head:
    # Simple component - define inline
    type: RPNHead
    in_channels: 256
    loss_cls:
      type: CrossEntropyLoss
      use_sigmoid: true
```

**Guideline:**
- Use `$ref` for complex components that are frequently swapped
- Use inline for simple, stable components

### 4. ConfigDict Attribute Access

Configs support both attribute and dict-style access:

```python
# Attribute access (dot notation)
model_type = cfg.model.type
backbone_dims = cfg.model.backbone.embed_dims

# Dict-style access (bracket notation)
model_type = cfg['model']['type']
backbone_dims = cfg['model']['backbone']['embed_dims']
```

## Config Structure

```
configs/
├── components/           # Reusable component library
│   ├── backbones/
│   │   ├── swin_tiny.yaml
│   │   └── swin_small.yaml
│   ├── necks/
│   │   └── fpn_256.yaml
│   ├── datasets/
│   │   ├── coco_instance.yaml
│   │   └── coco_detection.yaml
│   ├── optimizers/
│   │   └── adamw_default.yaml
│   └── schedules/
│       ├── 1x.yaml
│       └── 2x.yaml
└── experiments/          # Full experiment configs
    └── mask_rcnn_swin_tiny_coco.yaml
```

## Pydantic Validation (Optional)

Enable type-safe validation:

```python
# Load with validation
cfg = Config.fromfile('config.yaml', validate=True)
```

### Auto-generated Schemas

Pydantic schemas are auto-generated from component `__init__` signatures:

```python
from visdet.engine.config import generate_schema_from_class
from visdet.models.backbones.swin import SwinTransformer

# Auto-generate schema
schema = generate_schema_from_class(SwinTransformer)

# Validate config
config_data = {'type': 'SwinTransformer', 'embed_dims': 96, ...}
validated = schema(**config_data)
```

### Manual Schema Overrides

Add custom validation rules with the `@schema_for` decorator:

```python
from pydantic import BaseModel, Field
from visdet.engine.config import schema_for
from visdet.models.backbones.swin import SwinTransformer

@schema_for(SwinTransformer)
class SwinTransformerConfig(BaseModel):
    type: Literal['SwinTransformer'] = 'SwinTransformer'
    embed_dims: int = Field(gt=0, description="Embedding dimensions")
    depths: List[int] = Field(min_items=1, description="Stage depths")
    # ... other fields with validation rules
```

## Migration from Python Configs

### Using Deprecation Warnings

Python `.py` configs still work but show deprecation warnings:

```python
# This works but shows a warning
cfg = Config.fromfile('configs/_base_/models/mask_rcnn_r50_fpn.py')

# Disable the warning if needed
cfg = Config.fromfile('config.py', deprecation_warning=False)
```

### Converting Existing Configs

A migration tool will be provided to convert Python configs to YAML:

```bash
# Coming soon
python tools/convert_config.py \
  --input configs/_base_/models/mask_rcnn_r50_fpn.py \
  --output configs/components/models/ \
  --split  # Split into separate component files
```

## Testing

Run the test suite to verify the YAML system:

```bash
python scripts/test_yaml_simple.py
```

This tests:
- Individual component loading
- _base_ inheritance and overrides
- $ref resolution
- Attribute access
- ConfigDict functionality

## Best Practices

### 1. Component Organization

- Group related components in subdirectories
- Use descriptive filenames (`swin_tiny.yaml` not `config1.yaml`)
- Add comments explaining key parameters

### 2. Base Config Composition

- Order matters in `_base_` list (later files override earlier ones)
- Keep base configs focused (one concern per file)
- Override sparingly in experiment configs

### 3. File References

- Use relative paths for portability
- Keep referenced files close to avoid deep nesting
- Document dependencies with comments

### 4. Inline vs Reference

- Reference: Complex components, frequently swapped configs
- Inline: Simple components, experiment-specific settings

## API Reference

### Config Class

```python
class Config(BaseConfig):
    @staticmethod
    def fromfile(
        filename: Union[str, Path],
        validate: bool = False,
        deprecation_warning: bool = True,
    ) -> "Config":
        """Load config from YAML or Python file."""
```

### YAML Loader

```python
def load_yaml_config(filepath: Union[str, Path]) -> ConfigDict:
    """Load YAML with _base_ and $ref support."""
```

### Schema Generation

```python
def generate_schema_from_class(
    component_cls: Type,
    include_type_field: bool = True,
    type_literal: Optional[str] = None,
) -> Type[BaseModel]:
    """Auto-generate Pydantic schema from class."""

def schema_for(component_cls: Type) -> Callable:
    """Decorator for manual schema overrides."""
```

## Troubleshooting

### Circular Dependencies

**Error:** `Circular dependency detected: A -> B -> A`

**Solution:** Reorganize configs to break the cycle. Use inline definitions instead of `$ref` for one component.

### $ref Not Resolved

**Error:** `$ref must be the only key in a dict`

**Solution:** Ensure `$ref` is the only key:
```yaml
# ✗ Wrong
backbone:
  $ref: ./swin.yaml
  extra_param: value

# ✓ Correct
backbone:
  $ref: ./swin.yaml
```

### Validation Errors

**Error:** `ValidationError: ...`

**Solution:** Check the error message for specific field issues. Disable validation temporarily with `validate=False` to debug.

## Future Enhancements

Planned features:
- [ ] Python-to-YAML migration tool
- [ ] Pydantic schemas for all core components
- [ ] Config visualization and dependency graphs
- [ ] IDE support (autocomplete, validation)
- [ ] Config inheritance from remote URLs

## Credits

The YAML configuration system was designed based on:
- **Config inheritance patterns** - `_base_` inheritance pattern for modular configs
- **Hydra** - Composition and override mechanisms
- **Pydantic** - Type-safe validation
- **Expert consensus** (Gemini 2.5 Pro) - Architecture recommendations
