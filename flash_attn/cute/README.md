# Flash Attention CUTE

## Development Installation

1. Clone the repository (if you haven't already):
   ```bash
   git clone https://github.com/Dao-AILab/flash-attention.git
   cd flash-attention/cute
   ```

2. Install in editable mode with dev dependencies:
   ```bash
   pip install -e "./cute[dev]"
   ```

## Running Tests

```bash
pytest tests/cute/
```

## Linting

```bash
ruff check flash_attn/cute/
```
