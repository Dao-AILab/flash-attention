# Project Rules

## Build & Test Environment

- All build and test commands MUST be executed inside the Docker container `flash_attn_v100`, never on the host.
- Use `docker exec flash_attn_v100 <command>` for all operations.
