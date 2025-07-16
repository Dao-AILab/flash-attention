#!/bin/bash

set -e

# Flash Attention Minimal Build Script for PHI-1 Reproducer
# Uses subshell to automatically clean up environment variables

# Run in subshell - variables are automatically cleaned up when it exits
(
    # Set minimal build flags for PHI-1 reproducer
    export PYTHONBREAKPOINT="pdbp.set_trace"
    export FLASH_ATTENTION_DISABLE_BACKWARD=FALSE
    export FLASH_ATTENTION_DISABLE_SPLIT=FALSE
    export FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE
    export FLASH_ATTENTION_DISABLE_LOCAL=FALSE
    export FLASH_ATTENTION_DISABLE_CLUSTER=TRUE
    export FLASH_ATTENTION_DISABLE_VARLEN=FALSE
    export FLASH_ATTENTION_DISABLE_PACKGQA=FALSE
    export FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE
    export FLASH_ATTENTION_DISABLE_APPENDKV=FALSE
    export FLASH_ATTENTION_DISABLE_FP8=FALSE
    export FLASH_ATTENTION_DISABLE_FP16=TRUE
    export FLASH_ATTENTION_DISABLE_FP32=TRUE

    # Keep only 64-dim heads for PHI-1
    export FLASH_ATTENTION_DISABLE_HDIM96=TRUE
    export FLASH_ATTENTION_DISABLE_HDIM128=TRUE
    export FLASH_ATTENTION_DISABLE_HDIM192=TRUE
    export FLASH_ATTENTION_DISABLE_HDIM256=FALSE
    
    echo "Environment variables set for minimal build..."
    
    # Install flash-attention
    # python setup.py install
    # python -m pytest test_flash_attn_torch_compile.py --tb=line -x -rs -sv
    python -m pytest test_flash_attn.py --tb=line

)