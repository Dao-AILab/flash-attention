
export PYTHONPATH=$PWD:$PYTHONPATH
export TRITON_ALWAYS_COMPILE=1
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export TRITON_SHARED_OPT_PATH=/data/workspace/01triton_work/triton/build/cmake.linux-x86_64-cpython-3.10/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt
export LLVM_BINARY_DIR=/root/.triton/llvm/llvm-064f02da-ubuntu-x64/bin

# python -m pytest -s ./flash_attn/flash_attn_triton_amd/test.py 2>&1 | tee test.log
# pytest -v ./flash_attn/flash_attn_triton_amd/test.py -s 2>&1 | tee test.log
pytest -v ./flash_attn/flash_attn_triton_amd/test.py -s 

echo FINISH TEST... ...
