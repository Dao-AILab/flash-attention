# Autotuner

Autotuner can automatically generate the best config for flash-attention kernel with not-implemented headdim qk & headdim v , or different hardware such as nvidia Ampere, Ada Lovelace.

Currently, the autotuner only support flash attention forward. We plan to support backward and forward_split soon.

## Usage

Currently, you need to first install flashattn from source. Then, you can run the autotuner with head-dimensions of qk and v you want to tune. After that, you need to modify/create `csrc/flash_attn/src/flash_fwd_qkdim*_vdim*_sm80.h` with the tuned config. Finally, you need to rebuild the flashattn from source.



The detailed steps are as follows:

- Install flashattn from source
- run ```python autotuner/test_run_tunner.py ``` with problem size you want to tune.
- If the headdim already exists in `csrc/flash_attn/src`, you need to modify `csrc/flash_attn/src/flash_fwd_qkdim*_vdim*_sm80.h` with the tuned best config. If the headdim does not exist, you need to create  `csrc/flash_attn/src/flash_fwd_qkdim*_vdim*_sm80.h`, `csrc/flash_attn/src/flash_bwd_qkdim*_vdim*_sm80.h` with the tuned best config and the corresponding `.cu` files; After that, you need to add the headdim in `headdim.json`.
- Rebuild the flashattn from source.


