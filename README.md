# Flash Attention 2 with Tree Attention

## Install

```bash
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTN_CUDA_ARCHS=80  # For Ampere or Ada Lovelace GPUs
# export FLASH_ATTN_CUDA_ARCHS=90  # For Hopper GPUs
pip install .
```

If the build fails due to missing `nvcc`, ensure the CUDA toolkit is installed and available.
For example, you can install CUDA 12.4 via conda:
```
conda install -c nvidia cuda-toolkit=12.4
```
After installation, verify that `nvcc` is in your PATH:
```
which nvcc
```
If not, you may need to manually add it to PATH:
```
export PATH=$CONDA_PREFIX/bin:$PATH
```


## Usage

Add the following two lines in your code to enable `flash_attn` with tree attention:

```python
from flash_attn.patch_fa_tree_attn import patch_FA_tree_attn
patch_FA_tree_attn()
```

Additionally, for `transformers` models, make sure to enable `attn_implementation=flash_attention_2` during model initialization. Also, provide extra `FlashAttentionKwargs` in the same way as shown in [`patch.py`](https://github.com/efsotr/nano-patch-sequence-pack/blob/main/patch.py) from the nano-patch-sequence-pack repository, as demonstrated [here](https://github.com/efsotr/flash-attention-w-tree-attn/blob/0c43a382841cbc48d7b57d20fbea7a0b7887eaf8/flash_attn/patch_fa_tree_attn.py#L14) in the tree attention patch.

Specifically, set `tree_dfs_order_end_k` and `tree_dfs_order_start_q` according to the following [Example](#Example)

## Tree Attention

Tree attention restricts each token to attend only to its ancestor nodes (including itself) within a hierarchical tree structure. This differs from the self-attention mechanism used in decoder-only models, where each token attends only to itself and all preceding tokens in a linear sequence to preserve causality.

Tree attention is based on a tree structure that is serialized using a depth-first search (DFS) traversal. For each node `i`, the following indices are recorded:

* `tree_dfs_order_start[i]`: the DFS index when node `i` is first visited
* `tree_dfs_order_end[i]`: the highest DFS index among all nodes in the subtree rooted at node `i`

## Ancestor Check in DFS Sequence

After serialization, the ancestor relationship between nodes can be determined as follows: node `k` is an ancestor of node `q` if `k` appears before `q` in the DFS sequence **and**
`tree_dfs_order_start[q] <= tree_dfs_order_end[k]`.

## Example

Assume the input corresponds to the following sequence:

```
[prompt][response1][response2][response3]
```

* Node 0 (`prompt`) is the root, and its subtree spans indices \[0, 3].
* Node 1 (`response1`) is a leaf node, with its subtree limited to \[1, 1].
* Node 2 (`response2`) and Node 3 (`response3`) are also leaf.

The DFS traversal results in:

```
tree_dfs_order_start (per node): [0, 1, 2, 3]
tree_dfs_order_end   (per node): [3, 1, 2, 3]
```

Accordingly, the index arrays for individual tokens are:

```
tree_dfs_order_start_q = [0] * len(prompt) + [1] * len(response1) + [2] * len(response2) + [3] * len(response3)
tree_dfs_order_end_k   = [3] * len(prompt) + [1] * len(response1) + [2] * len(response2) + [3] * len(response3)
```

Here, both `tree_dfs_order_start_q` and `tree_dfs_order_end_k` are 1D tensors aligned with the full token sequence.
