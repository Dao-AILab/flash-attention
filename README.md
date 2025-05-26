# Flash Attention 2 with Tree Attention

To install:
```bash
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTN_CUDA_ARCHS=80 # amper or adalove
# export FLASH_ATTN_CUDA_ARCHS=90 # hopper 
pip install .
```

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
* Node 2 (`response2`) and Node 3 (`response3`) are also leaf or shallow nodes.

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
