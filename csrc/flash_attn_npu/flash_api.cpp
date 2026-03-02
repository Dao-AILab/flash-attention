// Copyright (c) 2023, flash-attention
// All rights reserved.
//
// 该文件实现了FlashAttention的Python API接口
// 主要包含了带KV缓存的多头注意力前向传播函数

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "mha_fwd_kvcache.cpp"
#include "tilingdata.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "acl/acl.h"
#include "runtime/rt_ffts.h"
#include "kernel_common.hpp"
#include "kernel_operator.h"
#include "tiling/platform/platform_ascendc.h"

// 获取Q序列的N维度分块大小
// 该函数用于计算Q序列在N维度（头维度）上的最优分块大小
// qSeqlen: Q序列的长度（时间维度）
// groupSize: 注意力头的分组大小（用于分组查询注意力GQA）
// 返回值: Q序列在N维度上的分块大小
uint32_t GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize)
{
    uint32_t qRowNumCeil = Q_TILE_CEIL;  // Q序列的行分块大小上限，从kernel_common.hpp中定义
    
    // 计算QN块的分块大小，确保能被N_SPLIT_HELPER整除，以优化并行计算
    uint32_t qNBlockTile = (qSeqlen != 0) ?
        (qRowNumCeil / qSeqlen) / N_SPLIT_HELPER * N_SPLIT_HELPER : Q_TILE_CEIL;
    
    // 确保分块大小不超过groupSize（分组查询注意力的头分组大小），且至少为1
    qNBlockTile = std::min(qNBlockTile, groupSize);
    qNBlockTile = std::max(qNBlockTile, static_cast<uint32_t>(1));
    
    return qNBlockTile;
}

// 获取Q序列的S维度分块大小
// 该函数用于计算Q序列在S维度（序列长度维度）上的分块大小
// kvSeqlen: KV序列的长度（上下文窗口大小）
// 返回值: Q序列在S维度上的分块大小
// 注意：当前实现简单返回Q_TILE_CEIL，未来可根据KV序列长度进行优化
uint32_t GetQSBlockTile(int64_t kvSeqlen)
{
    // 目前简单返回Q_TILE_CEIL，可根据KV序列长度进行扩展优化
    uint32_t qSBlockTile = Q_TILE_CEIL;
    return qSBlockTile;
}

// 带KV缓存的多头注意力前向传播函数
// q: 查询张量，形状为 [batch_size, seqlen_q, num_heads, head_size]
// kcache: K缓存张量，形状为 [batch_size_c, seqlen_k, num_heads_k, head_size] 或 [num_blocks, page_block_size, num_heads_k, head_size]（如果有block_table）
// vcache: V缓存张量，形状与kcache相同
// k_: 新的K张量，形状为 [batch_size, seqlen_knew, num_heads_k, head_size]（可选）
// v_: 新的V张量，形状与k_相同（可选）
// seqlens_k_: K序列长度张量，形状为 [batch_size]（可选）
// rotary_cos_: 旋转位置编码的余弦张量，形状为 [seqlen_ro, (rotary_dim / 2)]（可选）
// rotary_sin_: 旋转位置编码的正弦张量，形状与rotary_cos_相同（可选）
// cache_batch_idx_: 缓存批次索引张量（可选）
// leftpad_k_: 左填充K张量，形状为 [batch_size]（可选）
// block_table_: 块表张量，形状为 [batch_size, max_num_blocks_per_seq]（可选，用于分页KV缓存）
// alibi_slopes_: ALiBi斜率张量，形状为 [num_heads] 或 [batch_size, num_heads]（可选）
// out_: 输出张量，形状为 [batch_size, seqlen_q, num_heads, head_size]（可选）
// softmax_scale: Softmax缩放因子
// is_causal: 是否使用因果掩码
// window_size_left: 窗口左侧大小（用于滑动窗口注意力）
// window_size_right: 窗口右侧大小（用于滑动窗口注意力）
// softcap: Softmax上限值
// is_rotary_interleaved: 旋转位置编码是否交错
// num_splits: 分割数量
std::vector<at::Tensor>
mha_fwd_kvcache(at::Tensor &q,                 // batch_size x seqlen_q x num_heads x head_size
                const at::Tensor &kcache,            // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                const at::Tensor &vcache,            // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                std::optional<const at::Tensor> &k_, // batch_size x seqlen_knew x num_heads_k x head_size
                std::optional<const at::Tensor> &v_, // batch_size x seqlen_knew x num_heads_k x head_size
                std::optional<const at::Tensor> &seqlens_k_, // batch_size
                std::optional<const at::Tensor> &rotary_cos_, // seqlen_ro x (rotary_dim / 2)
                std::optional<const at::Tensor> &rotary_sin_, // seqlen_ro x (rotary_dim / 2)
                std::optional<const at::Tensor> &cache_batch_idx_, // indices to index into the KV cache
                std::optional<const at::Tensor> &leftpad_k_, // batch_size
                std::optional<at::Tensor> &block_table_, // batch_size x max_num_blocks_per_seq
                std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
                std::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x head_size
                const float softmax_scale,
                bool is_causal,
                int window_size_left,
                int window_size_right,
                const float softcap,
                bool is_rotary_interleaved,   // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
                int num_splits
                )
{
    // 获取当前NPU流
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    // 创建CPU上的分块数据张量
    at::Tensor tiling_cpu_tensor = at::empty({1024}, at::device(c10::kCPU).dtype(at::kByte));

    // 解释分块数据指针
    FAInferTilingData* tiling_cpu_ptr = reinterpret_cast<FAInferTilingData*>(tiling_cpu_tensor.data_ptr<uint8_t>());
    // 获取可用的AI Core数量
    uint32_t blockDim = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    
    // 初始化张量变量
    at::Tensor seqlens_k, block_table, out;
    at::Tensor k, v, rotary_cos, rotary_sin, cache_batch_idx, alibi_slopes;
    // 检查是否为bfloat16类型
    bool is_bf16 = q.dtype() == torch::kBFloat16;
    // 检查是否使用分页KV缓存
    const bool paged_KV = block_table_.has_value();
    // 处理可选参数
    if (seqlens_k_.has_value()) {
        seqlens_k = seqlens_k_.value();
    }
    if (k_.has_value()) {
        k = k_.value();
    }
    if (v_.has_value()) {
        v = v_.value();
    }
    if (rotary_cos_.has_value()) {
        rotary_cos = rotary_cos_.value();
    }
    if (rotary_sin_.has_value()) {
        rotary_sin = rotary_sin_.value();
    }
    if (cache_batch_idx_.has_value()) {
        cache_batch_idx = cache_batch_idx_.value();
    }
    if (alibi_slopes_.has_value()) {
        alibi_slopes = alibi_slopes_.value();
    }
    if (paged_KV) {
        block_table = block_table_.value();
    }
    // 处理输出张量
    if (out_.has_value()) {
        out = out_.value();
    }  else {
        // 如果没有提供输出张量，则创建一个与q相同形状的张量
        out = torch::empty_like(q);
    }
    
    // 获取q的形状信息
    const auto sizes = q.sizes();
    const int batch_size = sizes[0];     // 批次大小
    int seqlen_q = sizes[1];             // 查询序列长度
    int num_heads = sizes[2];            // 注意力头数量
    const int head_size_og = sizes[3];   // 原始头维度大小
    
    // 获取分页KV缓存相关参数
    const int max_num_blocks_per_seq = !paged_KV ? 0 : block_table.size(1);  // 每个序列的最大块数
    const int num_blocks = !paged_KV ? 0 : kcache.size(0);                  // 总块数
    const int page_block_size = !paged_KV ? 128 : kcache.size(1);           // 页块大小（默认128）
    
    // 获取KV头的数量
    const int num_heads_k = kcache.size(2);
    // 获取K序列长度的CPU指针
    int64_t* seqlens_k_cpu = static_cast<int64_t *>(seqlens_k.data_ptr());
    
    // 设置分块数据
    tiling_cpu_ptr->set_batch(static_cast<uint32_t>(batch_size));                 // 设置批次大小
    tiling_cpu_ptr->set_numHeads(static_cast<uint32_t>(num_heads));              // 设置注意力头数量
    tiling_cpu_ptr->set_kvHeads(static_cast<uint32_t>(num_heads_k));             // 设置KV头数量
    tiling_cpu_ptr->set_embeddingSize(static_cast<uint32_t>(head_size_og));      // 设置查询嵌入维度
    tiling_cpu_ptr->set_embeddingSizeV(static_cast<uint32_t>(head_size_og));     // 设置值嵌入维度
    tiling_cpu_ptr->set_numBlocks(static_cast<uint32_t>(num_blocks));            // 设置块数量
    tiling_cpu_ptr->set_blockSize(static_cast<uint32_t>(page_block_size));       // 设置块大小
    tiling_cpu_ptr->set_maxNumBlocksPerBatch(static_cast<uint32_t>(max_num_blocks_per_seq));  // 设置每批次最大块数
    tiling_cpu_ptr->set_maskType(static_cast<uint32_t>(is_causal));              // 设置掩码类型（因果掩码或无掩码）
    tiling_cpu_ptr->set_scaleValue(softmax_scale);                               // 设置Softmax缩放因子
    tiling_cpu_ptr->set_maxQSeqlen(seqlen_q);                                    // 设置最大查询序列长度
    
    // 计算工作区大小
    uint64_t WORKSPACE_BLOCK_SIZE_DB = 128 * 512;  // 双缓冲工作区块大小（128x512）
    uint64_t PRELANCH_NUM = 3;                     // 预启动任务数量（用于流水线并行）
    uint64_t mm1OutSize = static_cast<uint64_t>(blockDim) * WORKSPACE_BLOCK_SIZE_DB * 4 * PRELANCH_NUM;  // QK矩阵乘法输出大小（4字节/元素）
    uint64_t smOnlineOutSize = static_cast<uint64_t>(blockDim) * WORKSPACE_BLOCK_SIZE_DB * 2 * PRELANCH_NUM;  // 在线Softmax输出大小（2字节/元素）
    uint64_t mm2OutSize = static_cast<uint64_t>(blockDim) * WORKSPACE_BLOCK_SIZE_DB * 4 * PRELANCH_NUM;  // PV矩阵乘法输出大小（4字节/元素）
    uint64_t UpdateSize = static_cast<uint64_t>(blockDim) * WORKSPACE_BLOCK_SIZE_DB * 4 * PRELANCH_NUM;  // 更新大小（4字节/元素）
    int64_t workSpaceSize = mm1OutSize + smOnlineOutSize + mm2OutSize + UpdateSize;  // 总工作区大小

    // 创建工作区张量
    at::Tensor workspace_tensor = at::empty({workSpaceSize}, at::device(at::kPrivateUse1).dtype(at::kByte));
    // 创建Softmax对数和张量
    at::Tensor softmaxlse = at::empty({batch_size, seqlen_q, num_heads}, at::device(at::kPrivateUse1).dtype(at::kFloat));
    // 初始化对数和为负无穷
    softmaxlse.fill_(std::numeric_limits<float>::infinity()); 

    // 更新分块数据中的工作区大小信息
    tiling_cpu_ptr->set_mm1OutSize(mm1OutSize);
    tiling_cpu_ptr->set_smOnlineOutSize(smOnlineOutSize);
    tiling_cpu_ptr->set_mm2OutSize(mm2OutSize);
    tiling_cpu_ptr->set_UpdateSize(UpdateSize);
    tiling_cpu_ptr->set_workSpaceSize(workSpaceSize);

    // 计算总任务数量
    uint32_t totalTaskNum = 0;  // 总任务数量
    uint32_t groupSize = num_heads / num_heads_k;  // 每组的头数（用于分组查询注意力GQA）
    
    // 遍历每个批次，计算每个批次的任务数量
    for (int32_t batchIdx = 0; batchIdx < batch_size; batchIdx++) {
        uint64_t qSeqlen = seqlen_q;  // 当前批次的查询序列长度
        uint64_t kvSeqlen = *(seqlens_k_cpu + batchIdx);  // 当前批次的KV序列长度
        
        // 获取当前批次的QN块分块大小（N维度：头维度）
        uint64_t curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
        // 计算每组的QN块数量（向上取整）
        uint64_t qNBlockNumPerGroup = (groupSize + curQNBlockTile - 1) / curQNBlockTile;
        // 计算当前批次的QN块总数（组数 × KV头数）
        uint64_t curQNBlockNum = qNBlockNumPerGroup * num_heads_k;
        
        // 获取当前批次的QS块分块大小（S维度：序列长度维度）
        uint64_t curQSBlockTile = GetQSBlockTile(kvSeqlen);
        // 计算当前批次的QS块数量（向上取整）
        uint64_t curQSBlockNum = (qSeqlen + curQSBlockTile - 1) / curQSBlockTile;
        
        // 计算当前批次的任务数量（QN块数 × QS块数）
        uint64_t curTaskNum = curQNBlockNum * curQSBlockNum;
        // 设置第一批次的任务数量（用于内核调度）
        if (batchIdx == 0) {
            tiling_cpu_ptr->set_firstBatchTaskNum(curTaskNum);
        }
        // 累加到总任务数量
        totalTaskNum += curTaskNum;
    }
    // 设置总任务数量到分块数据结构中
    tiling_cpu_ptr->set_totalTaskNum(totalTaskNum);
    // 处理因果掩码
    at::Tensor mask_gpu_tensor;  // 因果掩码张量（NPU上）
    if (is_causal) {
        // 创建CPU上的掩码张量（固定大小2048×2048）
        at::Tensor mask_cpu_tensor = at::empty({2048, 2048}, at::device(c10::kCPU).dtype(at::kByte));
        // 创建上三角掩码（对角线以上元素为1，表示未来位置不可见）
        mask_cpu_tensor = at::triu(at::ones_like(mask_cpu_tensor), 1);
        // 将掩码张量从CPU传输到NPU设备
        mask_gpu_tensor = mask_cpu_tensor.to(at::Device(at::kPrivateUse1));
    }
    
    // 将分块数据和序列长度张量传输到NPU
    at::Tensor tiling_gpu_tensor = tiling_cpu_tensor.to(at::Device(at::kPrivateUse1));
    at::Tensor seqlenk_gpu_tensor = seqlens_k.to(at::Device(at::kPrivateUse1));
    
    // 获取FFTS控制地址
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    rtError_t error = rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    
    // 获取各张量的设备指针
    auto qDevice = static_cast<uint8_t *>(const_cast<void *>(q.storage().data()));
    auto kDevice = static_cast<uint8_t *>(const_cast<void *>(kcache.storage().data()));
    auto vDevice = static_cast<uint8_t *>(const_cast<void *>(vcache.storage().data()));
    uint8_t * blockTableDevice = nullptr;
    uint8_t * maskDevice = nullptr;
    
    // 处理分页KV缓存的块表
    if (paged_KV) {
        blockTableDevice = static_cast<uint8_t *>(const_cast<void *>(block_table.storage().data()));
    }
    // 处理因果掩码
    if (is_causal) {
        maskDevice = static_cast<uint8_t *>(const_cast<void *>(mask_gpu_tensor.storage().data()));
    }
    
    // 获取输出张量、序列长度张量、工作区张量、分块数据张量和Softmax对数和张量的设备指针
    auto oDevice = static_cast<uint8_t *>(const_cast<void *>(out.storage().data()));
    auto qSeqDevice = static_cast<uint8_t *>(const_cast<void *>(seqlenk_gpu_tensor.storage().data()));
    auto kvSeqDevice = static_cast<uint8_t *>(const_cast<void *>(seqlenk_gpu_tensor.storage().data()));
    auto workspaceDevice = static_cast<uint8_t *>(const_cast<void *>(workspace_tensor.storage().data()));
    auto tilingDevice = static_cast<uint8_t *>(const_cast<void *>(tiling_gpu_tensor.storage().data()));
    auto softmaxLseDevice = static_cast<uint8_t *>(const_cast<void *>(softmaxlse.storage().data()));
    // 根据数据类型和配置选择合适的内核函数
    if (is_bf16) {  // bfloat16数据类型
        if (paged_KV) {  // 分页KV缓存模式
            if (is_causal) {  // 启用因果掩码
                // 调用带因果掩码的分页KV缓存bfloat16内核
                SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, true, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseModeT::OUT_ONLY>
                    <<<blockDim, nullptr, aclStream>>>(
                        fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                        qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
            } else {  // 无掩码
                // 调用无掩码的分页KV缓存bfloat16内核
                SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, true, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseModeT::OUT_ONLY>
                    <<<blockDim, nullptr, aclStream>>>(
                        fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                        qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
            }
        } else {  // 非分页KV缓存模式
            if (is_causal) {  // 启用因果掩码
                // 调用带因果掩码的非分页KV缓存bfloat16内核
                SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, false, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseModeT::OUT_ONLY>
                    <<<blockDim, nullptr, aclStream>>>(
                        fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                        qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
            } else {  // 无掩码
                // 调用无掩码的非分页KV缓存bfloat16内核
                SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, false, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseModeT::OUT_ONLY>
                    <<<blockDim, nullptr, aclStream>>>(
                        fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                        qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
            }
        }
    } else {  // half (fp16)数据类型
        if (paged_KV) {  // 分页KV缓存模式
            if (is_causal) {  // 启用因果掩码
                // 调用带因果掩码的分页KV缓存half内核
                SplitFuse::FAInfer<half, half, float, true, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseModeT::OUT_ONLY>
                    <<<blockDim, nullptr, aclStream>>>(
                        fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                        qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
            } else {  // 无掩码
                // 调用无掩码的分页KV缓存half内核
                SplitFuse::FAInfer<half, half, float, true, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseModeT::OUT_ONLY>
                    <<<blockDim, nullptr, aclStream>>>(
                        fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                        qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
            }
        } else {  // 非分页KV缓存模式
            if (is_causal) {  // 启用因果掩码
                // 调用带因果掩码的非分页KV缓存half内核
                SplitFuse::FAInfer<half, half, float, false, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseModeT::OUT_ONLY>
                    <<<blockDim, nullptr, aclStream>>>(
                        fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                        qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
            } else {  // 无掩码
                // 调用无掩码的非分页KV缓存half内核
                SplitFuse::FAInfer<half, half, float, false, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseModeT::OUT_ONLY>
                    <<<blockDim, nullptr, aclStream>>>(
                        fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                        qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
            }
        }
    }
    
    // 返回输出张量和Softmax对数和张量
    return {out, softmaxlse};
}

// 定义Python模块
PYBIND11_MODULE(flash_attn_2_cuda, m)
{
    m.doc() = "FlashAttention";
    // 导出fwd_kvcache函数到Python
    m.def("fwd_kvcache", &mha_fwd_kvcache, "Forward pass, with KV-cache");
}