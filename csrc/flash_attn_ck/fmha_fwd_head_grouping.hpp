/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include "fmha_fwd.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <utility>

namespace flash::ck_fmha_head_grouping {

inline bool log_enabled()
{
    const char* env = std::getenv("CK_TILE_FMHA_HEAD_GROUP_LOG");
    return env != nullptr && std::atoi(env) == 1;
}

inline bool disabled_by_env()
{
    const char* env_disable = std::getenv("CK_TILE_FMHA_DISABLE_HEAD_GROUPING");
    return env_disable != nullptr && std::atoi(env_disable) == 1;
}

inline std::string trim_gfx_arch(const char* arch_name)
{
    if(arch_name == nullptr)
        return {};
    std::string arch = arch_name;
    const auto pos   = arch.find(':');
    if(pos != std::string::npos)
        arch = arch.substr(0, pos);
    return arch;
}

inline bool is_rdna_arch(const std::string& arch)
{
    return arch.rfind("gfx11", 0) == 0 || arch.rfind("gfx12", 0) == 0;
}

inline bool is_decimal_string(const std::string& s)
{
    if(s.empty())
        return false;
    return std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c) != 0; });
}

inline std::optional<long long> read_property_value(const std::string& filepath,
                                                    const std::string& key)
{
    std::ifstream fs(filepath);
    if(!fs.is_open())
        return std::nullopt;

    std::string k, v;
    while(fs >> k >> v)
    {
        if(k == key)
        {
            try
            {
                return std::stoll(v, nullptr, 0);
            }
            catch(...)
            {
                return std::nullopt;
            }
        }
        std::string rest;
        std::getline(fs, rest);
    }
    return std::nullopt;
}

struct kfd_device_location
{
    int domain      = 0;
    int location_id = 0;
};

inline std::optional<kfd_device_location> get_current_kfd_location()
{
    int device = 0;
    if(hipGetDevice(&device) != hipSuccess)
        return std::nullopt;

    char bdf[64] = {};
    if(hipDeviceGetPCIBusId(bdf, sizeof(bdf), device) == hipSuccess)
    {
        unsigned int domain = 0, bus = 0, dev = 0, fn = 0;
        if(std::sscanf(bdf, "%x:%x:%x.%x", &domain, &bus, &dev, &fn) == 4)
        {
            return kfd_device_location{
                static_cast<int>(domain),
                static_cast<int>(((bus & 0xff) << 8) | ((dev & 0x1f) << 3) | (fn & 0x7))};
        }
    }

    hipDeviceProp_t props{};
    if(hipGetDeviceProperties(&props, device) != hipSuccess)
        return std::nullopt;

    return kfd_device_location{
        props.pciDomainID, ((props.pciBusID & 0xff) << 8) | ((props.pciDeviceID & 0x1f) << 3)};
}

inline std::optional<std::string> find_matching_kfd_node(const kfd_device_location& loc)
{
    constexpr const char* kKfdNodesDir = "/sys/class/kfd/kfd/topology/nodes";
    DIR* dir                            = opendir(kKfdNodesDir);
    if(dir == nullptr)
        return std::nullopt;

    std::optional<std::string> matched;
    while(auto* ent = readdir(dir))
    {
        const std::string node_name(ent->d_name);
        if(!is_decimal_string(node_name))
            continue;

        const std::string prop_path = std::string(kKfdNodesDir) + "/" + node_name + "/properties";
        const auto location_val     = read_property_value(prop_path, "location_id");
        if(!location_val.has_value() || static_cast<int>(*location_val) != loc.location_id)
            continue;

        const auto domain_val = read_property_value(prop_path, "domain");
        if(domain_val.has_value() && static_cast<int>(*domain_val) != loc.domain)
            continue;

        matched = node_name;
        break;
    }

    closedir(dir);
    return matched;
}

inline size_t read_kfd_node_l3_bytes(const std::string& node_name)
{
    const std::string caches_dir = "/sys/class/kfd/kfd/topology/nodes/" + node_name + "/caches";
    DIR* dir                      = opendir(caches_dir.c_str());
    if(dir == nullptr)
        return 0;

    size_t l3_kb = 0;
    while(auto* ent = readdir(dir))
    {
        const std::string cache_name(ent->d_name);
        if(!is_decimal_string(cache_name))
            continue;

        const std::string prop_path = caches_dir + "/" + cache_name + "/properties";
        const auto level_val        = read_property_value(prop_path, "level");
        if(!level_val.has_value() || *level_val != 3)
            continue;

        const auto size_val = read_property_value(prop_path, "size");
        if(!size_val.has_value() || *size_val <= 0)
            continue;

        l3_kb = std::max(l3_kb, static_cast<size_t>(*size_val));
    }

    closedir(dir);
    return l3_kb * 1024ull;
}

inline size_t get_kfd_sysfs_llc_cache_bytes()
{
    const auto loc = get_current_kfd_location();
    if(!loc.has_value())
        return 0;

    const auto node = find_matching_kfd_node(*loc);
    if(!node.has_value())
        return 0;

    return read_kfd_node_l3_bytes(*node);
}

inline size_t get_default_llc_cache_bytes_for_arch(const std::string& arch)
{
    if(arch == "gfx1100")
        return 96ull * 1024ull * 1024ull;
    if(arch == "gfx1101")
        return 64ull * 1024ull * 1024ull;
    if(arch == "gfx1102")
        return 32ull * 1024ull * 1024ull;
    if(arch == "gfx1151")
        return 32ull * 1024ull * 1024ull;
    if(arch == "gfx1200")
        return 32ull * 1024ull * 1024ull;
    if(arch == "gfx1201")
        return 64ull * 1024ull * 1024ull;
    return 0;
}

inline size_t resolve_llc_cache_bytes_uncached(const std::string& arch)
{
    constexpr size_t kMinValidKfdLlcBytes = 32ull * 1024ull;

    const size_t kfd_llc_bytes = get_kfd_sysfs_llc_cache_bytes();
    if(kfd_llc_bytes >= kMinValidKfdLlcBytes)
        return kfd_llc_bytes;

    const size_t default_cache_bytes = get_default_llc_cache_bytes_for_arch(arch);
    if(default_cache_bytes > 0)
        return default_cache_bytes;

    return 0;
}

inline size_t get_llc_cache_bytes(const std::string& arch)
{
    // Single-GPU assumption: resolve once and reuse.
    static const size_t resolved_llc_bytes = [&]() -> size_t {
        const char* env_llc_mb = std::getenv("CK_TILE_FMHA_LLC_CACHE_MB");
        if(env_llc_mb != nullptr)
        {
            const int mb = std::atoi(env_llc_mb);
            if(mb > 0)
                return static_cast<size_t>(mb) * 1024ull * 1024ull;
        }
        return resolve_llc_cache_bytes_uncached(arch);
    }();

    return resolved_llc_bytes;
}

inline std::optional<ck_tile::index_t> get_head_group_size(ck_tile::index_t nhead_q,
                                                            ck_tile::index_t nhead_k,
                                                            ck_tile::index_t batch,
                                                            ck_tile::index_t seqlen_k,
                                                            ck_tile::index_t hdim_q,
                                                            ck_tile::index_t hdim_v,
                                                            size_t elem_bytes_k,
                                                            size_t elem_bytes_v)
{
    if(disabled_by_env())
        return std::nullopt;

    int device = 0;
    if(hipGetDevice(&device) != hipSuccess)
        return std::nullopt;
    hipDeviceProp_t props{};
    if(hipGetDeviceProperties(&props, device) != hipSuccess)
        return std::nullopt;

    const std::string arch = trim_gfx_arch(props.gcnArchName);
    if(arch.empty() || !is_rdna_arch(arch))
        return std::nullopt;

    const size_t llc_bytes = get_llc_cache_bytes(arch);
    if(llc_bytes == 0)
        return std::nullopt;

    if(nhead_k <= 0 || nhead_q <= 0 || (nhead_q % nhead_k) != 0)
        return std::nullopt;

    if(seqlen_k <= 0 || hdim_q <= 0 || hdim_v <= 0 || batch <= 0)
        return std::nullopt;
    static_cast<void>(batch); // heuristic does not use batch in trigger condition

    const size_t kv_bytes_per_head =
        static_cast<size_t>(seqlen_k) *
        (static_cast<size_t>(hdim_q) * elem_bytes_k + static_cast<size_t>(hdim_v) * elem_bytes_v);
    if(kv_bytes_per_head == 0)
        return std::nullopt;

    // Hybrid policy:
    // - large LLC GPUs (>= 64MB): slightly more cache-resident grouping
    // - smaller LLC GPUs: Triton policy
    constexpr size_t kLargeLlcThresholdBytes = 64ull * 1024ull * 1024ull;
    const bool is_large_llc                  = llc_bytes >= kLargeLlcThresholdBytes;
    const long double llc_utilization        = is_large_llc ? 0.85L : 1.0L;
    const long double threshold_ratio        = is_large_llc ? 1.3L : 1.5L;
    const size_t target_bytes =
        static_cast<size_t>(static_cast<long double>(llc_bytes) * llc_utilization);
    if(target_bytes == 0)
        return std::nullopt;

    const size_t total_kv_bytes = static_cast<size_t>(nhead_q) * kv_bytes_per_head;
    if(static_cast<long double>(total_kv_bytes) <
       static_cast<long double>(target_bytes) * threshold_ratio)
        return std::nullopt;

    ck_tile::index_t group = static_cast<ck_tile::index_t>(target_bytes / kv_bytes_per_head);
    if(group < 1)
        group = 1;

    const ck_tile::index_t min_group_size = std::max<ck_tile::index_t>(1, nhead_q / 16);
    if(group < min_group_size)
        group = min_group_size;

    // Cap the number of groups to avoid excessive launch overhead.
    constexpr ck_tile::index_t kMaxGroups = 8;
    const ck_tile::index_t min_group_for_max_groups =
        ck_tile::integer_divide_ceil(nhead_q, kMaxGroups);
    if(group < min_group_for_max_groups)
        group = min_group_for_max_groups;

    const ck_tile::index_t gqa_ratio = nhead_q / nhead_k;
    if(gqa_ratio > 1)
    {
        group = ((group + gqa_ratio - 1) / gqa_ratio) * gqa_ratio;
    }

    group = std::min(group, nhead_q);
    if(group >= nhead_q)
        return std::nullopt;

    return group;
}

struct GroupedLaunchElemBytes
{
    size_t q_elem_bytes    = 0;
    size_t k_elem_bytes    = 0;
    size_t v_elem_bytes    = 0;
    size_t o_elem_bytes    = 0;
    size_t lse_elem_bytes  = 0;
    size_t rand_elem_bytes = 0;
    size_t bias_elem_bytes = 0;
};

inline const void* offset_const_ptr(const void* p, ck_tile::index_t elem_offset, size_t elem_bytes)
{
    if(p == nullptr || elem_offset == 0)
        return p;
    return static_cast<const void*>(static_cast<const char*>(p) + elem_offset * elem_bytes);
}

inline void* offset_mut_ptr(void* p, ck_tile::index_t elem_offset, size_t elem_bytes)
{
    if(p == nullptr || elem_offset == 0)
        return p;
    return static_cast<void*>(static_cast<char*>(p) + elem_offset * elem_bytes);
}

template <typename RunFn>
float run_fwd_head_grouped(const ck_tile::stream_config& stream_config,
                           const fmha_fwd_traits& traits,
                           const fmha_fwd_args& args,
                           ck_tile::index_t nhead_q_total,
                           ck_tile::index_t nhead_k_total,
                           ck_tile::index_t group_size,
                           const GroupedLaunchElemBytes& elem_bytes,
                           RunFn&& run_fn)
{
    if(nhead_k_total <= 0 || nhead_q_total <= 0 || (nhead_q_total % nhead_k_total) != 0)
        return -1.0f;

    const ck_tile::index_t gqa_ratio = nhead_q_total / nhead_k_total;
    float total_time                  = 0.0f;
    bool first_group                  = true;
    for(ck_tile::index_t start_h = 0; start_h < nhead_q_total; start_h += group_size)
    {
        const ck_tile::index_t end_h     = std::min(start_h + group_size, nhead_q_total);
        const ck_tile::index_t heads_q   = end_h - start_h;
        const ck_tile::index_t start_h_k = start_h / gqa_ratio;
        const ck_tile::index_t end_h_k   = ck_tile::integer_divide_ceil(end_h, gqa_ratio);
        const ck_tile::index_t heads_k   = end_h_k - start_h_k;

        fmha_fwd_args fmha_args   = args;
        fmha_args.nhead_q         = heads_q;
        fmha_args.nhead_k         = heads_k;
        fmha_args.num_head_q_total = nhead_q_total;
        fmha_args.head_start       = start_h;

        fmha_args.q_ptr =
            offset_const_ptr(fmha_args.q_ptr, start_h * fmha_args.nhead_stride_q, elem_bytes.q_elem_bytes);
        fmha_args.k_ptr = offset_const_ptr(
            fmha_args.k_ptr, start_h_k * fmha_args.nhead_stride_k, elem_bytes.k_elem_bytes);
        fmha_args.v_ptr = offset_const_ptr(
            fmha_args.v_ptr, start_h_k * fmha_args.nhead_stride_v, elem_bytes.v_elem_bytes);
        fmha_args.o_ptr =
            offset_mut_ptr(fmha_args.o_ptr, start_h * fmha_args.nhead_stride_o, elem_bytes.o_elem_bytes);

        if(fmha_args.bias_ptr != nullptr)
        {
            fmha_args.bias_ptr = offset_const_ptr(
                fmha_args.bias_ptr, start_h * fmha_args.nhead_stride_bias, elem_bytes.bias_elem_bytes);
        }
        if(fmha_args.lse_ptr != nullptr)
        {
            fmha_args.lse_ptr = offset_mut_ptr(
                fmha_args.lse_ptr, start_h * fmha_args.nhead_stride_lse, elem_bytes.lse_elem_bytes);
        }
        if(fmha_args.rand_val_ptr != nullptr)
        {
            fmha_args.rand_val_ptr = offset_mut_ptr(
                fmha_args.rand_val_ptr,
                start_h * fmha_args.nhead_stride_randval,
                elem_bytes.rand_elem_bytes);
        }
        if(fmha_args.sink_ptr != nullptr)
        {
            fmha_args.sink_ptr = offset_const_ptr(fmha_args.sink_ptr, start_h, sizeof(float));
        }
        if(fmha_args.q_descale_ptr != nullptr)
        {
            fmha_args.q_descale_ptr = offset_const_ptr(
                fmha_args.q_descale_ptr,
                start_h * fmha_args.nhead_stride_q_descale,
                sizeof(float));
        }
        if(fmha_args.k_descale_ptr != nullptr)
        {
            fmha_args.k_descale_ptr = offset_const_ptr(
                fmha_args.k_descale_ptr,
                start_h_k * fmha_args.nhead_stride_k_descale,
                sizeof(float));
        }
        if(fmha_args.v_descale_ptr != nullptr)
        {
            fmha_args.v_descale_ptr = offset_const_ptr(
                fmha_args.v_descale_ptr,
                start_h_k * fmha_args.nhead_stride_v_descale,
                sizeof(float));
        }

        if(log_enabled())
        {
            std::cout << "[LLC Head Grouping] group heads_q=[" << start_h << ", " << end_h
                      << ") heads_k=[" << start_h_k << ", " << end_h_k << ")" << std::endl;
        }

        ck_tile::stream_config sc_group = stream_config;
        if(!first_group)
            sc_group.log_level_ = 0;
        const float t = run_fn(traits, fmha_args, sc_group);
        if(t < 0.0f)
            return t;
        total_time += t;
        first_group = false;
    }
    return total_time;
}

inline void log_grouping_enabled(ck_tile::index_t nhead_q,
                                 ck_tile::index_t nhead_k,
                                 ck_tile::index_t group_size)
{
    if(!log_enabled())
        return;

    int device = 0;
    hipDeviceProp_t props{};
    std::string arch = {};
    size_t llc_bytes = 0;
    if(hipGetDevice(&device) == hipSuccess && hipGetDeviceProperties(&props, device) == hipSuccess)
    {
        arch      = trim_gfx_arch(props.gcnArchName);
        llc_bytes = get_llc_cache_bytes(arch);
    }

    const ck_tile::index_t gqa_ratio = (nhead_k > 0 ? (nhead_q / nhead_k) : 1);
    const ck_tile::index_t n_groups  = ck_tile::integer_divide_ceil(nhead_q, group_size);
    std::cout << "[LLC Head Grouping] enabled" << std::endl;
    std::cout << "[LLC Head Grouping] arch=" << (arch.empty() ? "unknown" : arch)
              << " llc_mb=" << (llc_bytes / (1024ull * 1024ull)) << " nhead_q=" << nhead_q
              << " nhead_k=" << nhead_k << " gqa_ratio=" << gqa_ratio
              << " group_size=" << group_size << " groups=" << n_groups << std::endl;
}

} // namespace flash::ck_fmha_head_grouping
