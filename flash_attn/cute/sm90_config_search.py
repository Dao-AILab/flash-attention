"""Search feasible SM90 backward configs for given (head_dim, head_dim_v).

Each config specifies tile sizes, swap modes, atom layouts, and staging.
Constraints: GMMA M-dim divisible by 64, register budget, smem budget (224 KB).

Usage:
    python flash_attn/cute/bwd_config_search.py --headdim 128
    python flash_attn/cute/bwd_config_search.py --headdim 192-128
"""

import math


# Hardware limits
SMEM_LIMIT = 224 * 1024  # 228 KB minus ~3 KB for LSE, dPsum, mbarriers
REG_LIMITS = {2: 216, 3: 128}  # per-WG register budget: 2WG=240-24, 3WG=160-32
THREADS_PER_WG = 128


def _divisors(num_wg):
    """Return valid atom_layout_m values (divisors of num_wg)."""
    return [d for d in range(1, num_wg + 1) if num_wg % d == 0]


def _acc_regs(M, N, num_wg):
    """Accumulator registers per thread per WG: M * N / (num_wg * 128)."""
    return M * N // (num_wg * THREADS_PER_WG)


def _check_mma(M, N, num_wg, atom_layout_m, swap_AB):
    """Check MMA feasibility. Returns regs per WG, or None if infeasible.

    GMMA atom M-dim is always 64. Swap exchanges (M, N) and atom layout.
    Requires: eff_m divisible by (atom_layout_m * 64), eff_n by atom_layout_n.
    """
    if swap_AB:
        M, N = N, M
        atom_layout_m = num_wg // atom_layout_m
    atom_layout_n = num_wg // atom_layout_m
    if M % (atom_layout_m * 64) != 0 or N % (atom_layout_n * 8) != 0:
        return None
    return _acc_regs(M, N, num_wg)


def _check_config(
    hdim,
    hdimv,
    tile_m,
    tile_n,
    num_wg,
    SdP_swapAB,
    dKV_swapAB,
    dQ_swapAB,
    AtomLayoutMSdP,
    AtomLayoutNdKV,
    AtomLayoutMdQ,
):
    """Check a single config. Returns dict if feasible, None otherwise."""
    reg_limit = REG_LIMITS[num_wg]

    # MMA feasibility: SdP, dK, dV, dQ
    regs_SdP = _check_mma(tile_m, tile_n, num_wg, AtomLayoutMSdP, SdP_swapAB)
    regs_dK = _check_mma(tile_n, hdim, num_wg, AtomLayoutNdKV, dKV_swapAB)
    regs_dV = _check_mma(tile_n, hdimv, num_wg, AtomLayoutNdKV, dKV_swapAB)
    regs_dQ = _check_mma(tile_m, hdim, num_wg, AtomLayoutMdQ, dQ_swapAB)
    if any(r is None for r in (regs_SdP, regs_dK, regs_dV, regs_dQ)):
        return None

    # Peak regs: max(S+dP, dQ) + dK + dV
    total_regs = max(2 * regs_SdP, regs_dQ) + regs_dK + regs_dV
    if total_regs > reg_limit:
        return None

    # SMEM check
    mma_dkv_is_rs = (
        AtomLayoutMSdP == 1 and AtomLayoutNdKV == num_wg and SdP_swapAB and not dKV_swapAB
    )
    Q_stage = 2
    PdS_stage = 1

    # dO_stage: prefer 2 (matches Q_stage for simpler pipeline), fall back to 1
    for dO_stage in (2, 1):
        sQ = tile_m * hdim * 2 * Q_stage
        sK = tile_n * hdim * 2
        sV = tile_n * hdimv * 2
        sdO = tile_m * hdimv * 2 * dO_stage
        sPdS = tile_m * tile_n * 2 * PdS_stage
        sP = sPdS if not mma_dkv_is_rs else 0
        sdQaccum = tile_m * hdim * 4
        smem = sQ + sK + sV + sdO + sP + sPdS + sdQaccum
        if smem <= SMEM_LIMIT:
            break
    else:
        return None

    # --- SMEM traffic per m_block iteration (per WG) ---
    # MMA operands: A is (64, K_red), B is (N_per_wg, K_red), both bf16.
    # For RS MMA, A comes from registers, only count B.
    def _mma_traffic(M_eff, N_eff, K_red, wg_n, is_rs=False):
        """Total SMEM traffic for one MMA. M_eff/N_eff are after swap.
        num_instr = (M_eff / 64) * wg_n. Each reads A(64, K) and B(N_eff/wg_n, K)."""
        num_instr = (M_eff // 64) * wg_n
        A_per = 64 * K_red * 2 if not is_rs else 0
        B_per = (N_eff // wg_n) * K_red * 2
        return num_instr * (A_per + B_per)

    def _swap(M, N, swap):
        return (N, M) if swap else (M, N)

    # After swap, atom_layout_n for B partitioning:
    def _wg_n(atom_layout_m, swap):
        return atom_layout_m if swap else num_wg // atom_layout_m

    # S = Q @ K^T
    M_eff, N_eff = _swap(tile_m, tile_n, SdP_swapAB)
    wg_n_SdP = _wg_n(AtomLayoutMSdP, SdP_swapAB)
    traffic_S = _mma_traffic(M_eff, N_eff, hdim, wg_n_SdP)
    traffic_dP = _mma_traffic(M_eff, N_eff, hdimv, wg_n_SdP)
    # dV = P^T @ dO
    M_eff, N_eff = _swap(tile_n, hdimv, dKV_swapAB)
    wg_n_dKV = _wg_n(AtomLayoutNdKV, dKV_swapAB)
    traffic_dV = _mma_traffic(M_eff, N_eff, tile_m, wg_n_dKV, is_rs=mma_dkv_is_rs)
    # dK = dS^T @ Q
    M_eff, N_eff = _swap(tile_n, hdim, dKV_swapAB)
    traffic_dK = _mma_traffic(M_eff, N_eff, tile_m, wg_n_dKV, is_rs=mma_dkv_is_rs)
    # dQ = dS @ K
    M_eff, N_eff = _swap(tile_m, hdim, dQ_swapAB)
    wg_n_dQ = _wg_n(AtomLayoutMdQ, dQ_swapAB)
    traffic_dQ = _mma_traffic(M_eff, N_eff, tile_n, wg_n_dQ)

    # R2S stores: P (tile_m x tile_n x 2), dS (same)
    traffic_P_store = tile_m * tile_n * 2 if not mma_dkv_is_rs else 0
    traffic_dS_store = tile_m * tile_n * 2
    # dQ store to smem (f32) + TMA load from smem (f32)
    traffic_dQ_store = tile_m * hdim * 4
    traffic_dQ_tma = tile_m * hdim * 4

    smem_traffic = (
        traffic_S
        + traffic_dP
        + traffic_dV
        + traffic_dK
        + traffic_dQ
        + traffic_P_store
        + traffic_dS_store
        + traffic_dQ_store
        + traffic_dQ_tma
    )

    return dict(
        tile_m=tile_m,
        tile_n=tile_n,
        num_wg=num_wg,
        Q_stage=Q_stage,
        dO_stage=dO_stage,
        PdS_stage=PdS_stage,
        SdP_swapAB=SdP_swapAB,
        dKV_swapAB=dKV_swapAB,
        dQ_swapAB=dQ_swapAB,
        AtomLayoutMSdP=AtomLayoutMSdP,
        AtomLayoutNdKV=AtomLayoutNdKV,
        AtomLayoutMdQ=AtomLayoutMdQ,
        mma_dkv_is_rs=mma_dkv_is_rs,
        regs_SdP=regs_SdP,
        regs_dK=regs_dK,
        regs_dV=regs_dV,
        regs_dQ=regs_dQ,
        total_regs=total_regs,
        reg_limit=reg_limit,
        smem_bytes=smem,
        smem_kb=smem / 1024,
        smem_traffic=smem_traffic,
        smem_traffic_kb=smem_traffic / 1024,
        smem_traffic_per_block=smem_traffic / (tile_m * tile_n),
    )


def find_feasible_bwd_configs(
    head_dim,
    head_dim_v=None,
    tile_m_choices=(64, 80, 96, 112, 128),
    tile_n_choices=(64, 80, 96, 112, 128),
):
    """Find all feasible SM90 backward configs for given head dimensions."""
    if head_dim_v is None:
        head_dim_v = head_dim
    hdim = int(math.ceil(head_dim / 32) * 32)
    hdimv = int(math.ceil(head_dim_v / 32) * 32)

    results = []
    for num_wg in (2, 3):
        divs = _divisors(num_wg)
        for tile_m in tile_m_choices:
            for tile_n in tile_n_choices:
                for SdP_swapAB in (False, True):
                    # SdP eff_m: tile_n if swap else tile_m — must be % 64
                    if (tile_n if SdP_swapAB else tile_m) % 64 != 0:
                        continue
                    for dKV_swapAB in (False, True):
                        # dKV eff_m: hdim/hdimv if swap else tile_n
                        if not dKV_swapAB and tile_n % 64 != 0:
                            continue
                        if dKV_swapAB and (hdim % 64 != 0 or hdimv % 64 != 0):
                            continue
                        for dQ_swapAB in (False, True):
                            # dQ eff_m: hdim if swap else tile_m
                            if (hdim if dQ_swapAB else tile_m) % 64 != 0:
                                continue
                            for aMSdP in divs:
                                for aNdKV in divs:
                                    for aMdQ in divs:
                                        cfg = _check_config(
                                            hdim,
                                            hdimv,
                                            tile_m,
                                            tile_n,
                                            num_wg,
                                            SdP_swapAB,
                                            dKV_swapAB,
                                            dQ_swapAB,
                                            aMSdP,
                                            aNdKV,
                                            aMdQ,
                                        )
                                        if cfg is not None:
                                            results.append(cfg)

    # Largest tile_n first, then largest tile_m, then lowest traffic per block
    results.sort(key=lambda c: (-c["tile_n"], -c["tile_m"], c["smem_traffic_per_block"]))
    return results


def print_configs(configs, max_results=20):
    if not configs:
        print("No feasible configs found!")
        return
    print(
        f"Found {len(configs)} feasible configs (showing top {min(len(configs), max_results)}):\n"
    )
    hdr = (
        f"{'wg':>2} {'tm':>3} {'tn':>3}  "
        f"{'SdP':>3} {'dKV':>3} {'dQ':>3}  "
        f"{'aSdP':>4} {'adKV':>4} {'adQ':>4}  "
        f"{'Qs':>2} {'dOs':>3}  "
        f"{'rS':>3} {'rdK':>3} {'rdV':>3} {'rdQ':>3} {'tot':>4}/{'':<3}  "
        f"{'smem':>5}  {'traffic':>7}  {'tr/blk':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    B = lambda b: "T" if b else "F"
    for c in configs[:max_results]:
        print(
            f"{c['num_wg']:>2} {c['tile_m']:>3} {c['tile_n']:>3}  "
            f"{B(c['SdP_swapAB']):>3} {B(c['dKV_swapAB']):>3} {B(c['dQ_swapAB']):>3}  "
            f"{c['AtomLayoutMSdP']:>4} {c['AtomLayoutNdKV']:>4} {c['AtomLayoutMdQ']:>4}  "
            f"{c['Q_stage']:>2} {c['dO_stage']:>3}  "
            f"{c['regs_SdP']:>3} {c['regs_dK']:>3} {c['regs_dV']:>3} {c['regs_dQ']:>3} "
            f"{c['total_regs']:>4}/{c['reg_limit']:<3}  "
            f"{c['smem_kb']:>4.0f}K  "
            f"{c['smem_traffic_kb']:>6.0f}K  "
            f"{c['smem_traffic_per_block']:>6.1f}"
        )


# ============================================================================
# Forward pass config search
# ============================================================================


def _check_fwd_config(hdim, hdimv, tile_n, num_wg, pv_is_rs, overlap_wg):
    """Check a single forward config. Returns dict if feasible, None otherwise."""
    reg_limit = REG_LIMITS[num_wg]
    tile_m = num_wg * 64  # no swap, each WG handles 64 rows

    # S = Q @ K^T: output (tile_m, tile_n). No swap, all WGs in M.
    # tile_m % 64 == 0 guaranteed. tile_n % 8 == 0 needed.
    if tile_n % 8 != 0:
        return None

    regs_S = _acc_regs(tile_m, tile_n, num_wg)  # f32
    regs_O = _acc_regs(tile_m, hdimv, num_wg)  # f32
    regs_P = regs_S // 2  # bf16, half the regs of f32 S

    if overlap_wg:
        # S, P, and O all live simultaneously
        total_regs = regs_S + regs_P + regs_O
    else:
        # S and O live simultaneously. P can reuse S registers.
        total_regs = regs_S + regs_O

    if total_regs > reg_limit:
        return None

    # SMEM: 1 stage Q, 2 stages K and V, sP if not RS. O overlaps with Q.
    sQ = tile_m * hdim * 2  # 1 stage
    sK = tile_n * hdim * 2 * 2  # 2 stages
    sV = tile_n * hdimv * 2 * 2  # 2 stages
    sO = tile_m * hdimv * 2  # overlaps with sQ
    sP = tile_m * tile_n * 2 if not pv_is_rs else 0
    smem = max(sQ, sO) + sK + sV + sP
    if smem > SMEM_LIMIT:
        return None

    # SMEM traffic per iteration (per tile_n block)
    # S = Q @ K^T: num_instr = num_wg (tile_m/64 × wg_n=1)
    #   A = Q: 64 × hdim × 2 per instr
    #   B = K: tile_n × hdim × 2 per instr (wg_n=1, full tile_n)
    traffic_S = num_wg * (64 * hdim * 2 + tile_n * hdim * 2)

    # O = P @ V: num_instr = num_wg
    #   A = P: 64 × tile_n × 2 per instr (0 if RS)
    #   B = V: hdimv × tile_n × 2 per instr (wg_n=1, full hdimv)
    A_pv = 64 * tile_n * 2 if not pv_is_rs else 0
    B_pv = hdimv * tile_n * 2
    traffic_O = num_wg * (A_pv + B_pv)

    # P R2S store (if not RS)
    traffic_P_store = tile_m * tile_n * 2 if not pv_is_rs else 0

    smem_traffic = traffic_S + traffic_O + traffic_P_store

    return dict(
        tile_m=tile_m,
        tile_n=tile_n,
        num_wg=num_wg,
        pv_is_rs=pv_is_rs,
        overlap_wg=overlap_wg,
        regs_S=regs_S,
        regs_O=regs_O,
        regs_P=regs_P,
        total_regs=total_regs,
        reg_limit=reg_limit,
        smem_bytes=smem,
        smem_kb=smem / 1024,
        smem_traffic=smem_traffic,
        smem_traffic_kb=smem_traffic / 1024,
        smem_traffic_per_block=smem_traffic / (tile_m * tile_n),
    )


def find_feasible_fwd_configs(
    head_dim, head_dim_v=None, tile_n_choices=(64, 80, 96, 112, 128, 144, 160, 176, 192)
):
    """Find all feasible SM90 forward configs for given head dimensions."""
    if head_dim_v is None:
        head_dim_v = head_dim
    hdim = int(math.ceil(head_dim / 32) * 32)
    hdimv = int(math.ceil(head_dim_v / 32) * 32)

    results = []
    for num_wg in (2, 3):
        for tile_n in tile_n_choices:
            for pv_is_rs in (True, False):
                for overlap_wg in (True, False):
                    cfg = _check_fwd_config(hdim, hdimv, tile_n, num_wg, pv_is_rs, overlap_wg)
                    if cfg is not None:
                        results.append(cfg)

    # Largest tile_n first, then lowest traffic per block
    results.sort(key=lambda c: (-c["tile_n"], c["smem_traffic_per_block"]))
    return results


def print_fwd_configs(configs, max_results=20):
    if not configs:
        print("No feasible configs found!")
        return
    print(
        f"Found {len(configs)} feasible configs (showing top {min(len(configs), max_results)}):\n"
    )
    hdr = (
        f"{'wg':>2} {'tm':>3} {'tn':>3}  "
        f"{'RS':>2} {'olap':>4}  "
        f"{'rS':>3} {'rP':>3} {'rO':>3} {'tot':>4}/{'':<3}  "
        f"{'smem':>5}  {'traffic':>7}  {'tr/blk':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    B = lambda b: "T" if b else "F"
    for c in configs[:max_results]:
        print(
            f"{c['num_wg']:>2} {c['tile_m']:>3} {c['tile_n']:>3}  "
            f"{B(c['pv_is_rs']):>2} {B(c['overlap_wg']):>4}  "
            f"{c['regs_S']:>3} {c['regs_P']:>3} {c['regs_O']:>3} "
            f"{c['total_regs']:>4}/{c['reg_limit']:<3}  "
            f"{c['smem_kb']:>4.0f}K  "
            f"{c['smem_traffic_kb']:>6.0f}K  "
            f"{c['smem_traffic_per_block']:>6.1f}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Search feasible SM90 MMA configs")
    parser.add_argument(
        "--mode",
        choices=["fwd", "bwd", "both"],
        default="both",
        help="Search forward, backward, or both (default: both)",
    )
    parser.add_argument(
        "--headdim", type=str, default="128", help="Head dim, or hdim-hdimv (e.g. 192-128)"
    )
    parser.add_argument(
        "--tile-m", type=str, default="64,80,96,112,128", help="Comma-separated bwd tile_m choices"
    )
    parser.add_argument(
        "--tile-n",
        type=str,
        default=None,
        help="Comma-separated tile_n choices (default: fwd up to 192, bwd up to 128)",
    )
    parser.add_argument("-n", "--num-results", type=int, default=30, help="Max results to display")
    args = parser.parse_args()

    parts = args.headdim.split("-")
    hdim = int(parts[0])
    hdimv = int(parts[1]) if len(parts) > 1 else hdim
    tile_n_default_fwd = "64,80,96,112,128,144,160,176,192"
    tile_n_default_bwd = "64,80,96,112,128"
    tile_n_str = args.tile_n

    if args.mode in ("fwd", "both"):
        tn = tuple(int(x) for x in (tile_n_str or tile_n_default_fwd).split(","))
        print(f"=== FWD configs: hdim={hdim}, hdimv={hdimv} ===\n")
        fwd_configs = find_feasible_fwd_configs(hdim, hdimv, tn)
        print_fwd_configs(fwd_configs, args.num_results)
        print()

    if args.mode in ("bwd", "both"):
        tile_m_choices = tuple(int(x) for x in args.tile_m.split(","))
        tn = tuple(int(x) for x in (tile_n_str or tile_n_default_bwd).split(","))
        print(f"=== BWD configs: hdim={hdim}, hdimv={hdimv} ===\n")
        bwd_configs = find_feasible_bwd_configs(hdim, hdimv, tile_m_choices, tn)
        print_configs(bwd_configs, args.num_results)
