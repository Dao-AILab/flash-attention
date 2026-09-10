"""Symmetric D512 backward on SM100 using two-CTA TCGEN05.

The dedicated D256 backward supplies the producer/consumer design: TMA sends
transaction completion directly to the leader CTA, MMA releases input buffers,
and elementwise warps publish the probability or dS operand. DQ/DK compute
probabilities while dP runs; DV selects its query tile by mask type. Two input stages
split the 512-wide reduction into 256-wide chunks. FP32 gradient accumulators
remain in TMEM until the final write, with an FP32 GQA reduction when needed.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, const_expr
from cutlass.cute.nvgpu import tcgen05, cpasync
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100
from flash_attn.cute.cache_utils import get_jit_cache


class NativeD512DqDk:
    def __init__(self, mode, causal=False, softcap=0.0, window=(-1, -1), maxsq=None, maxsk=None):
        self.mode = mode
        self.causal = causal
        self.softcap = softcap
        self.window = window
        assert mode in ("dq", "dk")
        self.maxsq = maxsq
        self.maxsk = maxsk
        self.threads = 256

    @cute.jit
    def __call__(
        self,
        q,
        k,
        v,
        do,
        lse,
        delta,
        dq,
        dk,
        dv,
        scale: Float32,
        cuq,
        cuk,
        usedq,
        usedk,
        stream: cuda.CUstream,
    ):
        self.dtype = q.element_type
        ms = sm100.make_trivial_tiled_mma(
            self.dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            Float32,
            tcgen05.CtaGroup.TWO,
            (128, 128),
        )
        mg = sm100.make_trivial_tiled_mma(
            self.dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            Float32,
            tcgen05.CtaGroup.TWO,
            (128, 256),
        )
        la = sm100.make_smem_layout_a(ms, (128, 128, 512), self.dtype, 1)
        lb = sm100.make_smem_layout_b(ms, (128, 128, 256), self.dtype, 2)
        lp = sm100.make_smem_layout_a(mg, (128, 256, 128), self.dtype, 1)
        lbg = sm100.make_smem_layout_b(mg, (128, 256, 128), self.dtype, 1)
        # TMA uses the per-CTA physical operand layouts; MMA coordinates remain cluster-wide.
        ls = cute.composition(
            cute.select(la, mode=[0, 1, 2]), cute.make_layout((64, 512), stride=(1, 64))
        )
        lbs = cute.composition(
            cute.select(lb, mode=[0, 1, 2]), cute.make_layout((64, 256), stride=(1, 64))
        )
        lg = cute.composition(
            cute.select(lbg, mode=[0, 1, 2]),
            cute.make_layout((128, 128), stride=(1, 128)),
        )
        op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        qa, qt = cpasync.make_tiled_tma_atom(op, self.view(q), ls, (64, 512))
        ka, kt = cpasync.make_tiled_tma_atom(op, self.view(k), ls, (64, 512))
        va, vt = cpasync.make_tiled_tma_atom(op, self.view(v), ls, (64, 512))
        oa, ot = cpasync.make_tiled_tma_atom(op, self.view(do), ls, (64, 512))
        op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.TWO)
        qga, qgt = cpasync.make_tiled_tma_atom(op, self.view(q, True), lg, (128, 128))
        kga, kgt = cpasync.make_tiled_tma_atom(op, self.view(k, True), lg, (128, 128))
        oga, ogt = cpasync.make_tiled_tma_atom(op, self.view(do, True), lg, (128, 128))
        qba, qbt = cpasync.make_tiled_tma_atom(op, self.view(q), lbs, (64, 256))
        kba, kbt = cpasync.make_tiled_tma_atom(op, self.view(k), lbs, (64, 256))
        vba, vbt = cpasync.make_tiled_tma_atom(op, self.view(v), lbs, (64, 256))
        oba, obt = cpasync.make_tiled_tma_atom(op, self.view(do), lbs, (64, 256))
        loaders = (
            qa,
            qt,
            ka,
            kt,
            va,
            vt,
            oa,
            ot,
            qga,
            qgt,
            kga,
            kgt,
            oga,
            ogt,
            qba,
            qbt,
            kba,
            kbt,
            vba,
            vbt,
            oba,
            obt,
        )
        maxsq = self.maxsq if self.maxsq is not None else q.shape[1]
        maxsk = self.maxsk if self.maxsk is not None else k.shape[1]
        batches = cuq.shape[0] - 1 if cuq is not None else q.shape[0]
        if const_expr(self.mode == "dq"):
            grid = (cute.ceil_div(maxsq, 128) * 2 * q.shape[2], 1, batches)
        else:
            grid = (cute.ceil_div(maxsk, 128) * 2 * q.shape[2], 1, batches)
        self.kernel(
            q,
            k,
            v,
            do,
            lse,
            delta,
            dq,
            dk,
            dv,
            scale,
            ms,
            mg,
            la,
            lb,
            lp,
            lbg,
            loaders,
            cuq,
            cuk,
            usedq,
            usedk,
            maxsq,
        ).launch(grid=grid, block=(self.threads, 1, 1), cluster=(2, 1, 1), stream=stream)

    @cute.jit
    def view(self, x, transpose: cutlass.Constexpr = False):
        if const_expr(transpose):
            shape = (x.shape[3], x.shape[1], (x.shape[2], x.shape[0]))
            stride = (x.stride[3], x.stride[1], (x.stride[2], x.stride[0]))
        else:
            shape = (x.shape[1], x.shape[3], (x.shape[2], x.shape[0]))
            stride = (x.stride[1], x.stride[3], (x.stride[2], x.stride[0]))
        return cute.make_tensor(x.iterator, cute.make_layout(shape, stride=stride))

    @cute.jit
    def load(
        self,
        atom,
        tensor,
        dst,
        start: Int32,
        head: Int32,
        batch: Int32,
        col: Int32,
        bar,
        phase: Int32,
        transpose: cutlass.Constexpr = False,
        offset: Int32 = 0,
    ):
        tid, _, _ = cute.arch.thread_idx()
        m = tensor[None, None, (head, batch)]
        if const_expr(transpose):
            m = cute.domain_offset((0, offset), m)
        else:
            m = cute.domain_offset((offset, 0), m)
        if const_expr(transpose):
            g = cute.local_tile(m, (128, 128), (col // 128, start // 128))
            nbytes = 32768
        else:
            g = cute.local_tile(m, (64, 512), (start // 64, 0))
            nbytes = 65536
        d, s = cpasync.tma_partition(
            atom,
            0,
            cute.make_layout(1),
            cute.group_modes(dst, 0, 2),
            cute.group_modes(g, 0, 2),
        )
        if tid < 32:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(bar, nbytes)
            cute.copy(atom, s, d, tma_bar_ptr=bar)
        cute.arch.mbarrier_wait(bar, phase)
        cute.arch.barrier()
        return phase ^ 1

    @cute.jit
    def produce(
        self,
        atom,
        tensor,
        dst,
        start: Int32,
        head: Int32,
        batch: Int32,
        col: Int32,
        loadbar,
        ready,
        phase: Int32,
        transpose: cutlass.Constexpr = False,
        offset: Int32 = 0,
    ):
        m = tensor[None, None, (head, batch)]
        if const_expr(transpose):
            m = cute.domain_offset((0, offset), m)
        else:
            m = cute.domain_offset((offset, 0), m)
        if const_expr(transpose):
            g = cute.local_tile(m, (128, 128), (col // 128, start // 128))
            nbytes = 32768
        else:
            g = cute.local_tile(m, (64, 256), (start // 64, col // 256))
            nbytes = 32768
        d, s = cpasync.tma_partition(
            atom,
            0,
            cute.make_layout(1),
            cute.group_modes(dst, 0, 2),
            cute.group_modes(g, 0, 2),
        )
        rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        if rank == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(ready, 2 * nbytes)
        cute.copy(atom, s, d, tma_bar_ptr=ready)
        return phase ^ 1

    @cute.jit
    def issue(
        self,
        mma,
        acc,
        a,
        b,
        ready,
        empty,
        phase: Int32,
        zero: cutlass.Boolean,
        headpart: cutlass.Constexpr = 0,
    ):
        cute.arch.mbarrier_wait(ready, phase)
        ta = mma.make_fragment_A(a)
        tb = mma.make_fragment_B(b)
        atom = cute.make_mma_atom(mma.op)
        for step in cutlass.range_constexpr(cute.size(tb.shape[2])):
            atom.set(tcgen05.Field.ACCUMULATE, not zero or step != 0)
            cute.gemm(
                atom,
                acc,
                ta[None, None, step + headpart * cute.size(tb.shape[2])],
                tb[None, None, step],
                acc,
            )
        with cute.arch.elect_one():
            tcgen05.commit(empty, mask=3, cta_group=tcgen05.CtaGroup.TWO)

    @cute.jit
    def epilogue(
        self,
        tensor,
        coord,
        scratch,
        grad,
        batch: Int32,
        head: Int32,
        start: Int32,
        rows: Int32,
        col: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        rank = cute.arch.block_idx_in_cluster()
        layout = cute.composition(
            cute.make_layout((64, (32, 8)), stride=(32, (1, 2048))),
            cute.make_layout((64, 256), stride=(1, 64)),
        )
        stage = cute.make_tensor(
            cute.recast_ptr(scratch, cute.make_swizzle(3, 4, 3), Float32), layout
        )
        if tid < 128:
            for strip in cutlass.range(4, unroll=1):
                ts = cute.make_tensor(
                    tensor.iterator + strip * 32,
                    cute.make_layout((128, 32), stride=(65536, 1)),
                )
                cp = tcgen05.make_tmem_copy(
                    cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(32)), Float32),
                    ts,
                )
                th = cp.get_slice(tid)
                coords = th.partition_D(cute.make_identity_tensor((128, 32)))
                rr = cute.make_rmem_tensor(coords.shape, Float32)
                cute.copy(cp, th.partition_S(ts), rr)
                cute.arch.fence_view_async_tmem_load()
                cpstage = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128
                )
                for j in cutlass.range_constexpr(8):
                    row, cc = coords[j * 4]
                    offset = cute.assume(
                        cute.crd2idx(
                            (row % 64, (row // 64) * 128 + strip * 32 + cc),
                            stage.layout,
                        ),
                        divby=4,
                    )
                    target = cute.make_tensor(stage.iterator + offset, cute.make_layout(4))
                    source = cute.make_tensor(rr.iterator + j * 4, cute.make_layout(4))
                    cute.copy(cpstage, source, target)
        cute.arch.barrier()
        cpout = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            grad.element_type,
            num_bits_per_copy=4 * grad.element_type.width,
        )
        val = cute.make_rmem_tensor((4,), grad.element_type)
        for i in cutlass.range(tid, 64 * 256 // 4, self.threads):
            r = i // 64
            c = (i % 64) * 4
            if start + rank * 64 + r < rows:
                for j in cutlass.range_constexpr(4):
                    val[j] = grad.element_type(stage[r, c + j])
                # Widen coordinates before multiplying by user-provided strides.
                offset = cute.crd2idx(
                    (Int64(batch), Int64(start) + rank * 64 + r, Int64(head), Int64(col) + c),
                    grad.layout,
                )
                target = cute.make_tensor(
                    grad.iterator + cute.assume(offset, divby=4), cute.make_layout(4)
                )
                cute.copy(cpout, val, target)
        cute.arch.barrier()

    @cute.kernel
    def kernel(
        self,
        q,
        k,
        v,
        do,
        lse,
        delta,
        dq,
        dk,
        dv,
        scale: Float32,
        ms,
        mg,
        la,
        lb,
        lp,
        lbg,
        loaders,
        cuq,
        cuk,
        usedq,
        usedk,
        maxsq: cutlass.Constexpr,
    ):
        cute.arch.griddepcontrol_wait()
        tid, _, _ = cute.arch.thread_idx()
        bx, by, batch = cute.arch.block_idx()
        rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        if const_expr(self.mode == "dq"):
            by = (bx // 2) % q.shape[2]
            tile = (bx // 2) // q.shape[2]
            if const_expr(self.causal):
                tile = cute.ceil_div(maxsq, 128) - 1 - tile
        else:
            by = (bx // 2) % q.shape[2]
            tile = (bx // 2) // q.shape[2]
        ratio = q.shape[2] // k.shape[2]
        sq = q.shape[1]
        sk = k.shape[1]
        qb = batch
        kb = batch
        qo = Int32(0)
        ko = Int32(0)
        if const_expr(cuq is not None):
            qb = Int32(0)
            qo = cuq[batch]
            sq = cuq[batch + 1] - qo
        if const_expr(cuk is not None):
            kb = Int32(0)
            ko = cuk[batch]
            sk = cuk[batch + 1] - ko
        delta_offset = Int32(0)
        if const_expr(cuq is not None):
            delta_offset = ((qo + batch * 128) // 128) * 128
        sq_storage = sq
        sk_storage = sk
        if const_expr(usedq is not None):
            sq = usedq[batch]
        if const_expr(usedk is not None):
            sk = usedk[batch]
        tile_rows = sq_storage if const_expr(self.mode == "dq") else sk_storage
        if tile * 128 < tile_rows:
            qs = Int32(0)
            ks = Int32(0)
            if const_expr(self.mode == "dq"):
                h = by
                kh = h // ratio
                qs = tile * 128
                iterations = cute.ceil_div(sk, 128)
            else:
                h = by
                kh = h // ratio
                ks = tile * 128
                iterations = cute.ceil_div(sq, 128)
            smem = utils.SmemAllocator()
            slse = smem.allocate_array(Float32, 128)
            sdelta = smem.allocate_array(Float32, 128)
            ready = smem.allocate_array(Int64, 2)
            score_ready = smem.allocate_array(Int64, 1)
            compute_ready = smem.allocate_array(Int64, 1)
            ds_ready = smem.allocate_array(Int64, 1)
            loadbar = smem.allocate_array(Int64, 1)
            (
                qa,
                qt,
                ka,
                kt,
                va,
                vt,
                oa,
                ot,
                qga,
                qgt,
                kga,
                kgt,
                oga,
                ogt,
                qba,
                qbt,
                kba,
                kbt,
                vba,
                vbt,
                oba,
                obt,
            ) = loaders
            if cute.arch.warp_idx() == 5:
                for atom in (qa, ka, va, oa, qga, kga, oga, qba, kba, vba, oba):
                    cpasync.prefetch_descriptor(atom)
            lphase = Int32(0)
            bar = smem.allocate_array(Int64, 2)
            dealloc = smem.allocate_array(Int64, 1)
            ptrslot = smem.allocate_array(Int32, 1)
            sa_full = smem.allocate_tensor(
                self.dtype, la.outer, swizzle=la.inner, byte_alignment=1024
            )
            sao_full = smem.allocate_tensor(
                self.dtype, la.outer, swizzle=la.inner, byte_alignment=1024
            )
            aop = cute.make_tensor(sao_full.iterator, cute.select(la.outer, mode=[0, 1, 2]))
            sao = cute.make_tensor(
                aop.iterator,
                cute.composition(aop.layout, cute.make_layout((64, 512), stride=(1, 64))),
            )
            sb_full = smem.allocate_tensor(
                self.dtype, lb.outer, swizzle=lb.inner, byte_alignment=1024
            )
            sp_full = smem.allocate_tensor(
                self.dtype, lp.outer, swizzle=lp.inner, byte_alignment=1024
            )
            ss_full = sp_full
            ap = cute.make_tensor(sa_full.iterator, cute.select(la.outer, mode=[0, 1, 2]))
            bp0 = cute.make_tensor(sb_full.iterator, cute.select(lb.outer, mode=[0, 1, 2]))
            bp1 = cute.make_tensor(sb_full.iterator + 16384, cute.select(lb.outer, mode=[0, 1, 2]))
            ds = cute.make_tensor(ss_full.iterator, cute.select(lp.outer, mode=[0, 1, 2]))
            bgp0 = cute.make_tensor(
                cute.recast_ptr(sb_full.iterator, lbg.inner),
                cute.select(lbg.outer, mode=[0, 1, 2]),
            )
            bgp1 = cute.make_tensor(
                cute.recast_ptr(sb_full.iterator + 16384, lbg.inner),
                cute.select(lbg.outer, mode=[0, 1, 2]),
            )
            sa = cute.make_tensor(
                ap.iterator,
                cute.composition(ap.layout, cute.make_layout((64, 512), stride=(1, 64))),
            )
            sb0 = cute.make_tensor(
                bp0.iterator,
                cute.composition(bp0.layout, cute.make_layout((64, 256), stride=(1, 64))),
            )
            sb1 = cute.make_tensor(
                bp1.iterator,
                cute.composition(bp1.layout, cute.make_layout((64, 256), stride=(1, 64))),
            )
            ss = cute.make_tensor(
                ds.iterator,
                cute.composition(ds.layout, cute.make_layout((64, 128), stride=(1, 64))),
            )
            bg0 = cute.make_tensor(
                bgp0.iterator,
                cute.composition(bgp0.layout, cute.make_layout((128, 128), stride=(1, 128))),
            )
            bg1 = cute.make_tensor(
                bgp1.iterator,
                cute.composition(bgp1.layout, cute.make_layout((128, 128), stride=(1, 128))),
            )
            if tid == 0:
                cute.arch.mbarrier_init(bar, 1)
                cute.arch.mbarrier_init(bar + 1, 1)
                cute.arch.mbarrier_init(loadbar, 1)
                cute.arch.mbarrier_init(ready, 1)
                cute.arch.mbarrier_init(ready + 1, 1)
                cute.arch.mbarrier_init(score_ready, 1)
                cute.arch.mbarrier_init(compute_ready, 1)
                cute.arch.mbarrier_init(ds_ready, 2)
            cute.arch.mbarrier_init_fence()
            cute.arch.barrier()
            tmem = utils.TmemAllocator(
                ptrslot,
                barrier_for_retrieve=cutlass.pipeline.NamedBarrier(
                    barrier_id=1, num_threads=self.threads
                ),
                is_two_cta=True,
                two_cta_tmem_dealloc_mbar_ptr=dealloc,
            )
            # Both CTAs must finish shared barrier initialization before the
            # allocator accesses its peer CTA. A CTA-only barrier is insufficient.
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            tmem.allocate(512)
            tmem.wait_for_alloc()
            tp = tmem.retrieve_ptr(Float32)
            tmem.relinquish_alloc_permit()
            cg = mg.get_slice(rank).make_fragment_C(
                mg.get_slice(rank).partition_shape_C((128, 256))
            )
            cs = ms.get_slice(rank).make_fragment_C(
                ms.get_slice(rank).partition_shape_C((128, 128))
            )
            acc0 = cute.make_tensor(tp, cg.layout)
            acc1 = cute.make_tensor(tp + 128, cg.layout)
            scores = cute.make_tensor(tp + 256, cs.layout)
            dprob = cute.make_tensor(tp + 320, cs.layout)
            coord_g = mg.get_slice(rank).partition_C(cute.make_identity_tensor((128, 256)))
            initialized = False
            if const_expr(self.mode == "dq"):
                lphase = self.load(qa, qt, sa, qs + rank * 64, h, qb, 0, loadbar, lphase, offset=qo)
                lphase = self.load(
                    oa, ot, sao, qs + rank * 64, h, qb, 0, loadbar, lphase, offset=qo
                )
            else:
                lphase = self.load(
                    ka, kt, sa, ks + rank * 64, kh, kb, 0, loadbar, lphase, offset=ko
                )
                lphase = self.load(
                    va,
                    vt,
                    sao,
                    ks + rank * 64,
                    kh,
                    kb,
                    0,
                    loadbar,
                    lphase,
                    offset=ko,
                )
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            compute_sync = cutlass.pipeline.NamedBarrier(barrier_id=2, num_threads=128)
            cphase = Int32(0)
            for it in cutlass.range(iterations):
                if const_expr(self.mode == "dq"):
                    ks = it * 128
                else:
                    qs = it * 128
                active = True
                if const_expr(self.causal):
                    active = ks <= qs + 127 + sk - sq
                # Skip tiles outside the local window before TMA and MMA.
                # Elementwise masking still handles partial boundary tiles.
                if const_expr(self.window[0] >= 0):
                    active = active and ks + 127 >= qs + sk - sq - self.window[0]
                if const_expr(self.window[1] >= 0 and not self.causal):
                    active = active and ks <= qs + 127 + sk - sq + self.window[1]
                if active:
                    if warp == 5:
                        if initialized:
                            cute.arch.mbarrier_wait(bar, cphase ^ 1)
                        if const_expr(self.mode == "dq"):
                            lphase = self.produce(
                                kba,
                                kbt,
                                sb0,
                                ks + rank * 64,
                                kh,
                                kb,
                                0,
                                loadbar,
                                ready,
                                lphase,
                                offset=ko,
                            )
                        else:
                            lphase = self.produce(
                                qba,
                                qbt,
                                sb0,
                                qs + rank * 64,
                                h,
                                qb,
                                0,
                                loadbar,
                                ready,
                                lphase,
                                offset=qo,
                            )
                        if initialized:
                            cute.arch.mbarrier_wait(bar + 1, cphase ^ 1)
                        if const_expr(self.mode == "dq"):
                            lphase = self.produce(
                                kba,
                                kbt,
                                sb1,
                                ks + rank * 64,
                                kh,
                                kb,
                                256,
                                loadbar,
                                ready + 1,
                                lphase,
                                offset=ko,
                            )
                        else:
                            lphase = self.produce(
                                qba,
                                qbt,
                                sb1,
                                qs + rank * 64,
                                h,
                                qb,
                                256,
                                loadbar,
                                ready + 1,
                                lphase,
                                offset=qo,
                            )
                        cute.arch.mbarrier_wait(bar, cphase)
                        if const_expr(self.mode == "dq"):
                            lphase = self.produce(
                                vba,
                                vbt,
                                sb0,
                                ks + rank * 64,
                                kh,
                                kb,
                                0,
                                loadbar,
                                ready,
                                lphase,
                                offset=ko,
                            )
                        else:
                            lphase = self.produce(
                                oba,
                                obt,
                                sb0,
                                qs + rank * 64,
                                h,
                                qb,
                                0,
                                loadbar,
                                ready,
                                lphase,
                                offset=qo,
                            )
                        cute.arch.mbarrier_wait(bar + 1, cphase)
                        if const_expr(self.mode == "dq"):
                            lphase = self.produce(
                                vba,
                                vbt,
                                sb1,
                                ks + rank * 64,
                                kh,
                                kb,
                                256,
                                loadbar,
                                ready + 1,
                                lphase,
                                offset=ko,
                            )
                        else:
                            lphase = self.produce(
                                oba,
                                obt,
                                sb1,
                                qs + rank * 64,
                                h,
                                qb,
                                256,
                                loadbar,
                                ready + 1,
                                lphase,
                                offset=qo,
                            )
                        cute.arch.mbarrier_wait(bar, cphase ^ 1)
                        if const_expr(self.mode == "dq"):
                            lphase = self.produce(
                                kga,
                                kgt,
                                bg0,
                                ks,
                                kh,
                                kb,
                                rank * 128,
                                loadbar,
                                ready,
                                lphase,
                                True,
                                offset=ko,
                            )
                        else:
                            lphase = self.produce(
                                qga,
                                qgt,
                                bg0,
                                qs,
                                h,
                                qb,
                                rank * 128,
                                loadbar,
                                ready,
                                lphase,
                                True,
                                offset=qo,
                            )
                        cute.arch.mbarrier_wait(bar + 1, cphase ^ 1)
                        if const_expr(self.mode == "dq"):
                            lphase = self.produce(
                                kga,
                                kgt,
                                bg1,
                                ks,
                                kh,
                                kb,
                                256 + rank * 128,
                                loadbar,
                                ready + 1,
                                lphase,
                                True,
                                offset=ko,
                            )
                        else:
                            lphase = self.produce(
                                qga,
                                qgt,
                                bg1,
                                qs,
                                h,
                                qb,
                                256 + rank * 128,
                                loadbar,
                                ready + 1,
                                lphase,
                                True,
                                offset=qo,
                            )
                    if warp == 4:
                        if rank == 0:
                            self.issue(ms, scores, ap, bp0, ready, bar, cphase, True, 0)
                            self.issue(
                                ms,
                                scores,
                                ap,
                                bp1,
                                ready + 1,
                                bar + 1,
                                cphase,
                                False,
                                1,
                            )
                            with cute.arch.elect_one():
                                tcgen05.commit(score_ready, mask=3, cta_group=tcgen05.CtaGroup.TWO)
                            self.issue(ms, dprob, aop, bp0, ready, bar, cphase ^ 1, True, 0)
                            self.issue(
                                ms,
                                dprob,
                                aop,
                                bp1,
                                ready + 1,
                                bar + 1,
                                cphase ^ 1,
                                False,
                                1,
                            )
                            with cute.arch.elect_one():
                                tcgen05.commit(
                                    compute_ready,
                                    mask=3,
                                    cta_group=tcgen05.CtaGroup.TWO,
                                )
                            cute.arch.mbarrier_wait(ds_ready, cphase)
                            self.issue(mg, acc0, ds, bgp0, ready, bar, cphase, not initialized)
                            self.issue(
                                mg,
                                acc1,
                                ds,
                                bgp1,
                                ready + 1,
                                bar + 1,
                                cphase,
                                not initialized,
                            )
                    if warp < 4:
                        x = Float32(0)
                        y = Float32(0)
                        if qs + tid < sq:
                            x = lse[qb, h, qo + qs + tid]
                            y = delta[qb, h, delta_offset + qs + tid]
                        slse[tid] = x
                        sdelta[tid] = y
                        compute_sync.arrive_and_wait()
                        cute.arch.mbarrier_wait(score_ready, cphase)
                        probabilities = cute.make_rmem_tensor((64,), Float32)
                        if const_expr(self.softcap > 0):
                            derivatives = cute.make_rmem_tensor((64,), Float32)
                        for strip in cutlass.range_constexpr(4):
                            physical = cute.make_layout((128, 16), stride=(65536, 1))
                            ts = cute.make_tensor(scores.iterator + strip * 16, physical)
                            cp = tcgen05.make_tmem_copy(
                                cute.make_copy_atom(
                                    tcgen05.Ld32x32bOp(tcgen05.Repetition(16)), Float32
                                ),
                                ts,
                            )
                            th = cp.get_slice(tid)
                            coords = th.partition_D(cute.make_identity_tensor((128, 16)))
                            rs = cute.make_rmem_tensor(coords.shape, Float32)
                            cute.copy(cp, th.partition_S(ts), rs)
                            cute.arch.fence_view_async_tmem_load()
                            for j in cutlass.range_constexpr(cute.size(rs)):
                                row, col = coords[j]
                                r = rank * 64 + row % 64
                                c = (row // 64) * 64 + strip * 16 + col
                                if const_expr(self.mode == "dq"):
                                    qi = qs + r
                                    ki = ks + c
                                else:
                                    qi = qs + c
                                    ki = ks + r
                                valid = qi < sq and ki < sk
                                center = qi + sk - sq
                                if const_expr(self.causal):
                                    valid = valid and ki <= center
                                if const_expr(self.window[0] >= 0):
                                    valid = valid and ki >= center - self.window[0]
                                if const_expr(self.window[1] >= 0):
                                    valid = valid and ki <= center + self.window[1]
                                prob = Float32(0)
                                deriv = Float32(1)
                                if valid:
                                    value = rs[j] * scale
                                    if const_expr(self.softcap > 0):
                                        t = (
                                            2
                                            / (
                                                1
                                                + cute.math.exp(
                                                    -2 * value / self.softcap,
                                                    fastmath=True,
                                                )
                                            )
                                            - 1
                                        )
                                        value = self.softcap * t
                                        deriv = 1 - t * t
                                    prob = cute.math.exp(value - slse[qi - qs], fastmath=True)
                                probabilities[strip * 16 + j] = prob
                                if const_expr(self.softcap > 0):
                                    derivatives[strip * 16 + j] = deriv
                        cute.arch.mbarrier_wait(compute_ready, cphase)
                        copy_smem = cute.make_copy_atom(
                            cute.nvgpu.CopyUniversalOp(),
                            self.dtype,
                            num_bits_per_copy=128,
                        )
                        for strip in cutlass.range_constexpr(4):
                            td = cute.make_tensor(
                                dprob.iterator + strip * 16,
                                cute.make_layout((128, 16), stride=(65536, 1)),
                            )
                            cp = tcgen05.make_tmem_copy(
                                cute.make_copy_atom(
                                    tcgen05.Ld32x32bOp(tcgen05.Repetition(16)), Float32
                                ),
                                td,
                            )
                            th = cp.get_slice(tid)
                            coords = th.partition_D(cute.make_identity_tensor((128, 16)))
                            rd = cute.make_rmem_tensor(coords.shape, Float32)
                            rds = cute.make_rmem_tensor(coords.shape, self.dtype)
                            cute.copy(cp, th.partition_S(td), rd)
                            cute.arch.fence_view_async_tmem_load()
                            for j in cutlass.range_constexpr(cute.size(rd)):
                                row, col = coords[j]
                                r = rank * 64 + row % 64
                                c = (row // 64) * 64 + strip * 16 + col
                                if const_expr(self.mode == "dq"):
                                    qi = qs + r
                                else:
                                    qi = qs + c
                                dd = (
                                    probabilities[strip * 16 + j]
                                    * (rd[j] - sdelta[qi - qs])
                                    * scale
                                )
                                if const_expr(self.softcap > 0):
                                    dd = dd * derivatives[strip * 16 + j]
                                rds[j] = self.dtype(dd)
                            for j in cutlass.range_constexpr(2):
                                row, col = coords[j * 8]
                                r = row % 64
                                c = (row // 64) * 64 + strip * 16 + col
                                offset = cute.assume(cute.crd2idx((r, c), ss.layout), divby=8)
                                target = cute.make_tensor(ss.iterator + offset, cute.make_layout(8))
                                source = cute.make_tensor(rds.iterator + j * 8, cute.make_layout(8))
                                cute.copy(copy_smem, source, target)
                        cute.arch.fence_view_async_shared()
                        compute_sync.arrive_and_wait()
                        if tid == 0:
                            cute.arch.mbarrier_arrive(ds_ready, peer_cta_rank_in_cluster=0)
                    initialized = True
                    cphase = cphase ^ 1
            cute.arch.barrier()
            if initialized:
                cute.arch.mbarrier_wait(bar + 1, cphase ^ 1)
            cute.arch.barrier()
            if initialized:
                if const_expr(self.mode == "dq"):
                    self.epilogue(
                        acc0,
                        coord_g,
                        sa_full.iterator,
                        dq,
                        qb,
                        h,
                        qo + qs,
                        qo + sq_storage,
                        0,
                    )
                    self.epilogue(
                        acc1,
                        coord_g,
                        sa_full.iterator,
                        dq,
                        qb,
                        h,
                        qo + qs,
                        qo + sq_storage,
                        256,
                    )
                else:
                    self.epilogue(
                        acc0,
                        coord_g,
                        sa_full.iterator,
                        dk,
                        kb,
                        h,
                        ko + ks,
                        ko + sk_storage,
                        0,
                    )
                    self.epilogue(
                        acc1,
                        coord_g,
                        sa_full.iterator,
                        dk,
                        kb,
                        h,
                        ko + ks,
                        ko + sk_storage,
                        256,
                    )
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            tmem.free(tp)


class NativeD512Dv(NativeD512DqDk):
    def __init__(self, mode, causal=False, softcap=0.0, window=(-1, -1), maxsq=None, maxsk=None):
        self.mode = mode
        self.causal = causal
        self.softcap = softcap
        self.window = window
        assert mode == "dv"
        self.maxsq = maxsq
        self.maxsk = maxsk
        self.threads = 256
        # A smaller query tile reduces resource pressure for noncausal DV.
        # Causal attention retains the wider tile to amortize traversal costs.
        self.query_tile = 256 if causal else 128

    @cute.jit
    def __call__(
        self,
        q,
        k,
        v,
        do,
        lse,
        delta,
        dq,
        dk,
        dv,
        scale: Float32,
        cuq,
        cuk,
        usedq,
        usedk,
        stream: cuda.CUstream,
    ):
        self.dtype = q.element_type
        ms = sm100.make_trivial_tiled_mma(
            self.dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            Float32,
            tcgen05.CtaGroup.TWO,
            (128, self.query_tile),
        )
        mg = sm100.make_trivial_tiled_mma(
            self.dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            Float32,
            tcgen05.CtaGroup.TWO,
            (128, 256),
        )
        la = sm100.make_smem_layout_a(ms, (128, self.query_tile, 512), self.dtype, 1)
        lb = sm100.make_smem_layout_b(ms, (128, self.query_tile, 256), self.dtype, 2)
        lp = sm100.make_smem_layout_a(mg, (128, 256, self.query_tile), self.dtype, 1)
        lbg = sm100.make_smem_layout_b(mg, (128, 256, self.query_tile), self.dtype, 1)
        # TMA uses the per-CTA physical operand layouts; MMA coordinates remain cluster-wide.
        ls = cute.composition(
            cute.select(la, mode=[0, 1, 2]), cute.make_layout((64, 512), stride=(1, 64))
        )
        lg = cute.composition(
            cute.select(lbg, mode=[0, 1, 2]),
            cute.make_layout((128, self.query_tile), stride=(1, 128)),
        )
        lbs = cute.composition(
            cute.select(lb, mode=[0, 1, 2]),
            cute.make_layout((self.query_tile // 2, 256), stride=(1, self.query_tile // 2)),
        )
        op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        qa, qt = cpasync.make_tiled_tma_atom(op, self.view(q), ls, (64, 512))
        ka, kt = cpasync.make_tiled_tma_atom(op, self.view(k), ls, (64, 512))
        va, vt = cpasync.make_tiled_tma_atom(op, self.view(v), ls, (64, 512))
        oa, ot = cpasync.make_tiled_tma_atom(op, self.view(do), ls, (64, 512))
        op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.TWO)
        qga, qgt = cpasync.make_tiled_tma_atom(op, self.view(q, True), lg, (128, self.query_tile))
        kga, kgt = cpasync.make_tiled_tma_atom(op, self.view(k, True), lg, (128, self.query_tile))
        oga, ogt = cpasync.make_tiled_tma_atom(op, self.view(do, True), lg, (128, self.query_tile))
        qba, qbt = cpasync.make_tiled_tma_atom(op, self.view(q), lbs, (self.query_tile // 2, 256))
        kba, kbt = cpasync.make_tiled_tma_atom(op, self.view(k), lbs, (self.query_tile // 2, 256))
        vba, vbt = cpasync.make_tiled_tma_atom(op, self.view(v), lbs, (self.query_tile // 2, 256))
        oba, obt = cpasync.make_tiled_tma_atom(op, self.view(do), lbs, (self.query_tile // 2, 256))
        loaders = (
            qa,
            qt,
            ka,
            kt,
            va,
            vt,
            oa,
            ot,
            qga,
            qgt,
            kga,
            kgt,
            oga,
            ogt,
            qba,
            qbt,
            kba,
            kbt,
            vba,
            vbt,
            oba,
            obt,
        )
        maxsq = self.maxsq if self.maxsq is not None else q.shape[1]
        maxsk = self.maxsk if self.maxsk is not None else k.shape[1]
        batches = cuq.shape[0] - 1 if cuq is not None else q.shape[0]
        grid = (cute.ceil_div(maxsk, 128) * 2 * q.shape[2], 1, batches)
        self.kernel(
            q,
            k,
            v,
            do,
            lse,
            delta,
            dq,
            dk,
            dv,
            scale,
            ms,
            mg,
            la,
            lb,
            lp,
            lbg,
            loaders,
            cuq,
            cuk,
            usedq,
            usedk,
            maxsq,
        ).launch(grid=grid, block=(self.threads, 1, 1), cluster=(2, 1, 1), stream=stream)

    @cute.jit
    def produce(
        self,
        atom,
        tensor,
        dst,
        start: Int32,
        head: Int32,
        batch: Int32,
        col: Int32,
        loadbar,
        ready,
        phase: Int32,
        transpose: cutlass.Constexpr = False,
        offset: Int32 = 0,
    ):
        m = tensor[None, None, (head, batch)]
        if const_expr(transpose):
            m = cute.domain_offset((0, offset), m)
        else:
            m = cute.domain_offset((offset, 0), m)
        if const_expr(transpose):
            g = cute.local_tile(m, (128, self.query_tile), (col // 128, start // self.query_tile))
            nbytes = self.query_tile * 256
        else:
            g = cute.local_tile(
                m, (self.query_tile // 2, 256), (start // (self.query_tile // 2), col // 256)
            )
            nbytes = self.query_tile * 256
        d, s = cpasync.tma_partition(
            atom,
            0,
            cute.make_layout(1),
            cute.group_modes(dst, 0, 2),
            cute.group_modes(g, 0, 2),
        )
        rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        if rank == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(ready, 2 * nbytes)
        cute.copy(atom, s, d, tma_bar_ptr=ready)
        return phase ^ 1

    @cute.kernel
    def kernel(
        self,
        q,
        k,
        v,
        do,
        lse,
        delta,
        dq,
        dk,
        dv,
        scale: Float32,
        ms,
        mg,
        la,
        lb,
        lp,
        lbg,
        loaders,
        cuq,
        cuk,
        usedq,
        usedk,
        maxsq: cutlass.Constexpr,
    ):
        cute.arch.griddepcontrol_wait()
        tid, _, _ = cute.arch.thread_idx()
        bx, by, batch = cute.arch.block_idx()
        rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        by = (bx // 2) % q.shape[2]
        tile = (bx // 2) // q.shape[2]
        ratio = q.shape[2] // k.shape[2]
        sq = q.shape[1]
        sk = k.shape[1]
        qb = batch
        kb = batch
        qo = Int32(0)
        ko = Int32(0)
        if const_expr(cuq is not None):
            qb = Int32(0)
            qo = cuq[batch]
            sq = cuq[batch + 1] - qo
        if const_expr(cuk is not None):
            kb = Int32(0)
            ko = cuk[batch]
            sk = cuk[batch + 1] - ko
        sk_storage = sk
        if const_expr(usedq is not None):
            sq = usedq[batch]
        if const_expr(usedk is not None):
            sk = usedk[batch]
        tile_rows = sk_storage
        if tile * 128 < tile_rows:
            qs = Int32(0)
            ks = Int32(0)
            h = by
            kh = h // ratio
            ks = tile * 128
            iterations = cute.ceil_div(sq, self.query_tile)
            smem = utils.SmemAllocator()
            slse = smem.allocate_array(Float32, self.query_tile)
            ready = smem.allocate_array(Int64, 2)
            compute_ready = smem.allocate_array(Int64, 1)
            ds_ready = smem.allocate_array(Int64, 1)
            loadbar = smem.allocate_array(Int64, 2)
            (
                qa,
                qt,
                ka,
                kt,
                va,
                vt,
                oa,
                ot,
                qga,
                qgt,
                kga,
                kgt,
                oga,
                ogt,
                qba,
                qbt,
                kba,
                kbt,
                vba,
                vbt,
                oba,
                obt,
            ) = loaders
            lphase = Int32(0)
            bar = smem.allocate_array(Int64, 2)
            dealloc = smem.allocate_array(Int64, 1)
            ptrslot = smem.allocate_array(Int32, 1)
            sa_full = smem.allocate_tensor(
                self.dtype, la.outer, swizzle=la.inner, byte_alignment=1024
            )
            sb_full = smem.allocate_tensor(
                self.dtype, lb.outer, swizzle=lb.inner, byte_alignment=1024
            )
            sp_full = smem.allocate_tensor(
                self.dtype, lp.outer, swizzle=lp.inner, byte_alignment=1024
            )
            ap = cute.make_tensor(sa_full.iterator, cute.select(la.outer, mode=[0, 1, 2]))
            bp = cute.make_tensor(sb_full.iterator, cute.select(lb.outer, mode=[0, 1, 2]))
            bp1 = cute.make_tensor(
                sb_full.iterator + self.query_tile * 128, cute.select(lb.outer, mode=[0, 1, 2])
            )
            pp = cute.make_tensor(sp_full.iterator, cute.select(lp.outer, mode=[0, 1, 2]))
            bgp = cute.make_tensor(
                cute.recast_ptr(sb_full.iterator, lbg.inner),
                cute.select(lbg.outer, mode=[0, 1, 2]),
            )
            bgp1 = cute.make_tensor(
                cute.recast_ptr(sb_full.iterator + self.query_tile * 128, lbg.inner),
                cute.select(lbg.outer, mode=[0, 1, 2]),
            )
            sa = cute.make_tensor(
                ap.iterator,
                cute.composition(ap.layout, cute.make_layout((64, 512), stride=(1, 64))),
            )
            sb = cute.make_tensor(
                bp.iterator,
                cute.composition(
                    bp.layout,
                    cute.make_layout((self.query_tile // 2, 256), stride=(1, self.query_tile // 2)),
                ),
            )
            sb1 = cute.make_tensor(
                bp1.iterator,
                cute.composition(
                    bp1.layout,
                    cute.make_layout((self.query_tile // 2, 256), stride=(1, self.query_tile // 2)),
                ),
            )
            bg1 = cute.make_tensor(
                bgp1.iterator,
                cute.composition(
                    bgp1.layout, cute.make_layout((128, self.query_tile), stride=(1, 128))
                ),
            )
            sp = cute.make_tensor(
                pp.iterator,
                cute.composition(
                    pp.layout, cute.make_layout((64, self.query_tile), stride=(1, 64))
                ),
            )
            bg = cute.make_tensor(
                bgp.iterator,
                cute.composition(
                    bgp.layout, cute.make_layout((128, self.query_tile), stride=(1, 128))
                ),
            )
            if tid == 0:
                cute.arch.mbarrier_init(bar, 1)
                cute.arch.mbarrier_init(bar + 1, 1)
                cute.arch.mbarrier_init(loadbar, 1)
                cute.arch.mbarrier_init(ready, 1)
                cute.arch.mbarrier_init(ready + 1, 1)
                cute.arch.mbarrier_init(compute_ready, 1)
                cute.arch.mbarrier_init(ds_ready, 2)
            cute.arch.mbarrier_init_fence()
            cute.arch.barrier()
            tmem = utils.TmemAllocator(
                ptrslot,
                barrier_for_retrieve=cutlass.pipeline.NamedBarrier(
                    barrier_id=1, num_threads=self.threads
                ),
                is_two_cta=True,
                two_cta_tmem_dealloc_mbar_ptr=dealloc,
            )
            # Both CTAs must finish shared barrier initialization before the
            # allocator accesses its peer CTA. A CTA-only barrier is insufficient.
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            tmem.allocate(512)
            tmem.wait_for_alloc()
            tp = tmem.retrieve_ptr(Float32)
            tmem.relinquish_alloc_permit()
            cg = mg.get_slice(rank).make_fragment_C(
                mg.get_slice(rank).partition_shape_C((128, 256))
            )
            cs = ms.get_slice(rank).make_fragment_C(
                ms.get_slice(rank).partition_shape_C((128, self.query_tile))
            )
            acc0 = cute.make_tensor(tp, cg.layout)
            acc1 = cute.make_tensor(tp + 128, cg.layout)
            scores = cute.make_tensor(tp + 256, cs.layout)
            coord_g = mg.get_slice(rank).partition_C(cute.make_identity_tensor((128, 256)))
            initialized = False
            lphase = self.load(ka, kt, sa, ks + rank * 64, kh, kb, 0, loadbar, lphase, offset=ko)
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            compute_sync = cutlass.pipeline.NamedBarrier(barrier_id=2, num_threads=128)
            cphase = Int32(0)
            for it in cutlass.range(iterations):
                qs = it * self.query_tile
                active = True
                if const_expr(self.causal):
                    active = ks <= qs + self.query_tile - 1 + sk - sq
                # DV traverses query_tile rows for a 128-key-row output tile.
                if const_expr(self.window[0] >= 0):
                    active = active and ks + 127 >= qs + sk - sq - self.window[0]
                if const_expr(self.window[1] >= 0 and not self.causal):
                    active = active and ks <= qs + self.query_tile - 1 + sk - sq + self.window[1]
                if active:
                    if warp == 5:
                        if initialized:
                            cute.arch.mbarrier_wait(bar, 1)
                        lphase = self.produce(
                            qba,
                            qbt,
                            sb,
                            qs + rank * (self.query_tile // 2),
                            h,
                            qb,
                            0,
                            loadbar,
                            ready,
                            lphase,
                            offset=qo,
                        )
                        if initialized:
                            cute.arch.mbarrier_wait(bar + 1, 1)
                        lphase = self.produce(
                            qba,
                            qbt,
                            sb1,
                            qs + rank * (self.query_tile // 2),
                            h,
                            qb,
                            256,
                            loadbar,
                            ready + 1,
                            lphase,
                            offset=qo,
                        )
                        cute.arch.mbarrier_wait(bar, 0)
                        lphase = self.produce(
                            oga,
                            ogt,
                            bg,
                            qs,
                            h,
                            qb,
                            rank * 128,
                            loadbar,
                            ready,
                            lphase,
                            True,
                            offset=qo,
                        )
                        cute.arch.mbarrier_wait(bar + 1, 0)
                        lphase = self.produce(
                            oga,
                            ogt,
                            bg1,
                            qs,
                            h,
                            qb,
                            256 + rank * 128,
                            loadbar,
                            ready + 1,
                            lphase,
                            True,
                            offset=qo,
                        )
                    if warp == 4:
                        if rank == 0:
                            self.issue(ms, scores, ap, bp, ready, bar, 0, True, 0)
                            self.issue(ms, scores, ap, bp1, ready + 1, bar + 1, 0, False, 1)
                            with cute.arch.elect_one():
                                tcgen05.commit(
                                    compute_ready,
                                    mask=3,
                                    cta_group=tcgen05.CtaGroup.TWO,
                                )
                            cute.arch.mbarrier_wait(ds_ready, cphase)
                            self.issue(mg, acc0, pp, bgp, ready, bar, 1, not initialized)
                            self.issue(
                                mg,
                                acc1,
                                pp,
                                bgp1,
                                ready + 1,
                                bar + 1,
                                1,
                                not initialized,
                            )
                    if warp < 4:
                        for chunk in cutlass.range_constexpr(self.query_tile // 128):
                            x = Float32(0)
                            pos = tid + chunk * 128
                            if qs + pos < sq:
                                x = lse[qb, h, qo + qs + pos]
                            slse[pos] = x
                        compute_sync.arrive_and_wait()
                        cute.arch.mbarrier_wait(compute_ready, cphase)
                        copy_smem = cute.make_copy_atom(
                            cute.nvgpu.CopyUniversalOp(),
                            self.dtype,
                            num_bits_per_copy=128,
                        )
                        for strip in cutlass.range_constexpr(self.query_tile // 32):
                            physical = cute.make_layout((128, 16), stride=(65536, 1))
                            ts = cute.make_tensor(scores.iterator + strip * 16, physical)
                            cp = tcgen05.make_tmem_copy(
                                cute.make_copy_atom(
                                    tcgen05.Ld32x32bOp(tcgen05.Repetition(16)), Float32
                                ),
                                ts,
                            )
                            th = cp.get_slice(tid)
                            coords = th.partition_D(cute.make_identity_tensor((128, 16)))
                            rs = cute.make_rmem_tensor(coords.shape, Float32)
                            rp = cute.make_rmem_tensor(coords.shape, self.dtype)
                            cute.copy(cp, th.partition_S(ts), rs)
                            cute.arch.fence_view_async_tmem_load()
                            for j in cutlass.range_constexpr(cute.size(rs)):
                                row, col = coords[j]
                                r = rank * 64 + row % 64
                                c = (row // 64) * (self.query_tile // 2) + strip * 16 + col
                                qi = qs + c
                                ki = ks + r
                                valid = qi < sq and ki < sk
                                center = qi + sk - sq
                                if const_expr(self.causal):
                                    valid = valid and ki <= center
                                if const_expr(self.window[0] >= 0):
                                    valid = valid and ki >= center - self.window[0]
                                if const_expr(self.window[1] >= 0):
                                    valid = valid and ki <= center + self.window[1]
                                prob = Float32(0)
                                if valid:
                                    value = rs[j] * scale
                                    if const_expr(self.softcap > 0):
                                        t = (
                                            2
                                            / (
                                                1
                                                + cute.math.exp(
                                                    -2 * value / self.softcap,
                                                    fastmath=True,
                                                )
                                            )
                                            - 1
                                        )
                                        value = self.softcap * t
                                    prob = cute.math.exp(value - slse[qi - qs], fastmath=True)
                                rp[j] = self.dtype(prob)
                            for j in cutlass.range_constexpr(2):
                                row, col = coords[j * 8]
                                r = row % 64
                                c = (row // 64) * (self.query_tile // 2) + strip * 16 + col
                                offset = cute.assume(cute.crd2idx((r, c), sp.layout), divby=8)
                                ptarget = cute.make_tensor(
                                    sp.iterator + offset, cute.make_layout(8)
                                )
                                psource = cute.make_tensor(rp.iterator + j * 8, cute.make_layout(8))
                                cute.copy(copy_smem, psource, ptarget)
                        cute.arch.fence_view_async_shared()
                        compute_sync.arrive_and_wait()
                        if tid == 0:
                            cute.arch.mbarrier_arrive(ds_ready, peer_cta_rank_in_cluster=0)
                    initialized = True
                    cphase = cphase ^ 1
            cute.arch.barrier()
            if initialized:
                cute.arch.mbarrier_wait(bar + 1, 1)
            cute.arch.barrier()
            if initialized:
                self.epilogue(
                    acc0,
                    coord_g,
                    sa_full.iterator,
                    dv,
                    kb,
                    h,
                    ko + ks,
                    ko + sk_storage,
                    0,
                )
                self.epilogue(
                    acc1,
                    coord_g,
                    sa_full.iterator,
                    dv,
                    kb,
                    h,
                    ko + ks,
                    ko + sk_storage,
                    256,
                )
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            tmem.free(tp)


class ReduceD512Gqa:
    """Sum per-query-head partial gradients in FP32 and convert once."""

    def __init__(self, ratio):
        self.ratio = ratio

    @cute.jit
    def __call__(self, pk, pv, dk, dv, stream: cuda.CUstream):
        count = cute.size(dk)
        self.kernel(pk, pv, dk, dv, count).launch(
            grid=(cute.ceil_div(count, 1024), 1, 1), block=(256, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, pk, pv, dk, dv, count: Int32):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        index = (block * 256 + tid) * 4
        if index < count:
            col = index % 512
            head = (index // 512) % dk.shape[2]
            row = (index // (512 * dk.shape[2])) % dk.shape[1]
            batch = index // (512 * dk.shape[2] * dk.shape[1])
            load_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128
            )
            store_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                dk.element_type,
                num_bits_per_copy=4 * dk.element_type.width,
            )
            ak = cute.make_rmem_tensor((4,), Float32)
            av = cute.make_rmem_tensor((4,), Float32)
            rk = cute.make_rmem_tensor((4,), Float32)
            rv = cute.make_rmem_tensor((4,), Float32)
            ak.fill(0)
            av.fill(0)
            for group in cutlass.range_constexpr(self.ratio):
                off = cute.assume(
                    cute.crd2idx(
                        (Int64(batch), Int64(row), Int64(head) * self.ratio + group, Int64(col)),
                        pk.layout,
                    ),
                    divby=4,
                )
                cute.copy(
                    load_atom,
                    cute.make_tensor(pk.iterator + off, cute.make_layout(4)),
                    rk,
                )
                cute.copy(
                    load_atom,
                    cute.make_tensor(pv.iterator + off, cute.make_layout(4)),
                    rv,
                )
                ak.store(ak.load() + rk.load())
                av.store(av.load() + rv.load())
            ok = cute.make_rmem_tensor((4,), dk.element_type)
            ov = cute.make_rmem_tensor((4,), dv.element_type)
            ok.store(ak.load().to(dk.element_type))
            ov.store(av.load().to(dv.element_type))
            offk = cute.assume(
                cute.crd2idx((Int64(batch), Int64(row), Int64(head), Int64(col)), dk.layout),
                divby=4,
            )
            offv = cute.assume(
                cute.crd2idx((Int64(batch), Int64(row), Int64(head), Int64(col)), dv.layout),
                divby=4,
            )
            cute.copy(
                store_atom,
                ok,
                cute.make_tensor(dk.iterator + offk, cute.make_layout(4)),
            )
            cute.copy(
                store_atom,
                ov,
                cute.make_tensor(dv.iterator + offv, cute.make_layout(4)),
            )


def _tensor_signature(tensor):
    if tensor is None:
        return None
    return (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)


def backward_sm100_d512(
    q,
    k,
    v,
    out,
    dout,
    lse,
    dq,
    dk,
    dv,
    scale,
    causal,
    softcap,
    window_left,
    window_right,
    cuq,
    cuk,
    usedq,
    usedk,
    maxsq,
    maxsk,
    dlse,
    *,
    arch: int,
    fake_mode=False,
):
    """Launch native D512 backward after the public interface validates inputs."""
    import torch
    from cutlass.cute.runtime import from_dlpack
    from flash_attn.cute.interface import _bwd_preprocess, torch2cute_dtype_map

    if fake_mode:
        return dq, dk, dv
    if q.numel() == 0 or k.numel() == 0:
        return dq.zero_(), dk.zero_(), dv.zero_()

    # Use the existing FA4 preprocessor and its padded varlen offsets. A packed
    # non-padded output would let one sequence's padding overwrite its neighbor.
    hq, hk = q.shape[-2], k.shape[-2]
    ratio = hq // hk
    if cuq is None:
        delta_shape = (q.shape[0], hq, ((q.shape[1] + 127) // 128) * 128)
    else:
        length = ((q.shape[0] + cuq.shape[0] * 128 - 1) // 128) * 128
        delta_shape = (hq, length)
    delta = torch.empty(delta_shape, device=q.device, dtype=torch.float32)
    lse_log2 = torch.empty_like(delta)
    _bwd_preprocess(
        out,
        dout,
        delta,
        lse,
        lse_log2,
        None,
        cuq,
        usedq,
        dlse,
        torch2cute_dtype_map[q.dtype],
        512,
        512,
        128,
        fake_mode=False,
    )

    # Preserve supplied output buffers, including views with a strided last axis.
    outputs = (dq, dk, dv)
    work_outputs = tuple(
        t
        if t.stride(-1) == 1 and t.data_ptr() % 16 == 0 and all(s % 4 == 0 for s in t.stride()[:-1])
        else torch.empty(t.shape, device=t.device, dtype=t.dtype)
        for t in outputs
    )
    work_dq, work_dk, work_dv = work_outputs
    # Dense square attention writes every gradient element. Other layouts may
    # contain empty or unused rows and need zero initialization.
    zero_outputs = (
        any(x is not None for x in (cuq, cuk, usedq, usedk)) or q.shape[-3] != k.shape[-3]
    )
    if zero_outputs:
        work_dq.zero_()
    if ratio == 1:
        pk, pv = work_dk, work_dv
        if zero_outputs:
            pk.zero_()
            pv.zero_()
    else:
        partial_shape = (*k.shape[:-2], hq, 512)
        pk = torch.empty(partial_shape, device=k.device, dtype=torch.float32)
        pv = torch.empty_like(pk)
        if zero_outputs:
            pk.zero_()
            pv.zero_()

    def batch_view(t):
        return t.unsqueeze(0) if t.ndim == 3 else t

    tensors = tuple(batch_view(t) for t in (q, k, v, dout))
    lse_view = lse.unsqueeze(0) if cuq is not None else lse
    delta_view = delta.unsqueeze(0) if cuq is not None else delta
    tensors += (
        lse_view,
        delta_view,
        batch_view(work_dq),
        batch_view(pk),
        batch_view(pv),
    )
    metadata = (cuq, cuk, usedq, usedk)
    # Tensor-valued maximum lengths must not be read back to the host during a
    # CUDA graph capture. The physical length is a safe launch upper bound.
    maxsq = maxsq if isinstance(maxsq, int) else tensors[0].shape[1]
    maxsk = maxsk if isinstance(maxsk, int) else tensors[1].shape[1]
    window = tuple(-1 if x is None else x for x in (window_left, window_right))
    signature = tuple(_tensor_signature(t) for t in (*tensors, *metadata))
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    for mode in ("dq", "dk", "dv"):
        key = (arch, mode, signature, causal, softcap, window, maxsq, maxsk)
        if key not in _native_cache:
            args = [from_dlpack(t.detach(), assumed_align=16) for t in tensors]
            meta = [from_dlpack(t, assumed_align=4) if t is not None else None for t in metadata]
            kernel = (NativeD512Dv if mode == "dv" else NativeD512DqDk)(
                mode, causal, softcap, window, maxsq, maxsk
            )
            _native_cache[key] = cute.compile(
                kernel,
                *args,
                Float32(0),
                *meta,
                stream,
                options="--enable-tvm-ffi --ptxas-options='--minnctapersm=1 --maxntid=256'",
            )
        _native_cache[key](*tensors, scale, *metadata)

    if ratio > 1:
        reduce_tensors = tuple(batch_view(t) for t in (pk, pv, work_dk, work_dv))
        key = (arch, ratio, tuple(_tensor_signature(t) for t in reduce_tensors))
        if key not in _reduce_cache:
            args = [from_dlpack(t.detach(), assumed_align=16) for t in reduce_tensors]
            _reduce_cache[key] = cute.compile(
                ReduceD512Gqa(ratio), *args, stream, options="--enable-tvm-ffi"
            )
        _reduce_cache[key](*reduce_tensors)
    for output, work in zip(outputs, work_outputs):
        if output is not work:
            output.copy_(work)
    return outputs


_native_cache = get_jit_cache("bwd_sm100_d512_pipeline")
_reduce_cache = get_jit_cache("bwd_sm100_d512_pipeline_reduce")
