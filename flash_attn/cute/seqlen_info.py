import cutlass
import cutlass.cute as cute


class SeqlenInfo:

    def __init__(self, seqlen_q: cutlass.Int32, seqlen_k: cutlass.Int32):
        self.seqlen_q = seqlen_q
        self.seqlen_k = seqlen_k
