import cutlass
import cutlass.cute as cute


class SeqlenInfo:

    def __init__(self, seqlen_q: cutlass.Int32, seqlen_k: cutlass.Int32, *, loc=None, ip=None):
        self.seqlen_q = seqlen_q
        self.seqlen_k = seqlen_k
        self._loc = loc

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.seqlen_q, self.seqlen_k]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [self.seqlen_q, self.seqlen_k], self._values_pos
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return SeqlenInfo(*(tuple(obj_list)), loc=self._loc)
