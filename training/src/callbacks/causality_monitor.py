
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only

import torch
from torch.autograd import grad

class CausalityMonitor(Callback):
    r"""Monitor causality of a model by tracking gradient leakage  forward in time.
    In a fully causal model, dy[k]du[s] ~= 0 for all k < s.

    Args:
        seq_len (int): Length of the sequence to monitor.
        input_dim (int): Dimension of the input to monitor. If 0, the callback assumes
            the task to be language modeling, and skips the embedding layer. If > 0,
            input_dim is interpreted as the input channel dimension, i.e. D with
            dummy input of dimension [B, L, D].
    
    Notes:
        This callback assumes that `pl_module.model` has a `net` or `s4seq` attribute,
        indicating the primary model to monitor. For LMs, `net` or `s4seq` should 
        be after the embedding layer.
    """

    def __init__(self, seq_len: int  = 10, input_dim: int = 0):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim

    @rank_zero_only
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        model = pl_module.model

        with torch.enable_grad():
            if self.input_dim == 0: 
                # [MP] LongTensors cannot have gradients - we start from post
                # embedding in the LM case
                input_dim = model.d_model
                x = torch.randn((2, self.seq_len, input_dim), \
                    requires_grad=True).to(pl_module.device)
                # [DF] HACK: we need to get the layer that comes after the embedding
                if hasattr(model, 'net'):
                    y = model.net(x)
                else:
                    y = model.s4seq(x)
            else:
                x = torch.randn(1, self.seq_len, self.input_dim, \
                    requires_grad=True).to(pl_module.device)
                y =  model(x)

            stats = {}
            for i in range(self.seq_len):
                # total gradients flowing from y_i to x 
                g =  grad(y[0,0,i].mean(), x, retain_graph=True, allow_unused=True)[0]
                g = g[0,i+1:,:].abs().mean()
                stats[f'stats/causality_{i}'] = g.item()

        if trainer.loggers is not None:
            for logger in trainer.loggers:
                logger.log_metrics(stats, step=trainer.global_step)
