import numpy as np
import mxnet as mx
from collections import namedtuple

Batch = namedtuple('Batch',['data'])

class PredictorMxNet:
    def __init__(self, mprefix, epoch, size, ctx=mx.cpu()):
        self.size = size
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix=mprefix,
                                                                   epoch=epoch)
        #internals = sym.get_internals()
        #print internals.list_outputs()
        ##data = internals["conv_6sep_relu_output"]
        #pool1 = internals['pool1_output']
        ##sym2 = internals["squeezenetv20_pool3_fwd_output"]
        #group = mx.symbol.Group([sym, pool1])
        self.mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        self.mod.bind(for_training=False, data_shapes=[('data',size)],
                          label_shapes = self.mod._label_shapes)
        self.mod.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)                  
    
    def forward(self, tensor):
        self.mod.forward(Batch([mx.nd.array(tensor)]))
        out = self.mod.get_outputs()
        return out