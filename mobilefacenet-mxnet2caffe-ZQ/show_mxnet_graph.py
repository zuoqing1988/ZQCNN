import mxnet as mx

def load_model_sym(mprefix, epoch):
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix=mprefix,
                                                            epoch=epoch)    
    return sym

if __name__ == "__main__":
    sym = load_model_sym("model_mxnet/face/facega", 0)
    mx.viz.plot_network(sym)        