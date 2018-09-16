try:
    import mxnet as mx
except ImportError:
    import os, sys
    curr_path = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(os.path.join(curr_path, "/home/chen/Documents/GitHub/MXNet/mxnet-master/mxnet-master/python"))
    import mxnet as mx
