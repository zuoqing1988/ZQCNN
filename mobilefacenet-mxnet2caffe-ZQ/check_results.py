from predictor_caffe import PredictorCaffe
from predictor_mxnet import PredictorMxNet
import numpy as np

def compare_diff_sum(tensor1, tensor2):
    pass

def compare_cosin_dist(tensor1, tensor2):
    pass

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def compare_models(prefix_mxnet, prefix_caffe, size):
    netmx = PredictorMxNet(prefix_mxnet, 0, size)    
    model_file = prefix_caffe + ".prototxt"
    pretrained_file = prefix_caffe + ".caffemodel"
    netcaffe = PredictorCaffe(model_file, pretrained_file, size)
    tensor = np.ones(size, dtype=np.float32)
    out_mx = netmx.forward(tensor)
    print out_mx
    netcaffe.forward(tensor)
    out_caffe = netcaffe.blob_by_name("fc1")
    print out_caffe.data
    #print softmax(out_caffe.data)
    out_caffe = netcaffe.blob_by_name("fc2")
    print out_caffe.data
    #print softmax(out_caffe.data)     
    print "done"
    
if __name__ == "__main__":
    prefix_mxnet = "model_mxnet/face/facega2"
    prefix_caffe = "model_caffe/face/facega2"
    size = (1, 3, 96, 96)
    compare_models(prefix_mxnet, prefix_caffe, size)