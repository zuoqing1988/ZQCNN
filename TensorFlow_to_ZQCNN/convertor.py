from tensorflow.python.framework import tensor_util
from google.protobuf import text_format
#import tensorflow as tf
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.platform import gfile
from tensorflow.python.framework import tensor_util
import struct
import os,sys
import numpy as np

def search_node(all_node, name):
    for i,n in enumerate(all_node):
        if n.name == name:
            return n
    return None
	
def get_NCHW(n):
    tensor = n.attr["value"]
    #print(type(tensor))
    #print(dir(tensor))
    tensor_tensor = tensor.tensor
    #print(type(tensor_tensor))
    #print(dir(tensor_tensor))
    #print(tensor_tensor)
    tensor_shape = tensor_tensor.tensor_shape
    #print(type(tensor_shape))
    #print(dir(tensor_shape))
    #print(tensor_shape)
    tensor_shape_dim = tensor_shape.dim
    #print(type(tensor_shape_dim))
    #print(dir(tensor_shape_dim))
    #print(tensor_shape_dim)
    dim_num = len(tensor_shape_dim)
    N,C,H,W = [1,1,1,1]
    if dim_num == 4:
        H = tensor_shape_dim[0].size
        W = tensor_shape_dim[1].size
        C = tensor_shape_dim[2].size
        N = tensor_shape_dim[3].size

    elif dim_num == 3:
        H = tensor_shape_dim[0].size
        W = tensor_shape_dim[1].size
        C = tensor_shape_dim[2].size
    elif dim_num == 2:
        C = tensor_shape_dim[0].size
        N = tensor_shape_dim[1].size
    elif dim_num == 1:
        C = tensor_shape_dim[0].size
    return [N,C,H,W]
	
def put_rnn_node_binaray_to_file_split(fout2, fw_kernel_node, fw_bias_node, bw_kernel_node, bw_bias_node):
    # fw_kernel
    tensor = fw_kernel_node.attr["value"]
    tensor_tensor = tensor.tensor
    tensor_shape = tensor_tensor.tensor_shape
    tensor_shape_dim = tensor_shape.dim
    tensor_content = tensor_tensor.tensor_content
    num_bytes = len(tensor_content)
    num_float = int(num_bytes/4)
    fw_N = tensor_shape_dim[1].size
    fw_C = tensor_shape_dim[0].size
    fw_hidden_dim = fw_N // 4
    fw_input_dim = fw_C - fw_hidden_dim
    fw_float_weights_ori = struct.unpack('<%df'%num_float, struct.pack('%dB'%num_bytes, *tensor_content))
    fw_float_weights = HWCN_to_NCHW(fw_float_weights_ori,fw_N,fw_C,1,1)

    #fw_bias    
    tensor = fw_bias_node.attr["value"]
    tensor_tensor = tensor.tensor
    tensor_shape = tensor_tensor.tensor_shape
    tensor_shape_dim = tensor_shape.dim
    tensor_content = tensor_tensor.tensor_content
    num_bytes = len(tensor_content)
    num_float = int(num_bytes/4)
    fw_float_bias = struct.unpack('<%df'%num_float, struct.pack('%dB'%num_bytes, *tensor_content))

    #bw_kernel
    tensor = bw_kernel_node.attr["value"]
    tensor_tensor = tensor.tensor
    tensor_shape = tensor_tensor.tensor_shape
    tensor_shape_dim = tensor_shape.dim
    tensor_content = tensor_tensor.tensor_content
    num_bytes = len(tensor_content)
    num_float = int(num_bytes/4)
    bw_N = tensor_shape_dim[1].size
    bw_C = tensor_shape_dim[0].size
    bw_hidden_dim = bw_N // 4
    bw_input_dim = bw_C - bw_hidden_dim
    bw_float_weights_ori = struct.unpack('<%df'%num_float, struct.pack('%dB'%num_bytes, *tensor_content))
    bw_float_weights = HWCN_to_NCHW(bw_float_weights_ori,bw_N,bw_C,1,1)
    
    #bw_bias    
    tensor = bw_bias_node.attr["value"]
    tensor_tensor = tensor.tensor
    tensor_shape = tensor_tensor.tensor_shape
    tensor_shape_dim = tensor_shape.dim
    tensor_content = tensor_tensor.tensor_content
    num_bytes = len(tensor_content)
    num_float = int(num_bytes/4)
    bw_float_bias = struct.unpack('<%df'%num_float, struct.pack('%dB'%num_bytes, *tensor_content))

    assert(fw_N == bw_N and fw_C == bw_C)


	
    xc_I = list()
    xc_F = list()
    xc_O = list()
    xc_G = list()
    bias_I = list()
    bias_F = list()
    bias_O = list()
    bias_G = list()
    hc_I = list()
    hc_F = list()
    hc_O = list()
    hc_G = list()
    hidden_dim = fw_hidden_dim
    input_dim = fw_input_dim
    C = fw_C
    for nn in range(hidden_dim):
        nn0 = nn
        nn1 = nn + hidden_dim
        nn2 = nn + hidden_dim*2
        nn3 = nn + hidden_dim*3
        for cc in range(input_dim):
            xc_I.append(fw_float_weights[nn0*C+cc])
            xc_G.append(fw_float_weights[nn1*C+cc])
            xc_F.append(fw_float_weights[nn2*C+cc])
            xc_O.append(fw_float_weights[nn3*C+cc])
        for cc in range(hidden_dim):
            hc_I.append(fw_float_weights[nn0*C+cc+input_dim])
            hc_G.append(fw_float_weights[nn1*C+cc+input_dim])
            hc_F.append(fw_float_weights[nn2*C+cc+input_dim])
            hc_O.append(fw_float_weights[nn3*C+cc+input_dim])
        bias_I.append(fw_float_bias[nn0])
        bias_G.append(fw_float_bias[nn1])
        bias_F.append(fw_float_bias[nn2])
        bias_O.append(fw_float_bias[nn3])
    
    xc = xc_I + xc_F + xc_O + xc_G
    bias = bias_I + bias_F + bias_O + bias_G
    hc = hc_I + hc_F + hc_O + hc_G

    xc_I = list()
    xc_F = list()
    xc_O = list()
    xc_G = list()
    bias_I = list()
    bias_F = list()
    bias_O = list()
    bias_G = list()
    hc_I = list()
    hc_F = list()
    hc_O = list()
    hc_G = list()
    hidden_dim = bw_hidden_dim
    input_dim = bw_input_dim
    C = bw_C
    for nn in range(hidden_dim):
        nn0 = nn
        nn1 = nn + hidden_dim
        nn2 = nn + hidden_dim*2
        nn3 = nn + hidden_dim*3
        for cc in range(input_dim):
            xc_I.append(bw_float_weights[nn0*C+cc])
            xc_G.append(bw_float_weights[nn1*C+cc])
            xc_F.append(bw_float_weights[nn2*C+cc])
            xc_O.append(bw_float_weights[nn3*C+cc])
        for cc in range(hidden_dim):
            hc_I.append(bw_float_weights[nn0*C+cc+input_dim])
            hc_G.append(bw_float_weights[nn1*C+cc+input_dim])
            hc_F.append(bw_float_weights[nn2*C+cc+input_dim])
            hc_O.append(bw_float_weights[nn3*C+cc+input_dim])
        bias_I.append(bw_float_bias[nn0])
        bias_G.append(bw_float_bias[nn1])
        bias_F.append(bw_float_bias[nn2])
        bias_O.append(bw_float_bias[nn3])

    xc = xc + xc_I + xc_F + xc_O + xc_G
    bias = bias + bias_I + bias_F + bias_O + bias_G
    hc = hc + hc_I + hc_F + hc_O + hc_G

    fout2.write(struct.pack('%df'%(len(xc)), *xc))
    fout2.write(struct.pack('%df'%(len(bias)), *bias))
    fout2.write(struct.pack('%df'%(len(hc)), *hc))
    return hidden_dim
	
def put_rnn_node_binaray_to_file(fout2, fw_kernel_node, fw_bias_node, bw_kernel_node, bw_bias_node):
    # fw_kernel
    tensor = fw_kernel_node.attr["value"]
    tensor_tensor = tensor.tensor
    tensor_shape = tensor_tensor.tensor_shape
    tensor_shape_dim = tensor_shape.dim
    tensor_content = tensor_tensor.tensor_content
    num_bytes = len(tensor_content)
    num_float = int(num_bytes/4)
    fw_N = tensor_shape_dim[1].size
    fw_C = tensor_shape_dim[0].size
    fw_hidden_dim = fw_N // 4
    fw_input_dim = fw_C - fw_hidden_dim
    fw_float_weights_ori = struct.unpack('<%df'%num_float, struct.pack('%dB'%num_bytes, *tensor_content))
    fw_float_weights = HWCN_to_NCHW(fw_float_weights_ori,fw_N,fw_C,1,1)
    #fw_float_weights = fw_float_weights_ori

    #fw_bias    
    tensor = fw_bias_node.attr["value"]
    tensor_tensor = tensor.tensor
    tensor_shape = tensor_tensor.tensor_shape
    tensor_shape_dim = tensor_shape.dim
    tensor_content = tensor_tensor.tensor_content
    num_bytes = len(tensor_content)
    num_float = int(num_bytes/4)
    fw_float_bias = struct.unpack('<%df'%num_float, struct.pack('%dB'%num_bytes, *tensor_content))

    #bw_kernel
    tensor = bw_kernel_node.attr["value"]
    tensor_tensor = tensor.tensor
    tensor_shape = tensor_tensor.tensor_shape
    tensor_shape_dim = tensor_shape.dim
    tensor_content = tensor_tensor.tensor_content
    num_bytes = len(tensor_content)
    num_float = int(num_bytes/4)
    bw_N = tensor_shape_dim[1].size
    bw_C = tensor_shape_dim[0].size
    bw_hidden_dim = bw_N // 4
    bw_input_dim = bw_C - bw_hidden_dim
    bw_float_weights_ori = struct.unpack('<%df'%num_float, struct.pack('%dB'%num_bytes, *tensor_content))
    bw_float_weights = HWCN_to_NCHW(bw_float_weights_ori,bw_N,bw_C,1,1)
    #bw_float_weights = bw_float_weights_ori
    
    #bw_bias    
    tensor = bw_bias_node.attr["value"]
    tensor_tensor = tensor.tensor
    tensor_shape = tensor_tensor.tensor_shape
    tensor_shape_dim = tensor_shape.dim
    tensor_content = tensor_tensor.tensor_content
    num_bytes = len(tensor_content)
    num_float = int(num_bytes/4)
    bw_float_bias = struct.unpack('<%df'%num_float, struct.pack('%dB'%num_bytes, *tensor_content))

    assert(fw_N == bw_N and fw_C == bw_C)

    fout2.write(struct.pack('%df'%(fw_C*fw_N), *fw_float_weights))
    fout2.write(struct.pack('%df'%(fw_N), *fw_float_bias))
    fout2.write(struct.pack('%df'%(bw_C*bw_N), *bw_float_weights))
    fout2.write(struct.pack('%df'%(bw_N), *bw_float_bias))
    
    return fw_hidden_dim
	
	
def put_node_binaray_to_file(fout2, n, conv2d_transpose = False, need_add_eps = False, eps = 0.001):
    tensor = n.attr["value"]
    #print(type(tensor))
    #print(dir(tensor))
    tensor_tensor = tensor.tensor
    #print(type(tensor_tensor))
    #print(dir(tensor_tensor))
    #print(tensor_tensor)
    tensor_shape = tensor_tensor.tensor_shape
    #print(type(tensor_shape))
    #print(dir(tensor_shape))
    #print(tensor_shape)
    tensor_shape_dim = tensor_shape.dim
    #print(type(tensor_shape_dim))
    #print(dir(tensor_shape_dim))
    #print(tensor_shape_dim)
    dim_num = len(tensor_shape_dim)
    N,C,H,W = [1,1,1,1]
    if dim_num == 4:
        H = tensor_shape_dim[0].size
        W = tensor_shape_dim[1].size
        C = tensor_shape_dim[2].size
        N = tensor_shape_dim[3].size

    elif dim_num == 3:
        H = tensor_shape_dim[0].size
        W = tensor_shape_dim[1].size
        C = tensor_shape_dim[2].size
    elif dim_num == 2:
        C = tensor_shape_dim[0].size
        N = tensor_shape_dim[1].size
    elif dim_num == 1:
        C = tensor_shape_dim[0].size
    
    tensor_content = tensor_tensor.tensor_content
    num_bytes = len(tensor_content)
    if num_bytes == 0:
        #print('num_float=0')
        num_float = N*C*H*W
        float_val = tensor_tensor.float_val[0]
        #print(dir(float_val))
        s = struct.pack('f', float_val)
        for j in range(num_float):
            fout2.write(s)		
    else:
        if dim_num == 1:
            num_float = int(num_bytes/4)
            float_weights = struct.unpack('<%df'%num_float, struct.pack('%dB'%num_bytes, *tensor_content))
            if need_add_eps:
                float_weights0 = ()
                for k in range(num_float):
                    float_weights0 = float_weights0 + (float_weights[k]+eps,)
                float_weights = float_weights0
            fout2.write(struct.pack('%df'%num_float, *float_weights))
        else:
            num_float = int(num_bytes/4)
            #print('num_float=%d'%num_float)
            #print(type(tensor_content))
            #print(dir(tensor_content))
            float_weights = struct.unpack('<%df'%num_float, struct.pack('%dB'%num_bytes, *tensor_content))
            if need_add_eps:
                float_weights0 = ()
                for k in range(num_float):
                    float_weights0 = float_weights0 + (float_weights[k]+eps,)
                float_weights = float_weights0
            if conv2d_transpose:
                float_weights1 = _H_WNC_to_NCHW(float_weights,N,C,H,W)
            else:
                float_weights1 = HWCN_to_NCHW(float_weights,N,C,H,W)
            #print(float_weights)
            #print(float_weights1)
            #fout2.write(struct.pack('%dB'%num_bytes, *tensor_content))
            fout2.write(struct.pack('%df'%num_float, *float_weights1))
		
def HWCN_to_NCHW(in_data, N, C, H, W):
    out_data = list()
    num_float = N*C*H*W
    WCN = W*C*N
    CN = C*N
        
    for n in range(N):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    out_data.append(in_data[h*WCN+w*CN+c*N+n])
    return out_data
	
def _H_WNC_to_NCHW(in_data, N, C, H, W):
    out_data = list()
    num_float = N*C*H*W
    WNC = W*N*C
    NC = N*C
        
    for n in range(N):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    out_data.append(in_data[(H-1-h)*WNC+(W-1-w)*NC+n*C+c])
    return out_data

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
GRAPH_PB_PATH = './model-zq8-gray-660000.pb' #path to your .pb file
fout = open("headposegaze-112-gray.zqparams","w")
fout2 = open("headposegaze-112-gray.nchwbin","wb")
with tf.Session() as sess:
    #print("load graph")
    with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
        graph_def = tf.GraphDef()
        # Note: one of the following two lines work if required libraries are available
        #text_format.Merge(f.read(), graph_def)
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        #print(graph_def)
        all_node = graph_def.node;
        #print(type(all_node))
        #print(dir(all_node))
        node_num = len(all_node)
		
        all_visited_node_names = []
        visited_rnn_cell_name = []
        for i,n in enumerate(all_node):
            #print("Name of the node - %s" % n.name)
       
            rnn_key_name = 'stack_bidirectional_rnn'
            if(rnn_key_name in n.name):
                tmp_splits = n.name.split(rnn_key_name) 
                prefix_name = tmp_splits[0]
                cell_name = tmp_splits[1].split('/')[1]
                has_visited = False
                for j,nn in enumerate(visited_rnn_cell_name):
                    if cell_name == nn:
                        has_visited = True
                        break
                if has_visited:
                    continue
                #print(cell_name)
                #for jj,vv in enumerate( all_visited_node_names):
                #    print('%3d:%s'%(jj,vv))
                cell_nodes_ids = list()
                for j,nn in enumerate(all_node):
                    if rnn_key_name in nn.name:
                        j_tmp_splits = nn.name.split(rnn_key_name) 
                        j_prefix_name = j_tmp_splits[0]
                        j_cell_name = j_tmp_splits[1].split('/')[1]
                        if prefix_name == j_prefix_name and cell_name == j_cell_name:
                            cell_nodes_ids.append(j)
                #for jj,vv in enumerate( cell_nodes_ids):
                #    print('%3d:%s'%(jj,all_node[vv].name))
                visited_node_num = len(all_visited_node_names)
                has_found_input_name = False
                input_name = None
                for j in range(len(cell_nodes_ids)):
                    nn = all_node[cell_nodes_ids[j]]
                    node_input = nn.input
                    input_num = len(node_input)
                    for ii in range(input_num):
                        for jj in range(visited_node_num):
                            if node_input[ii] == all_visited_node_names[jj]:
                                has_found_input_name = True
                                input_name = node_input[ii]
                                break
                        if has_found_input_name:
                            break
                    if has_found_input_name:
                        break

                output_name = all_node[cell_nodes_ids[-1]].name
                visited_rnn_cell_name.append(cell_name)
                all_visited_node_names.append(output_name)
                					
                
                #print(cell_name)
                fw_kernel_name = prefix_name + rnn_key_name + '/' + cell_name + '/bidirectional_rnn/fw/basic_lstm_cell/kernel'
                fw_bias_name = prefix_name + rnn_key_name + '/' + cell_name + '/bidirectional_rnn/fw/basic_lstm_cell/bias'
                bw_kernel_name = prefix_name + rnn_key_name + '/' + cell_name + '/bidirectional_rnn/bw/basic_lstm_cell/kernel'
                bw_bias_name = prefix_name + rnn_key_name + '/' + cell_name + '/bidirectional_rnn/bw/basic_lstm_cell/bias'
                #print(fw_kernel_name)
                #print(fw_bias_name)
                #print(bw_kernel_name)
                #print(bw_bias_name)
                fw_kernel_node = search_node(all_node,fw_kernel_name)
                fw_bias_node = search_node(all_node,fw_bias_name)
                bw_kernel_node = search_node(all_node,bw_kernel_name)
                bw_bias_node = search_node(all_node,bw_bias_name)
                hidden_dim = put_rnn_node_binaray_to_file_split(fout2, fw_kernel_node, fw_bias_node, bw_kernel_node, bw_bias_node)
                
                line = 'LSTM_TF name=' + prefix_name + cell_name + ' bottom=%s top=%s type=2 hidden_dim=%d\n'%(input_name, output_name, hidden_dim)
                fout.write(line)
                continue
			
			
            all_visited_node_names.append(n.name)
            line = ''	
            if n.op == 'Conv2D':
                # write .nchwbin file 
                conv_name = n.name.replace('/Conv2D','')
                weight_name = conv_name+'/weights'
                weight_read_name = weight_name + '/read'
                bias_name = conv_name+'/bias'
                bias_read_name = bias_name+'/read'
                weight_node = search_node(all_node,weight_name)
                bias_node = search_node(all_node,bias_name)
                #print(weight_node)
                #print(bias_node)
                put_node_binaray_to_file(fout2, weight_node)
                if bias_node is None:
                    pass
                else:
                    put_node_binaray_to_file(fout2, bias_node)
					
                # write .zqparams file
                [N,C,H,W] = get_NCHW(weight_node)
                dilation = n.attr["dilations"].list
                #print(dilation)
                #print(type(dilation))
                #print(dir(dilation))
                dilation_H = int(dilation.i[1])
                dilation_W = int(dilation.i[2])
                stride = n.attr["strides"].list
                stride_H = int(stride.i[1])
                stride_W = int(stride.i[2])
                padding = str(n.attr["padding"].s,'utf-8')
                #print(type(padding))
                #print(dir(padding))
                #print(padding)
                line = 'Convolution name=' + n.name
                node_input = n.input
                in_num = len(node_input)
                for j in range(in_num):
                    if node_input[j] == weight_read_name or node_input[j] == bias_read_name:
                        pass
                    else:
                        line = line + ' bottom=%s'%node_input[j]
                line = line + ' top=%s'%n.name + ' num_output=%d kernel_H=%d kernel_W=%d dilate_H=%d dilate_W=%d stride_H=%d stride_W=%d pad_type=%s'%(N,H,W,dilation_H,dilation_W,stride_H,stride_W,padding)
                if bias_node is None:
                    pass
                else:
                    line = line + ' bias'				
                line = line + '\n'
				
				
            elif n.op == 'DepthwiseConv2dNative':
                # write .nchwbin file 
                conv_name = n.name
                weight_name = conv_name+'_weights'
                weight_read_name = weight_name + '/read'
                bias_name = conv_name+'_bias'
                bias_read_name = bias_name+'/read'
                weight_node = search_node(all_node,weight_name)
                bias_node = search_node(all_node,bias_name)
                put_node_binaray_to_file(fout2, weight_node)
                if bias_node is None:
                    pass
                else:
                    put_node_binaray_to_file(fout2, bias_node)
					
                # write .zqparams file
                [N,C,H,W] = get_NCHW(weight_node)
                dilation = n.attr["dilations"].list
                #print(dilation)
                #print(type(dilation))
                #print(dir(dilation))
                dilation_H = int(dilation.i[1])
                dilation_W = int(dilation.i[2])
                stride = n.attr["strides"].list
                stride_H = int(stride.i[1])
                stride_W = int(stride.i[2])
                padding = str(n.attr["padding"].s,'utf-8')
                line = 'DepthwiseConvolution name=' + n.name
                node_input = n.input
                in_num = len(node_input)
                for j in range(in_num):
                    if node_input[j] == weight_read_name or node_input[j] == bias_read_name:
                        pass
                    else:
                        line = line + ' bottom=%s'%node_input[j]
                line = line + ' top=%s'%n.name + ' num_output=%d kernel_H=%d kernel_W=%d dilate_H=%d dilate_W=%d stride_H=%d stride_W=%d pad_type=%s'%(C,H,W,dilation_H,dilation_W,stride_H,stride_W,padding)
                if bias_node is None:
                    pass
                else:
                    line = line + ' bias'				
                line = line + '\n'
				
            elif n.op == 'Conv2DBackpropInput':
                # write .nchwbin file 
                conv_name = n.name.replace('/conv2d_transpose','')
                stack_name = conv_name+'/stack'
                weight_name = conv_name+'/weights'
                weight_read_name = weight_name + '/read'
                weight_node = search_node(all_node,weight_name)
                put_node_binaray_to_file(fout2, weight_node, True)
                	
                # write .zqparams file
                [N,C,H,W] = get_NCHW(weight_node)
                dilation = n.attr["dilations"].list
                #print(dilation)
                #print(type(dilation))
                #print(dir(dilation))
                dilation_H = int(dilation.i[1])
                dilation_W = int(dilation.i[2])
                stride = n.attr["strides"].list
                stride_H = int(stride.i[1])
                stride_W = int(stride.i[2])
                padding = str(n.attr["padding"].s,'utf-8')
                #print(type(padding))
                #print(dir(padding))
                #print(padding)
                line = 'DeConvolution name=' + n.name
                node_input = n.input
                in_num = len(node_input)
                for j in range(in_num):
                    if node_input[j] == weight_read_name or node_input[j] == stack_name:
                        pass
                    else:
                        line = line + ' bottom=%s'%node_input[j]
                line = line + ' top=%s'%n.name + ' num_output=%d kernel_H=%d kernel_W=%d dilate_H=%d dilate_W=%d stride_H=%d stride_W=%d pad_type=%s'%(C,H,W,dilation_H,dilation_W,stride_H,stride_W,padding)
                line = line + '\n'
				
            elif n.op == 'BiasAdd':
                # write .nchwbin file 
                BiasAdd_name = n.name.replace('/BiasAdd','')
                bias_name = conv_name+'/biases'
                bias_read_name = bias_name+'/read'
                bias_node = search_node(all_node,bias_name)
                put_node_binaray_to_file(fout2, bias_node)
					
                # write .zqparams file
                
                line = 'AddBias name=' + n.name
                node_input = n.input
                in_num = len(node_input)
                for j in range(in_num):
                    if node_input[j] == bias_read_name:
                        pass
                    else:
                        line = line + ' bottom=%s'%node_input[j]
                line = line + ' top=%s'%n.name
                line = line + '\n'
				
            elif n.op == 'FusedBatchNorm':
                # write .nchwbin file 
                eps = n.attr["epsilon"].f
                #print(type(eps))
                #print(dir(eps))
                #print(eps)
                bn_name = n.name.replace('/FusedBatchNorm','')
                scale_const_name = bn_name+'/Const'
                scale_name = bn_name+'/scale'
                scale_read_name = scale_name+'/read'
                bias_name = bn_name+'/beta'
                bias_read_name = bias_name + '/read'
                mean_name = bn_name+'/moving_mean'
                mean_read_name = mean_name+'/read'
                variance_name = bn_name+'/moving_variance'
                variance_read_name = variance_name+'/read'
                scale_const_node = search_node(all_node,scale_const_name)
                scale_node = search_node(all_node,scale_name)
                bias_node = search_node(all_node,bias_name)
                mean_node = search_node(all_node,mean_name)
                variance_node = search_node(all_node,variance_name)
                if mean_node is None:
                    print('Error: mean_node is None in FusedBatchNorm Layer name: %s'%(n.name))
                    print('Maybe you forget to set training=False')
                    sys.exit(0)
                else:
                    put_node_binaray_to_file(fout2, mean_node)
                if variance_node is None:
                    print('Error: mean_node is None in FusedBatchNorm Layer name: %s'%(n.name))
                    print('Maybe you forget to set training=False')
                    sys.exit(0)
                else:
                    put_node_binaray_to_file(fout2, variance_node, False, True, eps)
                if scale_const_node is None:
                    pass
                else:
                    put_node_binaray_to_file(fout2, scale_const_node)
                if scale_node is None:
                    pass
                else:
                    put_node_binaray_to_file(fout2, scale_node)
                if bias_node is None:
                    pass
                else:
                    put_node_binaray_to_file(fout2, bias_node)
                
                # write .zqparams file
                line = 'BatchNormScale name=' + n.name
                node_input = n.input
                in_num = len(node_input)
                for j in range(in_num):
                    if node_input[j] == scale_const_name or node_input[j] == scale_read_name or node_input[j] == bias_read_name or node_input[j] == mean_read_name or node_input[j] == variance_read_name:
                        pass
                    else:
                        line = line + ' bottom=%s'%node_input[j]
                line = line + ' top=%s'%n.name
                if bias_node is None:
                    pass
                else:
                    line = line + ' bias'				
                line = line + '\n'
                
            elif n.op == 'FusedBatchNormV3':
                # write .nchwbin file 
                eps = n.attr["epsilon"].f
                #print(type(eps))
                #print(dir(eps))
                #print(eps)
                bn_name = n.name.replace('/FusedBatchNormV3','')
                scale_const_name = bn_name+'/Const'
                scale_name = bn_name+'/scale'
                scale_read_name = scale_name+'/read'
                bias_name = bn_name+'/beta'
                bias_read_name = bias_name + '/read'
                mean_name = bn_name+'/moving_mean'
                mean_read_name = mean_name+'/read'
                variance_name = bn_name+'/moving_variance'
                variance_read_name = variance_name+'/read'
                scale_const_node = search_node(all_node,scale_const_name)
                scale_node = search_node(all_node,scale_name)
                bias_node = search_node(all_node,bias_name)
                mean_node = search_node(all_node,mean_name)
                variance_node = search_node(all_node,variance_name)
                if mean_node is None:
                    print('Error: mean_node is None in FusedBatchNorm Layer name: %s'%(n.name))
                    print('Maybe you forget to set training=False')
                    sys.exit(0)
                else:
                    put_node_binaray_to_file(fout2, mean_node)
                if variance_node is None:
                    print('Error: mean_node is None in FusedBatchNorm Layer name: %s'%(n.name))
                    print('Maybe you forget to set training=False')
                    sys.exit(0)
                else:
                    put_node_binaray_to_file(fout2, variance_node, False, True, eps)
                if scale_const_node is None:
                    pass
                else:
                    put_node_binaray_to_file(fout2, scale_const_node)
                if scale_node is None:
                    pass
                else:
                    put_node_binaray_to_file(fout2, scale_node)
                if bias_node is None:
                    pass
                else:
                    put_node_binaray_to_file(fout2, bias_node)
                
                # write .zqparams file
                line = 'BatchNormScale name=' + n.name
                node_input = n.input
                in_num = len(node_input)
                for j in range(in_num):
                    if node_input[j] == scale_const_name or node_input[j] == scale_read_name or node_input[j] == bias_read_name or node_input[j] == mean_read_name or node_input[j] == variance_read_name:
                        pass
                    else:
                        line = line + ' bottom=%s'%node_input[j]
                line = line + ' top=%s'%n.name
                if bias_node is None:
                    pass
                else:
                    line = line + ' bias'				
                line = line + '\n'
				
				
            elif n.op == 'Relu6':
                line = 'ReLU6 name=' + n.name
                node_input = n.input
                in_num = len(node_input)
                for j in range(in_num):
                    line = line + ' bottom=%s'%node_input[j]
                line = line + ' top=%s'%n.name
                line = line + '\n'
            elif n.op == 'Relu':
                line = 'ReLU name=' + n.name
                node_input = n.input
                in_num = len(node_input)
                for j in range(in_num):
                    line = line + ' bottom=%s'%node_input[j]
                line = line + ' top=%s'%n.name
                line = line + '\n'
            elif n.op == 'Add':
                line = 'Eltwise operation=SUM name=' + n.name
                node_input = n.input
                in_num = len(node_input)
                for j in range(in_num):
                    line = line + ' bottom=%s'%node_input[j]
                line = line + ' top=%s'%n.name
                line = line + '\n'
            elif n.op == 'AddV2':
                line = 'Eltwise operation=SUM name=' + n.name
                node_input = n.input
                in_num = len(node_input)
                for j in range(in_num):
                    line = line + ' bottom=%s'%node_input[j]
                line = line + ' top=%s'%n.name
                line = line + '\n'
            elif n.op == 'ResizeBilinear':
                size_name = n.name + '/size'
                size_node = search_node(all_node, size_name)
                tensor = size_node.attr["value"].tensor
                dst_h,dst_w = [2,2]
                tensor_content = tensor_tensor.tensor_content
                num_bytes = len(tensor_content)
                num_int = int(num_bytes/4)
                dst_size = struct.unpack('<%di'%num_int, struct.pack('%dB'%num_bytes, *tensor_content))
                if num_int == 2:
                    dst_h,dst_w = dst_size
                elif num_int == 1:
                    dst_h = scales[0]
                    dst_w = scales[0]
                line = 'UpSampling sample_type=bilinear name=' + n.name
                node_input = n.input
                in_num = len(node_input)
                for j in range(in_num):
                    if node_input[j] == size_name:
                        pass
                    else:
                        line = line + ' bottom=%s'%node_input[j]
                line = line + ' top=%s dst_h=%f dst_w=%f'%(n.name, dst_h, dst_w)
                line = line + '\n'
            elif n.op == 'ResizeNearestNeighbor':
                size_name = n.name + '/size'
                size_node = search_node(all_node, size_name)
                tensor = size_node.attr["value"].tensor
                dst_h,dst_w = [2,2]
                tensor_content = tensor_tensor.tensor_content
                num_bytes = len(tensor_content)
                num_int = int(num_bytes/4)
                dst_size = struct.unpack('<%di'%num_int, struct.pack('%dB'%num_bytes, *tensor_content))
                if num_int == 2:
                    dst_h,dst_w = dst_size
                elif num_int == 1:
                    dst_h = scales[0]
                    dst_w = scales[0]
                line = 'UpSampling sample_type=nearest name=' + n.name
                node_input = n.input
                in_num = len(node_input)
                for j in range(in_num):
                    if node_input[j] == size_name:
                        pass
                    else:
                        line = line + ' bottom=%s'%node_input[j]
                line = line + ' top=%s dst_h=%f dst_w=%f'%(n.name, dst_h, dst_w)
                line = line + '\n'
            elif n.op == 'MaxPool':
                kernel = n.attr["ksize"].list
                kernel_H = int(kernel.i[1])
                kernel_W = int(kernel.i[2])
                stride = n.attr["strides"].list
                stride_H = int(stride.i[1])
                stride_W = int(stride.i[2])
                padding = str(n.attr["padding"].s,'utf-8')
                line = 'Pooling pool=MAX name=' + n.name
                node_input = n.input
                in_num = len(node_input)
                for j in range(in_num):
                    line = line + ' bottom=%s'%node_input[j]
                line = line + ' top=%s'%n.name + ' kernel_H=%d kernel_W=%d stride_H=%d stride_W=%d pad_type=%s'%(kernel_H,kernel_W,stride_H,stride_W,padding)
                line = line + '\n'
            elif n.op == 'AvgPool':
                kernel = n.attr["ksize"].list
                kernel_H = int(kernel.i[1])
                kernel_W = int(kernel.i[2])
                stride = n.attr["strides"].list
                stride_H = int(stride.i[1])
                stride_W = int(stride.i[2])
                padding = str(n.attr["padding"].s,'utf-8')
                line = 'Pooling pool=AVG name=' + n.name
                node_input = n.input
                in_num = len(node_input)
                for j in range(in_num):
                    line = line + ' bottom=%s'%node_input[j]
                line = line + ' top=%s'%n.name + ' kernel_H=%d kernel_W=%d stride_H=%d stride_W=%d pad_type=%s'%(kernel_H,kernel_W,stride_H,stride_W,padding)
                line = line + '\n'
            elif n.op == 'ConcatV2':
                axis_name = n.name+'/axis'
                axis_node = search_node(all_node,axis_name)
                axis = 1
                map_nhwc_to_nchw = [0,2,3,1]
                if axis_node is None:
                    pass
                else:
                    axis = axis_node.attr["value"].tensor.int_val[0]
                    #print(axis)
                    #print(dir(axis))
                    axis = map_nhwc_to_nchw[int(axis)]
                line = 'Concat name=' + n.name
                node_input = n.input
                in_num = len(node_input)
                for j in range(in_num):
                    if node_input[j] == axis_name:
                        pass
                    else:
                        line = line + ' bottom=%s'%node_input[j]
                line = line + ' top=%s'%n.name + ' axis=%d'%axis
                line = line + '\n'
            elif n.op == 'Identity':
                pass
            elif n.op == 'Placeholder':
                line = 'Input' + ' name=' + n.name
                
                shape = n.attr["shape"]
                shape_shape_dim = shape.shape.dim
                #print(dir(shape_shape_dim[0]))
                #print(shape_shape_dim[0].ListFields())
                #print(shape_shape_dim[0].size)
                #size_vals = shape_shape_dim.__getattribute__('size')
                #print(size_vals)
                size_num = len(shape_shape_dim)
                if size_num == 4:
                    line = line + ' C=%d H=%d W=%d'%(shape_shape_dim[3].size,shape_shape_dim[1].size,shape_shape_dim[2].size)
                
                line = line + '\n'
                
                #print(n)
            elif n.op == 'Const':
                print('name = %s'%n.name)
                tensor = n.attr["value"]
                #print(type(tensor))
                #print(dir(tensor))
                tensor_tensor = tensor.tensor
                #print(type(tensor_tensor))
                #print(dir(tensor_tensor))
                #print(tensor_tensor)
                tensor_shape = tensor_tensor.tensor_shape
                #print(type(tensor_shape))
                #print(dir(tensor_shape))
                #print(tensor_shape)
                tensor_shape_dim = tensor_shape.dim
                #print(type(tensor_shape_dim))
                #print(dir(tensor_shape_dim))
                #print(tensor_shape_dim)
                dim_num = len(tensor_shape_dim)
                N,C,H,W = [1,1,1,1]
                if dim_num == 4:
                    H = tensor_shape_dim[0].size
                    W = tensor_shape_dim[1].size
                    C = tensor_shape_dim[2].size
                    N = tensor_shape_dim[3].size

                elif dim_num == 3:
                    H = tensor_shape_dim[0].size
                    W = tensor_shape_dim[1].size
                    C = tensor_shape_dim[2].size
                elif dim_num == 2:
                    pass
                elif dim_num == 1:
                    C = tensor_shape_dim[0].size
                print([N,C,H,W])
                tensor_content = tensor_tensor.tensor_content
                num_bytes = len(tensor_content)
                if num_bytes == 0:
                    print('num_float=0')
                    num_float = N*C*H*W
                    print(tensor_tensor.float_val)
                else:
                    num_float = int(num_bytes/4)
                    print('num_float=%d'%num_float)
                    #print(type(tensor_content))
                    #print(dir(tensor_content))
                    float_weights = struct.unpack('<%df'%num_float, struct.pack('%dB'%num_bytes, *tensor_content))
                    print(float_weights)
                #print(tensor)
                print('\n\n')
            elif n.op == 'Reshape':
                shape_name = n.name+'/shape'
                shape_node = search_node(all_node,shape_name)
                tensor = shape_node.attr["value"]
                tensor_tensor = tensor.tensor
                #tensor_shape = tensor_tensor.tensor_shape
                tensor_content = tensor_tensor.tensor_content
                num_bytes = len(tensor_content)
                num_int = int(num_bytes/4)
                print('num_int=%d'%num_int)
                int_vals = struct.unpack('<%di'%num_int, struct.pack('%dB'%num_bytes, *tensor_content))
                print(int_vals)
                N,C,H,W = [1,1,1,1]
                if num_int == 1:
                    C = int_vals[0]
                elif num_int == 2:
                    W = int_vals[0]
                    C = int_vals[1]
                elif num_int == 3:
                    N = int_vals[0]
                    W = int_vals[1]
                    C = int_vals[2]
                elif num_int == 4:
                    N = int_vals[0]
                    H = int_vals[1]
                    W = int_vals[2]
                    C = int_vals[3]

                line = 'Reshape name=' + n.name
                node_input = n.input
                in_num = len(node_input)
                for j in range(in_num):
                    if node_input[j] == shape_name:
                        pass
                    else:
                        line = line + ' bottom=%s'%node_input[j]
                line = line + ' top=%s'%n.name + ' dim=%d dim=%d dim=%d dim=%d'%(N,C,H,W)
                line = line + '\n'
            elif n.op == 'MatMul':
                node_input = n.input
                input_name0 = node_input[0]
                input_name1 = node_input[1]
                if input_name1.split('/')[-1] == 'read':
                    input_name1 = input_name1[:-5]
                    weight_node = search_node(all_node,input_name1)
                    put_node_binaray_to_file(fout2, weight_node)
                    N,C,H,W = get_NCHW(weight_node)
                    line = 'Convolution name=' + n.name
                    node_input = n.input
                    line = line + ' bottom=%s'%input_name0
                    line = line + ' top=%s'%n.name + ' num_output=%d kernel_H=1 kernel_W=1 pad=0'%(N)
                    line = line + '\n'
            elif n.op == 'Squeeze':
                node_input = n.input
                dims = n.attr['squeeze_dims'].list.i
                print(dir(dims))
                print(type(dims))
                map_nhwc_to_nchw = [0,2,3,1]
                len_dim = len(dims)
                line = 'Squeeze name=' + n.name + ' bottom=%s'%node_input[0] + ' top=' + n.name
                for j in range(len_dim):
                    line = line + ' dim=%d'%(map_nhwc_to_nchw[dims[j]])
                line = line + '\n'
            else:
                print('unknown op: %s '%(n.op))
				
            if line == '':
                pass
            else:
                fout.write(line)

fout.close()
fout2.close()
