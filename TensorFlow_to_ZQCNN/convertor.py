from tensorflow.python.framework import tensor_util
from google.protobuf import text_format
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import tensor_util
import struct

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
        pass
    elif dim_num == 1:
        C = tensor_shape_dim[0].size
    return [N,C,H,W]
	
def put_node_binaray_to_file(fout2, n, need_add_eps = False, eps = 0.001):
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

GRAPH_PB_PATH = 'models/model.pb' #path to your .pb file
fout = open("Pose.zqparams","w")
fout2 = open("Pose.nchwbin","wb")
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
        for i,n in enumerate(all_node):
            #print("Name of the node - %s" % n.name)
       
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
                put_node_binaray_to_file(fout2, weight_node)
                if bias_node == None:
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
                if bias_node == None:
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
                if bias_node == None:
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
                if bias_node == None:
                    pass
                else:
                    line = line + ' bias'				
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
                put_node_binaray_to_file(fout2, mean_node)
                put_node_binaray_to_file(fout2, variance_node, True, eps)
                if scale_const_node == None:
                    pass
                else:
                    put_node_binaray_to_file(fout2, scale_const_node)
                if scale_node == None:
                    pass
                else:
                    put_node_binaray_to_file(fout2, scale_node)
                if bias_node == None:
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
                if bias_node == None:
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
                if axis_node == None:
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
            else:
                print('unknown op: %s '%(n.op))
				
            
            fout.write(line)
			
fout.close()
fout2.close()
