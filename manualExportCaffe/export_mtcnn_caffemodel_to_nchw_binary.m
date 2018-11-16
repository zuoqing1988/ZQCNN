%export mtcnn caffemodel to nchw binarys
function export_mtcnn_caffemodel_to_nchw_binary()
caffe.set_mode_cpu();

caffe_net  = caffe.Net('det1.prototxt', 'det1.caffemodel', 'test');
layers = {
    'conv1',2,'none';
    'PReLU1',1,'none';
    'conv2',2,'none';
    'PReLU2',1,'none';
    'conv3',2,'none';
    'PReLU3',1,'none';
    'conv4-1',2,'fc1x1';
    'conv4-2',2,'fc1x1'
    };
export_to_binary(layers,caffe_net,'det1.nchwbin');

caffe_net  = caffe.Net('det2.prototxt', 'det2.caffemodel', 'test');
layers = {
    'conv1',2,'none';
    'prelu1',1,'none';
    'conv2',2,'none';
    'prelu2',1,'none';
    'conv3',2,'none';
    'prelu3',1,'none';
    'conv4',2,'fc3x3';
    'prelu4',1,'none';
    'conv5-1',2,'fc1x1';
    'conv5-2',2,'fc1x1'
    };
export_to_binary(layers,caffe_net,'det2.nchwbin');

caffe_net  = caffe.Net('det3.prototxt', 'det3.caffemodel', 'test');
layers = {
    'conv1',2,'none';
    'prelu1',1,'none';
    'conv2',2,'none';
    'prelu2',1,'none';
    'conv3',2,'none';
    'prelu3',1,'none';
    'conv4',2,'none';
    'prelu4',1,'none';
    'conv5',2,'fc3x3';
    'prelu5',1,'none';
    'conv6-1',2,'fc1x1';
    'conv6-2',2,'fc1x1';
    'conv6-3',2,'fc1x1'
    };
export_to_binary(layers,caffe_net,'det3.nchwbin');

end

function [] = export_to_binary(layers, caffe_net, out_file)
n = size(layers,1);
all = [];
for i = 1:n
    layer_name = layers{i,1};
    flag = layers{i,3};
    for j = 1:uint32((layers{i,2}))
        tmp = caffe_net.params(layer_name,j).get_data();
        if j == 1
            tmp = auto_permute(tmp,flag);
        end
        all =[all;tmp(:)];
    end
end
fid = fopen(out_file,'wb');
fwrite(fid,all,'float');
fclose(fid);
end
    
function [out] = auto_permute(in,flag)
    out = in;
    if strcmp(flag,'fc3x3')==1
        if ndims(in) == 2
            [m,n]=size(in);
            out = reshape(in, [3 3 uint32(m/9) n]);
        end
    elseif strcmp(flag,'fc1x1') == 1
        if ndims(in) == 2
            [m,n]=size(in);
            out = reshape(in, [1 1 m n]);
        end
    end
    
    out = permute(out,[2 1 3 4]);
    
end

    
function [out] = auto_swap_N(in, flag)
    out = in;
    if strcmp(flag, 'none') == 1
        return ;
    elseif strcmp(flag, 'swapN_pair') == 1
        if ndims(in) == 4
            out(:,:,:,1:2:end) = in(:,:,:,2:2:end);
            out(:,:,:,2:2:end) = in(:,:,:,1:2:end);
        else
            out(:,1:2:end) = in(:,2:2:end);
            out(:,2:2:end) = in(:,1:2:end);
        end
    elseif strcmp(flag, 'swapN_half') == 1
        if ndims(in) == 4
            n = size(in,4);
            half_n = uint32(n/2);
            out(:,:,:,1:half_n) = in(:,:,:,half_n+1:end);
            out(:,:,:,half_n+1:end) = in(:,:,:,1:half_n);
        else
            n = size(in,2);
            half_n = uint32(n/2);
            out(:,1:half_n) = in(:,half_n+1:end);
            out(:,half_n+1:end) = in(:,1:half_n);
        end
    end
end

function [out] = auto_swap_C(in, flag)
    out = in;
    if strcmp(flag, 'none') == 1
        return ;
    elseif strcmp(flag, 'swapN_pair') == 1
        if ndims(in) == 4
            out(:,:,1:2:end,1) = in(:,:,2:2:end,1);
            out(:,:,2:2:end,1) = in(:,:,1:2:end,1);
        else
            out(1:2:end,1) = in(2:2:end,1);
            out(2:2:end,1) = in(1:2:end,1);
        end
    elseif strcmp(flag, 'swapN_half') == 1
        if ndims(in) == 4
            n = size(in,4);
            half_n = uint32(n/2);
            out(:,:,1:half_n,1) = in(:,:,half_n+1:end,1);
            out(:,:,half_n+1:end,1) = in(:,:,1:half_n,1);
        else
            n = size(in,1);
            half_n = uint32(n/2);
            out(1:half_n,1) = in(half_n+1:end,1);
            out(half_n+1:end,1) = in(1:half_n,1);
        end
    end
end
   
    
