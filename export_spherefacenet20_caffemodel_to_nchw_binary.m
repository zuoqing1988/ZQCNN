
function export_spherefacenet20_caffemodel_to_nchw_binary()
caffe.set_mode_cpu();

caffe_net  = caffe.Net('sphereface_deploy.prototxt', 'sphereface_model_iter_28000.caffemodel', 'test');

layers = {
    'conv1_1',2,'none';
    'relu1_1',1,'none';
    'conv1_2',2,'none';
    'relu1_2',1,'none';
    'conv1_3',2,'none';
    'relu1_3',1,'none';
    'res1_3',0,'none';
    'conv2_1',2,'none';
    'relu2_1',1,'none';
    'conv2_2',2,'none';
    'relu2_2',1,'none';
    'conv2_3',2,'none';
    'relu2_3',1,'none';
    'res2_3',0,'none';
    'conv2_4',2,'none';
    'relu2_4',1,'none';
    'conv2_5',2,'none';
    'relu2_5',1,'none';
    'res2_5',0,'none';
    'conv3_1',2,'none';
    'relu3_1',1,'none';
    'conv3_2',2,'none';
    'relu3_2',1,'none';
    'conv3_3',2,'none';
    'relu3_3',1,'none';
    'res3_3',0,'none';
    'conv3_4',2,'none';
    'relu3_4',1,'none';
    'conv3_5',2,'none';
    'relu3_5',1,'none';
    'res3_5',0,'none';
    'conv3_6',2,'none';
    'relu3_6',1,'none';
    'conv3_7',2,'none';
    'relu3_7',1,'none';
    'res3_7',0,'none';
    'conv3_8',2,'none';
    'relu3_8',1,'none';
    'conv3_9',2,'none';
    'relu3_9',1,'none';
    'res3_9',0,'none';
    'conv4_1',2,'none';
    'relu4_1',1,'none';
    'conv4_2',2,'none';
    'relu4_2',1,'none';
    'conv4_3',2,'none';
    'relu4_3',1,'none';
    'fc5',2,'fc7x6';
   };
export_to_binary(layers,caffe_net,'sphereface20.nchwbin');
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
    elseif strcmp(flag,'fc7x6') == 1
        if ndims(in) == 2
            [m,n]=size(in);
            out = reshape(in, [7 6 uint32(m/42) n]);
        end
    end
    
    out = permute(out,[1 2 3 4]);%not permuted
    
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
   
    
