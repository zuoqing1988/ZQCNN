function export_mobilenet_spherefacenet10bn512_caffemodel_to_nchw_binary()
caffe.set_mode_cpu();

caffe_net  = caffe.Net('mobilenet_sphereface10bn512_deploy.prototxt', 'mobilenet_sphereface10bn512_model_iter_50000.caffemodel', 'test');
out_file = 'mobilenet_sphereface10bn512_iter_50000.nchwbin';


layers = {
    'conv1_1',2,'none';
    'batchnorm1_1',3,'div3';
    'scale1_1',2,'none';
    'relu1_1',1,'none';
    'conv2_1/dw',2,'none';
    'conv2_1/dw/bn',3,'div3';
    'conv2_1/dw/scale',2,'none';
    'relu2_1/dw',1,'none';
    'conv2_1/sep',2,'none';
    'conv2_1/sep/bn',3,'div3';
    'conv2_1/sep/scale',2,'none';
    'relu2_1/sep',1,'none';
    'conv2_2/dw',2,'none';
    'conv2_2/dw/bn',3,'div3';
    'conv2_2/dw/scale',2,'none';
    'relu2_2/dw',1,'none';
    'conv2_2/sep',2,'none';
    'conv2_2/sep/bn',3,'div3';
    'conv2_2/sep/scale',2,'none';
    'relu2_2/sep',1,'none';
    'conv2_3/dw',2,'none';
    'conv2_3/dw/bn',3,'div3';
    'conv2_3/dw/scale',2,'none';
    'relu2_3/dw',1,'none';
    'conv2_3/sep',2,'none';
    'conv2_3/sep/bn',3,'div3';
    'conv2_3/sep/scale',2,'none';
    'relu2_3/sep',1,'none';
    'res2_3',0,'none';
    'conv3_1/dw',2,'none';
    'conv3_1/dw/bn',3,'div3';
    'conv3_1/dw/scale',2,'none';
    'relu3_1/dw',1,'none';
    'conv3_1/sep',2,'none';
    'conv3_1/sep/bn',3,'div3';
    'conv3_1/sep/scale',2,'none';
    'relu3_1/sep',1,'none';
    'conv3_2/dw',2,'none';
    'conv3_2/dw/bn',3,'div3';
    'conv3_2/dw/scale',2,'none';
    'relu3_2/dw',1,'none';
    'conv3_2/sep',2,'none';
    'conv3_2/sep/bn',3,'div3';
    'conv3_2/sep/scale',2,'none';
    'relu3_2/sep',1,'none';
    'conv3_3/dw',2,'none';
    'conv3_3/dw/bn',3,'div3';
    'conv3_3/dw/scale',2,'none';
    'relu3_3/dw',1,'none';
    'conv3_3/sep',2,'none';
    'conv3_3/sep/bn',3,'div3';
    'conv3_3/sep/scale',2,'none';
    'relu3_3/sep',1,'none';
    'res3_3',0,'none';
    'conv3_4/dw',2,'none';
    'conv3_4/dw/bn',3,'div3';
    'conv3_4/dw/scale',2,'none';
    'relu3_4/dw',1,'none';
    'conv3_4/sep',2,'none';
    'conv3_4/sep/bn',3,'div3';
    'conv3_4/sep/scale',2,'none';
    'relu3_4/sep',1,'none';
    'conv3_5/dw',2,'none';
    'conv3_5/dw/bn',3,'div3';
    'conv3_5/dw/scale',2,'none';
    'relu3_5/dw',1,'none';
    'conv3_5/sep',2,'none';
    'conv3_5/sep/bn',3,'div3';
    'conv3_5/sep/scale',2,'none';
    'relu3_5/sep',1,'none';
    'res3_5',0,'none';
    'conv4_1/dw',2,'none';
    'conv4_1/dw/bn',3,'div3';
    'conv4_1/dw/scale',2,'none';
    'relu4_1/dw',1,'none';
    'conv4_1/sep',2,'none';
    'conv4_1/sep/bn',3,'div3';
    'conv4_1/sep/scale',2,'none';
    'relu4_1/sep',1,'none';
    'fc5',2,'fc7x6';
    'batchnorm5',3,'div3';
    'scale5',2,'none';
   };
export_to_binary(layers,caffe_net,out_file);
end

function [] = export_to_binary(layers, caffe_net, out_file)
n = size(layers,1);
all = [];
for i = 1:n
    layer_name = layers{i,1};
    disp(layer_name);
    flag = layers{i,3};
    if strcmp(flag,'div3') == 1
        data1 = caffe_net.params(layer_name,1).get_data();
        data2 = caffe_net.params(layer_name,2).get_data();
        data3 = caffe_net.params(layer_name,3).get_data();
        if data3(1,1) ~= 0
            data1 = data1/data3(1,1);
            data2 = data2/data3(1,1);
            all =[all;data1(:);data2(:)];
        else
            data1 = zeros(size(data1));
            data2 = ones(size(data2));
            all =[all;data1(:);data2(:)];
        end
    else
        for j = 1:uint32((layers{i,2}))
            tmp = caffe_net.params(layer_name,j).get_data();
            if j == 1
                tmp = auto_permute(tmp,flag);
            end
            all =[all;tmp(:)];
        end
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
   
    
