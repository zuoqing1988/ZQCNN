function export_spherefacenet04bn256_caffemodel_to_nchw_binary()
caffe.set_mode_cpu();

%caffe_net  = caffe.Net('code\\dim256\\sphereface04_deploy.prototxt', 'result\\dim256_sphereface04_model_iter_80000.caffemodel', 'test');
%out_file = 'sphereface04bn256.nchwbin';
caffe_net  = caffe.Net('sphereface04bn256_deploy.prototxt', 'sphereface04bn256_model_iter_28000.caffemodel', 'test');
out_file = 'sphereface04bn256_iter_28000.nchwbin';


layers = {
    'conv1_1',2,'none';
    'batchnorm1_1',3,'div3';
    'scale1_1',2,'none';
    'relu1_1',1,'none';
    'conv2_1',2,'none';
    'batchnorm2_1',3,'div3';
    'scale2_1',2,'none';
    'relu2_1',1,'none';
    'conv3_1',2,'none';
    'batchnorm3_1',3,'div3';
    'scale3_1',2,'none';
    'relu3_1',1,'none';
    'conv4_1',2,'none';
    'batchnorm4_1',3,'div3';
    'scale4_1',2,'none';
    'relu4_1',1,'none';
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
   
    
