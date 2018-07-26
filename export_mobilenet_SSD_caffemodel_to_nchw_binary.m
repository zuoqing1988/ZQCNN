function export_mobilenet_SSD_caffemodel_to_nchw_binary()
caffe.set_mode_cpu();

caffe_net  = caffe.Net('MobileNetSSD_deploy.prototxt', 'MobileNetSSD_deploy.caffemodel', 'test');
out_file = 'MobileNetSSD_deploy.nchwbin';


layers = {
    'input',0,'none';
    %'data_input_0_split',0,'none';
    'conv0',2,'none';
    'conv0/relu',0,'none';
    'conv1/dw',2,'none';
    'conv1/dw/relu',0,'none';
    'conv1',2,'none';
    'conv1/relu',0,'none';
    'conv2/dw',2,'none';
    'conv2/dw/relu',0,'none';
    'conv2',2,'none';
    'conv2/relu',0,'none';
    'conv3/dw',2,'none';
    'conv3/dw/relu',0,'none';
    'conv3',2,'none';
    'conv3/relu',0,'none';
    'conv4/dw',2,'none';
    'conv4/dw/relu',0,'none';
    'conv4',2,'none';
    'conv4/relu',0,'none';
    'conv5/dw',2,'none';
    'conv5/dw/relu',0,'none';
    'conv5',2,'none';
    'conv5/relu',0,'none';
    'conv6/dw',2,'none';
    'conv6/dw/relu',0,'none';
    'conv6',2,'none';
    'conv6/relu',0,'none';
    'conv7/dw',2,'none';
    'conv7/dw/relu',0,'none';
    'conv7',2,'none';
    'conv7/relu',0,'none';
    'conv8/dw',2,'none';
    'conv8/dw/relu',0,'none';
    'conv8',2,'none';
    'conv8/relu',0,'none';
    'conv9/dw',2,'none';
    'conv9/dw/relu',0,'none';
    'conv9',2,'none';
    'conv9/relu',0,'none';
    'conv10/dw',2,'none';
    'conv10/dw/relu',0,'none';
    'conv10',2,'none';
    'conv10/relu',0,'none';
    'conv11/dw',2,'none';
    'conv11/dw/relu',0,'none';
    'conv11',2,'none';
    'conv11/relu',0,'none';
    %'conv11_conv11/relu_0_split',0,'none';
    'conv12/dw',2,'none';
    'conv12/dw/relu',0,'none';
    'conv12',2,'none';
    'conv12/relu',0,'none';
    'conv13/dw',2,'none';
    'conv13/dw/relu',0,'none';
    'conv13',2,'none';
    'conv13/relu',0,'none';
    %'conv13_conv13/relu_0_split',0,'none';
    'conv14_1',2,'none';
    'conv14_1/relu',0,'none';
    'conv14_2',2,'none';
    'conv14_2/relu',0,'none';
    %'conv14_2_conv14_2/relu_0_split',0,'none';
    'conv15_1',2,'none';
    'conv15_1/relu',0,'none';
    'conv15_2',2,'none';
    'conv15_2/relu',0,'none';
    %'conv15_2_conv15_2/relu_0_split',0,'none';
    'conv16_1',2,'none';
    'conv16_1/relu',0,'none';
    'conv16_2',2,'none';
    'conv16_2/relu',0,'none';
    %'conv16_2_conv16_2/relu_0_split',0,'none';
    'conv17_1',2,'none';
    'conv17_1/relu',0,'none';
    'conv17_2',2,'none';
    'conv17_2/relu',0,'none';
    %'conv17_2_conv17_2/relu_0_split',0,'none';
    'conv11_mbox_loc',2,'none';
    'conv11_mbox_loc_perm',0,'none';
    'conv11_mbox_loc_flat',0,'none';
    'conv11_mbox_conf',2,'none';
    'conv11_mbox_conf_perm',0,'none';
    'conv11_mbox_conf_flat',0,'none';
    'conv11_mbox_priorbox',0,'none';
    'conv13_mbox_loc',2,'none';
    'conv13_mbox_loc_perm',0,'none';
    'conv13_mbox_loc_flat',0,'none';
    'conv13_mbox_conf',2,'none';
    'conv13_mbox_conf_perm',0,'none';
    'conv13_mbox_conf_flat',0,'none';
    'conv13_mbox_priorbox',0,'none';
    'conv14_2_mbox_loc',2,'none';
    'conv14_2_mbox_loc_perm',0,'none';
    'conv14_2_mbox_loc_flat',0,'none';
    'conv14_2_mbox_conf',2,'none';
    'conv14_2_mbox_conf_perm',0,'none';
    'conv14_2_mbox_conf_flat',0,'none';
    'conv14_2_mbox_priorbox',0,'none';
    'conv15_2_mbox_loc',2,'none';
    'conv15_2_mbox_loc_perm',0,'none';
    'conv15_2_mbox_loc_flat',0,'none';
    'conv15_2_mbox_conf',2,'none';
    'conv15_2_mbox_conf_perm',0,'none';
    'conv15_2_mbox_conf_flat',0,'none';
    'conv15_2_mbox_priorbox',0,'none';
    'conv16_2_mbox_loc',2,'none';
    'conv16_2_mbox_loc_perm',0,'none';
    'conv16_2_mbox_loc_flat',0,'none';
    'conv16_2_mbox_conf',2,'none';
    'conv16_2_mbox_conf_perm',0,'none';
    'conv16_2_mbox_conf_flat',0,'none';
    'conv16_2_mbox_priorbox',0,'none';
    'conv17_2_mbox_loc',2,'none';
    'conv17_2_mbox_loc_perm',0,'none';
    'conv17_2_mbox_loc_flat',0,'none';
    'conv17_2_mbox_conf',2,'none';
    'conv17_2_mbox_conf_perm',0,'none';
    'conv17_2_mbox_conf_flat',0,'none';
    'conv17_2_mbox_priorbox',0,'none';
    'mbox_loc',0,'none';
    'mbox_conf',0,'none';
    'mbox_priorbox',0,'none';
    'mbox_conf_reshape',0,'none';
    'mbox_conf_softmax',0,'none';
    'mbox_conf_flatten',0,'none';
    'detection_out',0,'none';
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
   
    
