clear;
% change directory here to test on different datasets 
directory = 'E:/Umich/Generator/polygon2/';
files = dir(directory);
files = files(3:end);
N = numel(files);
shape_spec = [];
thickness = 500;
acc = 5;
for i = 3001:1:3500
    if files(i).name(1) ~= '.'
        tic
        img_path = strcat(directory,files(i).name);
        temp = strsplit(files(i).name,'_');
        name = str2double(temp{1});
        temp = strsplit(temp{2},'.');
        gap = str2double(temp{1});
        [TE,TM] = cal_spec(gap,thickness,acc,img_path);
        shape_spec(i,:) = [name,gap,TE,TM];
        disp(i);
        toc
    end
end

save 'shape_spec_500.mat' shape_spec