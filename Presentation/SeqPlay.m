% Nima A. Gard
% Visualizing semantic segmentation for CityScape Dataset

close all
base = dir('C:\Users\ajamgard.1\Desktop\TemporalPose\results\');
v = VideoWriter(strcat('results2\output_',num2str(400,'%06d')));
v.FrameRate = 5;
open(v)

path = strcat(base(3).folder,'\',base(3).name,'\');
img_files = dir(strcat(path,'*_gt_img.png')); 
img = imread(strcat(path, img_files(1).name));
[w, h] = size(img);


fig = figure('position',[0 0 h*2 w*2]);

clear('f');
ax1 = axes('Parent',fig);
ax2 = axes('Parent',fig);
set(ax1, 'Visible','off');
set(ax2, 'Visible','off');
j = 1;
for k = 3 : length(base)
    path = strcat(base(k).folder,'\',base(k).name,'\');
    img_files = dir(strcat(path,'*_gt_img.png')); 
    ann_files = dir(strcat(path,'*_sub2.png'));  

    nfiles = length(img_files);    % Number of files found
    for i=1:nfiles
        img =imread(strcat(path, img_files(i).name));
        ann = imread(strcat(path, ann_files(i).name));

        subplot(1,2,1)
        imshow(img);
        subplot(1,2,2)
        imshow(ann);

        f(j) = getframe(fig);
        writeVideo(v,f(j));
        j = j + 1;
    end
  
end
close all
[h, w, p] = size(f(1).cdata);  % use 1st frame to get dimensions
hf = figure; 
% resize figure based on frame's w x h, and place at (150, 150)
set(hf, 'position', [150 150 w h]);
axis off
movie(hf,f);
close(v);
