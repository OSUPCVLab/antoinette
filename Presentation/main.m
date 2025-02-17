% Nima A. Gard
% Visualizing semantic segmentation for CityScape Dataset

close all
baseRGB = 'C:\Users\ajamgard.1\Desktop\TemporalPose\Data\SYNTHIA-SEQS-01-SUMMER\RGB\Stereo_Right\Omni_B\';
baseGT = 'C:\Users\ajamgard.1\Desktop\TemporalPose\Data\SYNTHIA-SEQS-01-SUMMER\GT\Color\Stereo_Right\Omni_B\';
img_files = dir(strcat(baseRGB,'*.png')); 
ann_files = dir(strcat(baseGT,'*.png'));  

nfiles = length(img_files);    % Number of files found
img = imread(strcat(baseRGB, img_files(1).name));
[w, h] = size(img);


fig = figure('position',[0 0 h w]);
clear('f');
ax1 = axes('Parent',fig);
ax2 = axes('Parent',fig);
set(ax1, 'Visible','off');
set(ax2, 'Visible','off');

v = VideoWriter('output.mp4');
v.FrameRate = 30;
open(v)


for i=1:nfiles
   img =imread(strcat(baseRGB, img_files(i).name));
   ann = imread(strcat(baseGT, ann_files(i).name));

   
   alpha = (i+10) / nfiles;

  
   K = imlincomb(1 - alpha,img,alpha,ann);
   imshow(K);
   
   
   f(i) = getframe(fig);
   writeVideo(v,f(i));
   
end

close all
[h, w, p] = size(f(1).cdata);  % use 1st frame to get dimensions
hf = figure; 
% resize figure based on frame's w x h, and place at (150, 150)
set(hf, 'position', [150 150 w h]);
axis off
movie(hf,f);
close(v);