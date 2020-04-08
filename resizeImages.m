function [ images ] = resizeImages( directory )
data = cd(directory);
imagefiles = dir('*.tiff');
nfiles = length(imagefiles);   
for ii=1:nfiles
   currentfilename = imagefiles(ii).name;
   currentimage = imread(currentfilename);
   currentimage = imresize(currentimage, [48 48]);
   images{ii} = currentimage;
   imwrite(images{ii},sprintf('surprise%d.jpg' ,ii));
end

% row=64; col=64; counti=0; countj=0;
% Blocks2 = cell(48,48);
% for ii = 1: nfiles
%     i2=images{ii};
%  for i=1:row-47
%    
% %     for j=1:col-47
% %         if (i == j)
%         counti=counti+1;
%         Blocks2{counti} = i2(i:i+47,i:i+47);
%         %end
%        % imwrite(Blocks2{counti},sprintf('surprise%d.jpg' ,counti));
%         
% %     end
% end
% end
 end
function I = readFunctionTrain(filename)
% Resize the images to the size required by the network.
I = imread(filename);

I = imresize(I, [227 227]);

end