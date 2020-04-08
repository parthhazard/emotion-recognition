clear all
clc
%load training data


disp('Loading Data.....');
%resizeImages(dir);
testImages = imageDatastore('testImages');
allImages = imageDatastore('EMODATB', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
%need to setr for test and trainging data
[trainImages, validationImages] = splitEachLabel(allImages, 0.75, 'randomize');
% trainAug = imageDataAugmenter;
% trainFinal =  augmentedImageSource([48 48], allImages, 'DataAugmentation', trainAug);

%testImages = imageDatastore('EMODATB', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
disp('Finished Loading Data.....');
disp('All Data will be classified into following categories');
disp(unique(allImages.Labels));

for i=2:5
    img = readimage(testImages, i);
    img = rgb2gray(img);
end
