sizeImage = 48;

conv1 = convolution2dLayer([5 5], 10, 'Padding', 2);
% conv1.Bias = gpuArray(single(zeros([1 1 10])));
conv1.Bias = zeros([1 1 10]);
conv2 = convolution2dLayer([5 5], 10, 'Padding', 2);
% conv2.Bias = gpuArray(single(zeros([1 1 10])));
conv2.Bias = zeros([1 1 10]);

conv3 = convolution2dLayer([4 4], 10, 'Padding', 2);
% conv3.Bias = gpuArray(single(zeros([1 1 10])));
conv3.Bias = zeros([1 1 10]);

fc1 = fullyConnectedLayer(128);
% fc2.Bias = gpuArray(single(zeros([128 1])));
fc1.Bias = zeros([128 1]);

fc2 = fullyConnectedLayer(7);
% fc3.Bias = gpuArray(single(zeros([7 1])));
fc2.Bias = zeros([7 1]);

layersCnn = [ 
    imageInputLayer([48 48 1]);
    conv1;
    reluLayer();
    crossChannelNormalizationLayer(4)
    maxPooling2dLayer([3 3]);
    conv2;
    reluLayer();
    maxPooling2dLayer([3 3]);
    conv3;
    reluLayer();
    fc1;
    reluLayer();
    dropoutLayer(0.5);
    fc2;
    softmaxLayer()
    classificationLayer()];

options = trainingOptions('sgdm', 'InitialLearnRate', 0.01, 'MaxEpochs', 50, 'Verbose', true, 'MiniBatchSize', 50, 'Plots', 'training-progress', 'LearnRateSchedule', 'piecewise', 'LearnRateDropFactor', 0.5, 'ValidationData', validationImages, 'Shuffle', 'every-epoch');
[net, info] = trainNetwork(trainImages, layersCnn, options);
 
%% Face Detection
faceDetector = vision.CascadeObjectDetector;
I1 = imread('testImages\scarlet-johansson-cover.jpg');
I2 = imread('testImages\angry.jpg');
I3 = imread('testImages\smile.jpg');
I4 = imread('testImages\brando.jpg');


bbox = faceDetector(I);
IFaces = insertObjectAnnotation(I,'rectangle',bbox,'Face');
I = rgb2gray(IFaces);
figure
imshow(I)
title('Detected faces');

%% 
I = imread('testImages\surprise1');
I2 = imread('testImages\surprise2.jpg');
I3 = imread('testImages\surprise3.jpg');
I4 = imread('testImages\surprise4.jpg');
I5 = imread('testImages\surprise5.jpg');


I1 = rgb2gray(I1) ;
I4 = rgb2gray(I4) ;
I3 = rgb2gray(I3) ;
I2 = rgb2gray(I2) ;
I5 = rgb2gray(I5);

%%
im = imread('EMODATB\ANGRY\angry22.jpg');
%im = imread('EMODATB\DISGUST\disgust12.jpg');
%im = imread('EMODATB\SAD\sad1.jpg');
%im = imread('EMODATB\FEAR\fear1.jpg');
%im = imread('EMODATB\NEUTRAL\neutral11.jpg');
%im = rgb2gray(im);
%im = imresize(im, [48 48]);
imshow(im)
[label,score] = classify(net,im);
title({char(label),num2str(max(score),2)});


