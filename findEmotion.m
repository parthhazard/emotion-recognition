function findEmotion(net)
%FINDEMOTION Summary of this function goes here
[fig, ax1, ax2] = figureSetup(net);

wcam = webcam();
while ishandle(fig)
    img = snapshot(wcam);
    img = rgb2gray(img);
    img = imresize(img, [48 48]);
    [imagepred, probabilities] = predict(net, img);
    
     try
         imshow(insertText(img, [640, 1], upper(cellstr(imagepred)), 'AnchorPoint', 'RightTop', 'FontSize', 50, 'BoxColor', 'Green', 'BoxOpacity', 0.4), 'Parent', ax1);
         ax2.Children.YData = probabilities;
         ax2.YLine = [0, 1];
     catch err
     end
    drawnow
end
end

function [fig, ax1, ax2] = figureSetup(net)
set(0, 'defaultfigurewindowstyle', 'docked')
fig = figure('Name', 'Eotion Recognition', 'NumberTitle', 'off');
ax1 = subplot(2, 1, 1);
ax2 = subplot(2, 1, 2);
bar(ax2, zeros(1, numel(net.Layers(18, 1).ClassNames)), 'FaceColor', [0.2 0.6 0.8]);
set(ax2, 'XTickLabel', cellstr(net.Laters(18, 1).ClassNames));
set(0, 'defaultfigurewindowstyle', 'normal')
        

end