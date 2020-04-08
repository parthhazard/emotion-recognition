clc
FDetect = vision.CascadeObjectDetector();
FDetect.MergeThreshold=120;
camera = webcam();

h = figure;
h.Position(3) = 2*h.Position(3);
ax1 = subplot(1,2,1);
ax2 = subplot(1,2,2);

while(1)
I=snapshot(camera);
BB = step(FDetect,I);
size(BB,1);  
image(I); hold on
for i = 1:size(BB,1)
    rectangle('Position',BB(i,:),'LineWidth',2,'LineStyle','-','EdgeColor','r');
end
set(gcf,'CloseRequestFcn','keepRolling = false; closereq');
% Display and classify the image
 
    image(ax1,I)
    I = imresize(I, 48);
    [label,score] = classify(net,im);
    title(ax1,{char(label),num2str(max(score),2)});

    [~,idx] = sort(score,'descend');
    idx = idx(7:-1:1);
    scoreTop = score(idx);
    classNamesTop = classNames(idx);

    barh(ax2,scoreTop)
    title(ax2,'Emotions')
    xlabel(ax2,'Probability')
    xlim(ax2,[0 1])
    yticklabels(ax2,classNamesTop)
    ax2.YAxisLocation = 'right';

    drawnow
end
