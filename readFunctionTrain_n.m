function I = readFunctionTrain_n(filename)

% Read the image from file, if it has one colour dimension (gray scale image)
% convert it to the gray scale representation in the coloured space, 
%then resize it to the input dimensions required by AlexNet
I = imread(filename);

[~ , ~, d] = size(I);
if d == 1   
    [X, cmap] = gray2ind(I);
    I = ind2rgb(X, cmap);
end

I = imresize(I, [227 227]);
