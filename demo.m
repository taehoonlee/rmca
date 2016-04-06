clear all;
close all;

% Prerequisite: CVX @ Stanford
if exist('cvx', 'file') ~= 2
    filename = 'cvx-w64.zip';
    % Download CVX
    if exist(filename, 'file') ~= 2
        fprintf('downloading cvx from http://cvxr.com/cvx/download/\n');
        urlwrite('http://web.cvxr.com/cvx/cvx-w64.zip', filename);
    end
    % Unzip CVX
    fprintf('unzipping %s\n', filename);
    unzip(filename);
    addpath('./cvx');
    addpath('./cvx/builtins');
    addpath('./cvx/commands');
    addpath('./cvx/functions');
    addpath('./cvx/functions/vec_');
    addpath('./cvx/lib');
    addpath('./cvx/structures');
    cvx_setup
end

% Test data: MNIST @ NYU
if exist('t10k-images-idx3-ubyte', 'file') ~= 2
    filename1 = 't10k-images-idx3-ubyte.gz';
    filename2 = 't10k-labels-idx1-ubyte.gz';
    % Download MNIST
    if exist(filename1, 'file') ~= 2
        fprintf('downloading MNIST images from http://yann.lecun.com/exdb/mnist/\n');
        urlwrite('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', filename1);
    end
    if exist(filename2, 'file') ~= 2
        fprintf('downloading MNIST labels from http://yann.lecun.com/exdb/mnist/\n');
        urlwrite('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', filename2);
    end
    % Unzip MNIST
    fprintf('unzipping %s, %s\n', filename1, filename2);
    gunzip(filename1);
    gunzip(filename2);
end

% Load MNIST
filename1 = 't10k-images-idx3-ubyte';
filename2 = 't10k-labels-idx1-ubyte';
f = fopen(filename1, 'rb');
magic = fread(f, 1, 'int32', 0, 'ieee-be');
numImages = fread(f, 1, 'int32', 0, 'ieee-be');
numRows = fread(f, 1, 'int32', 0, 'ieee-be');
numCols = fread(f, 1, 'int32', 0, 'ieee-be');
images = fread(f, inf, 'uint8');
images = reshape(images, numCols, numRows, numImages);
images = permute(images, [2 1 3]) / 255;
fclose(f);
f = fopen(filename2, 'rb');
magic = fread(f, 1, 'int32', 0, 'ieee-be');
numLabels = fread(f, 1, 'int32', 0, 'ieee-be');
labels = fread(f, inf, 'uint8');
numClasses = max(labels) + 1;
fclose(f);

% Run RMCA
im2vec = @(x) reshape(x, numCols*numRows, []);
vec2im = @(x) reshape(x, numCols, numRows, []);
rmaximin = zeros(numCols*numRows, numClasses);
centroid = zeros(numCols*numRows, numClasses);
for c = 1:numClasses
    rmaximin(:,c) = rmca(im2vec(images(:,:,labels==(c-1))), 1.8);
    centroid(:,c) = mean(im2vec(images(:,:,labels==(c-1))), 2);
end
rmaximin = vec2im(rmaximin);
centroid = vec2im(centroid);

% Show Results
figure;
for c = 1:numClasses
    subplot(2, numClasses, c); imagesc(rmaximin(:,:,c)); axis off;
    subplot(2, numClasses, c+numClasses); imagesc(centroid(:,:,c)); axis off;
end