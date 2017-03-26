%MATLAB Code for Master's Thesis 
%code to output image in original format

%Using Chicago Face Data
% reads in image in format 750 x 750

%reads image 1 (young male 1)
A= imread('C:\Users\Luis\Documents\Graduate Statistics UCLA\Master Thesis Work\Chicago - Greyscale\WM200.bmp');
%imshow(A);
A = reshape(A, 562500,1);


basename = 'C:\Users\Luis\Documents\Graduate Statistics UCLA\Master Thesis Work\Chicago - Greyscale\WM';
basename2 = '.bmp';
for i=201:225   % for male images 201 to 225
filename=[basename,num2str(i),basename2];
[B] = imread(filename);
B = reshape(B, 562500, 1);
A=[A,B];
end

for i=227:245   % for male images 227 to 245
filename=[basename,num2str(i),basename2];
[B] = imread(filename);
B = reshape(B, 562500, 1);
A=[A,B];
end

for i=247:258   % for male images 247 to 258
filename=[basename,num2str(i),basename2];
[B] = imread(filename);
B = reshape(B, 562500, 1);
A=[A,B];
end

for i=31:41   % for male images 31 to 41
filename=[basename,num2str(i),basename2];
[B] = imread(filename);
B = reshape(B, 562500, 1);
A=[A,B];
end

for i=28:29   % for male images 28 to 29
filename=[basename,num2str(i),basename2];
[B] = imread(filename);
B = reshape(B, 562500, 1);
A=[A,B];
end

for i=18:26   % for male images 9 to 26
filename=[basename,num2str(i),basename2];
[B] = imread(filename);
B = reshape(B, 562500, 1);
A=[A,B];
end

%%%TEST 
testimages= imread('C:\Users\Luis\Documents\Graduate Statistics UCLA\Master Thesis Work\Chicago - Greyscale\WM9.bmp');
testimages = reshape(testimages, 562500,1);
for i=10:17   % for testing male images 9 to 17
filename=[basename,num2str(i),basename2];
[B] = imread(filename);
B = reshape(B, 562500, 1);
testimages=[testimages,B];
end

for i=6:6   % for male image WM6
filename=[basename,num2str(i),basename2];
[B] = imread(filename);
B = reshape(B, 562500, 1);
A=[A,B];
end

for i=1:4   % for male images 1 to 4
filename=[basename,num2str(i),basename2];
[B] = imread(filename);
B = reshape(B, 562500, 1);
A=[A,B];
end



% A is a 562500 x 84 matrix of uint8 pixels
doubleA = double(A);			% converts uint8 pixels to doubles
sumA = sum(doubleA, 2);
meanface = (1/84).*sumA; 	


%shows meanface
%This is the difference between the images and the mean face
submeanface = doubleA(:,1) - meanface;
for t = 2:84
calc = doubleA(:,t) - meanface;
submeanface = [submeanface,calc];	
end

ATA = (transpose(submeanface)*submeanface);  	%deviations squared. Calculates scatter matrix.
[V,D]= eig(ATA);			%computes eigenvalue and eigenvectors of scatter matrix
eigenface= submeanface*V;	

eigenvect = eigenface;

%Adds mean face to eigenvectors (creates “eigenface”)
normeigenface=eigenvect(:,1)+meanface;
for i = 2:84					
calc = eigenvect(:,i)+meanface;
normeigenface = [normeigenface, calc];
end


% show eigenface
%imshow(uint8(reshape(normeigenface(:,84), 750,750)))

% Now that we have eigenfaces for training image, read in testing images.

%for images 90-96
testimages= imread('C:\Users\Luis\Documents\Graduate Statistics UCLA\Master Thesis Work\Chicago - Greyscale\WM31.bmp');
testimages = reshape(testimages, 562500,1);
basename = 'C:\Users\Luis\Documents\Graduate Statistics UCLA\Master Thesis Work\Chicago - Greyscale\WM';
basename2 = '.bmp';
for i=32:41   % for male images 31 to 41
filename=[basename,num2str(i),basename2];
B = imread(filename);
B = reshape(B, 562500, 1);
testimages=[testimages,B];
end



doubtestimages = double(testimages);
testsubmean = doubtestimages(:,1)-meanface;  % Subtracting mean face from test images
calc = [];
for t = 2:9
calc = doubtestimages(:,t) - meanface;
testsubmean = [testsubmean,calc];     %This is the difference between the test images and the meanface
end


%These are just the eigenvectors without the mean face added
keigenface = eigenvect(:,65);  		%This saves the best k=20 Eigenfaces.
calc=[];
for t = 66:84
calc= eigenvect(:,t);
keigenface = [keigenface,calc];
end

%normalizes the eigenvectors
%keigenface is 562500 x 20
% This normalizes the “eigenfaces”
for i=1:20
keigenface(:,i)=keigenface(:,i)/(norm(keigenface(:,i)));	
end

%project is 20 x 10
%projection of testing images into eigenspace
project = transpose(keigenface)*testsubmean;	

% This initializez the reconst vector of 110592 x 1.
reconst = project(1,1)*keigenface(:,1);		
reconst = reconst - reconst;

%This creates matrix of reconstructed test images.
for j = 1:9					
sumR = 0;
for i = 1:20
calc = project(i,j)*keigenface(:,i)	;	% Calculates the reconstructed image
sumR = sumR + calc;		% Sums up entire eigenfaces and weights, obtains reconstructed image
end
reconst = [reconst, sumR];
end;
reconst(:,1)=[];			%This omits the first column, which was used to initialize matrix.

% adds the mean image value to the reconstructed images (using the eigenfaces)
for i=1:9
reconst(:,i) = reconst(:,i)+(meanface);
end

%This will show reconstructed test image.
imshow(uint8(reshape(reconst(:,1),750,750)))

%Need to work on reconstruction plot

%project is 84 x 9
%projection of testing images into eigenspace
% This initializez the reconst vector of 110592 x 1.
reconsteig = normc(eigenface);
projectplot = transpose(reconsteig)*testsubmean;	
reconst1 = projectplot(1,1)*reconsteig(:,1);		
reconst1 = reconst1 - reconst1;
testplot=0;
testplot(1:9,1:84)=0;
tempreconst=0;
tempreconst(1:84)=0;
reconstplot=0;
reconstplot(1:84)=0;
for j=1:9
    for k = 1:84
        sumR=0;
        for i=0:k-1
            calc = projectplot(84-i,j)*reconsteig(:,84-i);	% Calculates the reconstructed image
            sumR = sumR + calc;	
        end
        tempreconst(k)= norm(testsubmean(:,j)-sumR);
    end
    testplot(j,:)=tempreconst;
end
reconstplot = mean(testplot);
plot((1:84),reconstplot,'red')
title('Reconstruction Error Plot for Linear PCA')
legend('Avg Error = 1.2851 x 10^4')

%Polynomial kernel degree 2 below
polykernel = 0;
polykernel(1:84,1:84)=0;

tuneresults=0;
for q = 1:1
    tunecalc = q;
    alpha=tunecalc;

calc = 0;
for i=1:84
    for j=1:84
       calc = (1000000000000000+1250*(dot(doubleA(:,i),doubleA(:,j))))^2;
       polykernel(i,j)=calc;
    end
end

%centering the kernel matrix
% 1N = scalarmatrix NxN matrix with all values equal to 1/N
sizeN=size(polykernel);
sizeN=sizeN(1);
scalarmatrix(1:sizeN,1:sizeN)=1/sizeN;
centermatrix = polykernel-scalarmatrix*polykernel-polykernel*scalarmatrix+scalarmatrix*polykernel*scalarmatrix;

%compute eigenvectors for polynomial kernel
[polyV,polyD] = eig(centermatrix);
polyeigenface = submeanface*polyV; %try this for now

%Poly reconstruction

%These are just the eigenvectors without the mean face added
keigenfacepoly = polyeigenface(:,65);  		%This saves the best k=20 Eigenfaces.
calc=[];
for t = 66:84
calc= polyeigenface(:,t);
keigenfacepoly = [keigenfacepoly,calc];
end

%normalizes the eigenvectors
%keigenface is 90000 x 20
for i=1:20
keigenfacepoly(:,i)=keigenfacepoly(:,i)/(norm(keigenfacepoly(:,i)));	% This normalizes the “eigenfaces”
end

%project is 20 x 10
projectpoly = transpose(keigenfacepoly)*testsubmean;	%projection of testing images into eigenspace


reconstpoly = projectpoly(1,1)*keigenfacepoly(:,1);		% This initializez the reconst vector of 110592 x 1.
reconstpoly = reconstpoly - reconstpoly;
for j = 1:9					%This creates matrix of reconstructed test images.
sumR = 0;
for i = 1:20
calc = projectpoly(i,j)*keigenfacepoly(:,i)	;	% Calculates the reconstructed image
sumR = sumR + calc;		% Sums up entire eigenfaces and weights, obtains reconstructed image
end
reconstpoly = [reconstpoly, sumR];
end;
reconstpoly(:,1)=[];			%This omits the first column, which was used to initialize matrix.

% adds the mean image value to the reconstructed images (using the eigenfaces)
for i=1:9
reconstpoly(:,i) = reconstpoly(:,i)+(meanface);
end

normtune(1:9)=0;
for i=1:9
    normtune(i) = norm(reconstpoly(:,i)-doubtestimages(:,i));
end
tuneresults = mean(normtune);
end

%This will show reconstructed test image.
imshow(uint8(reshape(reconstpoly(:,1),750,750)))

%project is 84 x 9
reconsteig = normc(polyeigenface);
projectplot = transpose(reconsteig)*testsubmean;	%projection of testing images into eigenspace
reconst1 = projectplot(1,1)*reconsteig(:,1);		% This initializez the reconst vector of 110592 x 1.
reconst1 = reconst1 - reconst1;
testplot=0;
testplot(1:9,1:84)=0;
tempreconst=0;
tempreconst(1:84)=0;
reconstplot1=0;
reconstplot1(1:84)=0;
for j=1:9
    for k = 1:84
        sumR=0;
        for i=0:k-1
            calc = projectplot(84-i,j)*reconsteig(:,84-i);	% Calculates the reconstructed image
            sumR = sumR + calc;	
        end
        tempreconst(k)= norm(testsubmean(:,j)-sumR);
    end
    testplot(j,:)=tempreconst;
end
reconstplot1 = mean(testplot);
plot((1:84),reconstplot1(1:84),'red') %omit last two for inconsistencies
title('Reconstruction Plot for Polynomial Kernel (d=2)')
legend('Avg Error = 1.2852 x 10^4')


%Exponential Kernel Radial Basis Kernel with l2 norm (not squared)
expkernel=0;
expkernel(1:84,1:84)=0;
calc = 0;

%computes variance of Euclidean Distance between all vector pairs.
normcalc=0;
normcalc(1:84,1:84)=0;
for i=1:84
    for j=1:84
        normcalc(i,j)=norm(doubleA(:,i)-doubleA(:,j));
    end
end
normcalc = reshape(normcalc,7056,1); %135 * 135
varexp = var(normcalc);

%tuning sigma
tuneresults=0;
for q = 5:5
    tunecalc = q*100000;
    varexp=tunecalc;

for i=1:84
    for j=1:84
       calc = exp((-1/(2*varexp))*(norm(doubleA(:,i)-doubleA(:,j))));
       expkernel(i,j)=calc;
    end
end

%centering the kernel matrix
% 1N = scalarmatrix NxN matrix with all values equal to 1/N
sizeN=size(expkernel);
sizeN=sizeN(1);
scalarmatrix(1:sizeN,1:sizeN)=1/sizeN;
centermatrix = expkernel-scalarmatrix*expkernel-expkernel*scalarmatrix+scalarmatrix*expkernel*scalarmatrix;

%compute eigenvectors for exponential kernel
[expV,expD] = eig(centermatrix);
expeigenface = submeanface*expV; %try this for now

%Exponential reconstruction

%These are just the eigenvectors without the mean face added
keigenfaceexp = expeigenface(:,65);  		%This saves the best k=20 Eigenfaces.
calc=[];
for t =66:84
calc= expeigenface(:,t);
keigenfaceexp = [keigenfaceexp,calc];
end

%normalizes the eigenvectors
%keigenface is 90000 x 20
for i=1:20
keigenfaceexp(:,i)=keigenfaceexp(:,i)/(norm(keigenfaceexp(:,i)));	% This normalizes the “eigenfaces”
end

%project is 20 x 10
projectexp = transpose(keigenfaceexp)*testsubmean;	%projection of testing images into eigenspace


reconstexp = projectexp(1,1)*keigenfaceexp(:,1);		% This initializez the reconst vector of 110592 x 1.
reconstexp = reconstexp - reconstexp;
for j = 1:9					%This creates matrix of reconstructed test images.
sumR = 0;
for i = 1:20
calc = projectexp(i,j)*keigenfaceexp(:,i)	;	% Calculates the reconstructed image
sumR = sumR + calc;		% Sums up entire eigenfaces and weights, obtains reconstructed image
end
reconstexp = [reconstexp, sumR];
end;
reconstexp(:,1)=[];			%This omits the first column, which was used to initialize matrix.

% adds the mean image value to the reconstructed images (using the eigenfaces)
for i=1:9
reconstexp(:,i) = reconstexp(:,i)+(meanface);
end

normtune(1:9)=0;
for i=1:9
    normtune(i) = norm(reconstexp(:,i)-doubtestimages(:,i));
end
tuneresults = mean(normtune);
end

%This will show reconstructed test image.
imshow(uint8(reshape(reconstexp(:,1),750,750)))

%project is 84 x 9
reconsteig = normc(expeigenface);
projectplot = transpose(reconsteig)*testsubmean;	%projection of testing images into eigenspace
reconst1 = projectplot(1,1)*reconsteig(:,1);		% This initializez the reconst vector of 110592 x 1.
reconst1 = reconst1 - reconst1;
testplot=0;
testplot(1:9,1:84)=0;
tempreconst=0;
tempreconst(1:84)=0;
reconstplot2=0;
reconstplot2(1:84)=0;
for j=1:9
    for k = 1:84
        sumR=0;
        for i=0:k-1
            calc = projectplot(84-i,j)*reconsteig(:,84-i);	% Calculates the reconstructed image
            sumR = sumR + calc;	
        end
        tempreconst(k)= norm(testsubmean(:,j)-sumR);
    end
    testplot(j,:)=tempreconst;
end
reconstplot2 = mean(testplot);
plot((1:83),reconstplot2(1:83)) %omits 84 because very off
title('Reconstruction Error Plot for Exponential Kernel PCA')

%Cauchy Kernel l2 norm (not squared)
caukernel=0;
caukernel(1:84,1:84)=0;
calc = 0;

%computes variance of Euclidean Distance between all vector pairs.
normcalc=0;
normcalc(1:84,1:84)=0;
for i=1:84
    for j=1:84
        normcalc(i,j)=norm(doubleA(:,i)-doubleA(:,j));
    end
end
normcalc = reshape(normcalc,7056,1); %135 * 135
varcau = var(normcalc);

%tuning sigma
tuneresults = 0;
%tuneresults(1:8)=0;
for q = 13:13
     tunecalc = q*1000000000;
     varcau=tunecalc;


for i=1:84
    for j=1:84
       calc = 1/(1+(dot(doubleA(:,i),doubleA(:,i))+dot(doubleA(:,j),doubleA(:,j))+2*dot(doubleA(:,i),doubleA(:,j)))/varcau);
       caukernel(i,j)=calc;
    end
end

%centering the kernel matrix
% 1N = scalarmatrix NxN matrix with all values equal to 1/N
sizeN=size(caukernel);
sizeN=sizeN(1);
scalarmatrix(1:sizeN,1:sizeN)=1/sizeN;
centermatrix = caukernel-scalarmatrix*caukernel-caukernel*scalarmatrix+scalarmatrix*caukernel*scalarmatrix;

%compute eigenvectors for exponential kernel
[cauV,cauD] = eig(centermatrix);
caueigenface = submeanface*cauV; %try this for now

%Cauchy reconstruction

%These are just the eigenvectors without the mean face added
keigenfacecau = caueigenface(:,84);  		%This saves the best k=20 Eigenfaces.
calc=[];
for t =1:19
calc= caueigenface(:,t);
keigenfacecau = [keigenfacecau,calc];
end

%normalizes the eigenvectors
%keigenface is 90000 x 20
for i=1:20
keigenfacecau(:,i)=keigenfacecau(:,i)/(norm(keigenfacecau(:,i)));	% This normalizes the “eigenfaces”
end

%project is 20 x 10
projectcau = transpose(keigenfacecau)*testsubmean;	%projection of testing images into eigenspace


reconstcau = projectcau(1,1)*keigenfacecau(:,1);		% This initializez the reconst vector of 110592 x 1.
reconstcau = reconstcau - reconstcau;
for j = 1:9					%This creates matrix of reconstructed test images.
sumR = 0;
for i = 1:20
calc = projectcau(i,j)*keigenfacecau(:,i)	;	% Calculates the reconstructed image
sumR = sumR + calc;		% Sums up entire eigenfaces and weights, obtains reconstructed image
end
reconstcau = [reconstcau, sumR];
end;
reconstcau(:,1)=[];			%This omits the first column, which was used to initialize matrix.

% adds the mean image value to the reconstructed images (using the eigenfaces)
for i=1:9
reconstcau(:,i) = reconstcau(:,i)+(meanface);
end

normtune(1:9)=0;
for i=1:9
    normtune(i) = norm(reconstcau(:,i)-doubtestimages(:,i));
end
tuneresults = mean(normtune);
end

%This will show reconstructed test image.
imshow(uint8(reshape(reconstcau(:,1),750,750)))

%project is 84 x 9
reconsteig = normc(caueigenface);
projectplot = transpose(reconsteig)*testsubmean;	%projection of testing images into eigenspace
reconst1 = projectplot(1,1)*reconsteig(:,1);		% This initializez the reconst vector of 110592 x 1.
reconst1 = reconst1 - reconst1;
testplot=0;
testplot(1:9,1:84)=0;
tempreconst=0;
tempreconst(1:84)=0;
reconstplot3=0;
reconstplot3(1:84)=0;
for j=1:9
    for k = 1:84
        sumR=0;
        for i=0:k-1
            calc = projectplot(i+1,j)*reconsteig(:,i+1);	% Calculates the reconstructed image
            sumR = sumR + calc;	
        end
        tempreconst(k)= norm(testsubmean(:,j)-sumR);
    end
    testplot(j,:)=tempreconst;
end
reconstplot3 = mean(testplot);
plot((1:83),reconstplot3(1:83),'black') %omit last two for inconsistencies
title('Reconstruction Error Plot for Cauchy Kernel')
legend('Avg Error = 1.4200 x 10^4')



%Gaussian Kernel Radial Basis Kernel with squared l2 norm
rbfkernel=0;
rbfkernel(1:84,1:84)=0;
calc = 0;

%computes variance of Euclidean Distance between all vector pairs.
normcalc=0;
normcalc(1:84,1:84)=0;
for i=1:84
    for j=1:84
        normcalc(i,j)=(dot(doubleA(:,i),doubleA(:,i))+dot(doubleA(:,j),doubleA(:,j))-2*dot(doubleA(:,i),doubleA(:,j)));
        %normcalc(i,j)=norm(doubleA(:,i)-doubleA(:,j));
    end
end
normcalc = reshape(normcalc,7056,1); %135 * 135
varrbf = var(normcalc);

%tuning sigma
tuneresults = 0;
%tuneresults(1:2)=0;
for q = 2:2
     tunecalc = q*1000000000;
     varrbf=tunecalc;


for i=1:84
    for j=1:84
       calc = exp((-1/(2*varrbf))*(dot(doubleA(:,i),doubleA(:,i))+dot(doubleA(:,j),doubleA(:,j))-2*dot(doubleA(:,i),doubleA(:,j))));
       rbfkernel(i,j)=calc;
    end
end

%centering the kernel matrix
% 1N = scalarmatrix NxN matrix with all values equal to 1/N
sizeN=size(rbfkernel);
sizeN=sizeN(1);
scalarmatrix(1:sizeN,1:sizeN)=1/sizeN;
centermatrix = rbfkernel-scalarmatrix*rbfkernel-rbfkernel*scalarmatrix+scalarmatrix*rbfkernel*scalarmatrix;

%compute eigenvectors for exponential kernel
[rbfV,rbfD] = eig(centermatrix);
rbfeigenface = submeanface*rbfV; %try this for now

%Gaussian reconstruction

%These are just the eigenvectors without the mean face added
keigenfacerbf = rbfeigenface(:,65);  		%This saves the best k=20 Eigenfaces.
calc=[];
for t = 66:84
calc= rbfeigenface(:,t);
keigenfacerbf = [keigenfacerbf,calc];
end

%normalizes the eigenvectors
%keigenface is 90000 x 20
for i=1:20
keigenfacerbf(:,i)=keigenfacerbf(:,i)/(norm(keigenfacerbf(:,i)));	% This normalizes the “eigenfaces”
end

%project is 20 x 10
projectrbf = transpose(keigenfacerbf)*testsubmean;	%projection of testing images into eigenspace


reconstrbf = projectrbf(1,1)*keigenfacerbf(:,1);		% This initializez the reconst vector of 110592 x 1.
reconstrbf = reconstrbf - reconstrbf;
for j = 1:9					%This creates matrix of reconstructed test images.
sumR = 0;
for i = 1:20
calc = projectrbf(i,j)*keigenfacerbf(:,i)	;	% Calculates the reconstructed image
sumR = sumR + calc;		% Sums up entire eigenfaces and weights, obtains reconstructed image
end
reconstrbf = [reconstrbf, sumR];
end;
reconstrbf(:,1)=[];			%This omits the first column, which was used to initialize matrix.

% adds the mean image value to the reconstructed images (using the eigenfaces)
for i=1:9
reconstrbf(:,i) = reconstrbf(:,i)+(meanface);
end

normtune(1:9)=0;
for i=1:9
    normtune(i) = norm(reconstrbf(:,i)-doubtestimages(:,i));
end
tuneresults = mean(normtune);
end

%This will show reconstructed test image.
imshow(uint8(reshape(reconstrbf(:,1),750,750)))

%project is 84 x 9
reconsteig = normc(rbfeigenface);
projectplot = transpose(reconsteig)*testsubmean;	%projection of testing images into eigenspace
reconst1 = projectplot(1,1)*reconsteig(:,1);		% This initializez the reconst vector of 110592 x 1.
reconst1 = reconst1 - reconst1;
testplot=0;
testplot(1:9,1:84)=0;
tempreconst=0;
tempreconst(1:84)=0;
reconstplot4=0;
reconstplot4(1:84)=0;
for j=1:9
    for k = 1:84
        sumR=0;
        for i=0:k-1
            calc = projectplot(84-i,j)*reconsteig(:,84-i);	% Calculates the reconstructed image
            sumR = sumR + calc;	
        end
        tempreconst(k)= norm(testsubmean(:,j)-sumR);
    end
    testplot(j,:)=tempreconst;
end
reconstplot4 = mean(testplot);
plot((1:84),reconstplot4(1:84),'m') %omit last two for inconsistencies
title('Reconstruction Error Plot for Gaussian Kernel')
legend('Avg Error = 1.2929 x 10^4')

%Laplacian Kernel Radial Basis
lapkernel(1:84,1:84)=0;
calc = 0;

%computes variance of Euclidean Distance between all vector pairs.
normcalc = 0;
normcalc(1:84,1:84)=0;
for i=1:84
    for j=1:84
        normcalc(i,j)=norm(doubleA(:,i)-doubleA(:,j));
    end
end
normcalc = reshape(normcalc,7056,1); %84 * 84
varlap = var(normcalc);

%tuning sigma
tuneresults = 0;
%tuneresults(1:1)=0;
for q = 1:1
     tunecalc = q*1000000000;
     varlap=tunecalc;

for i=1:84
    for j=1:84
       calc = exp((1/(sqrt(varlap)))*(norm(doubleA(:,i)-doubleA(:,j))));
       lapkernel(i,j)=calc;
    end
end

%centering the kernel matrix
% 1N = scalarmatrix NxN matrix with all values equal to 1/N
sizeN=size(lapkernel);
sizeN=sizeN(1);
scalarmatrix(1:sizeN,1:sizeN)=1/sizeN;
centermatrix = lapkernel-scalarmatrix*lapkernel-lapkernel*scalarmatrix+scalarmatrix*lapkernel*scalarmatrix;

%compute eigenvectors for exponential kernel
[lapV,lapD] = eig(centermatrix);
lapeigenface = submeanface*lapV; %try this for now

%Laplacian reconstruction

%These are just the eigenvectors without the mean face added
keigenfacelap = lapeigenface(:,1);  		%This saves the best k=20 Eigenfaces.
calc=[];
for t =2:20
calc= lapeigenface(:,t);
keigenfacelap = [keigenfacelap,calc];
end

%normalizes the eigenvectors
%keigenface is 90000 x 20
for i=1:20
keigenfacelap(:,i)=keigenfacelap(:,i)/(norm(keigenfacelap(:,i)));	% This normalizes the “eigenfaces”
end

%project is 20 x 10
projectlap = transpose(keigenfacelap)*testsubmean;	%projection of testing images into eigenspace


reconstlap = projectlap(1,1)*keigenfacelap(:,1);		% This initializez the reconst vector of 110592 x 1.
reconstlap = reconstlap - reconstlap;
for j = 1:9					%This creates matrix of reconstructed test images.
sumR = 0;
for i = 1:20
calc = projectlap(i,j)*keigenfacelap(:,i)	;	% Calculates the reconstructed image
sumR = sumR + calc;		% Sums up entire eigenfaces and weights, obtains reconstructed image
end
reconstlap = [reconstlap, sumR];
end;
reconstlap(:,1)=[];			%This omits the first column, which was used to initialize matrix.

% adds the mean image value to the reconstructed images (using the eigenfaces)
for i=1:9
reconstlap(:,i) = reconstlap(:,i)+(meanface);
end

normtune(1:9)=0;
for i=1:9
    normtune(i) = norm(reconstlap(:,i)-doubtestimages(:,i));
end
tuneresults = mean(normtune);
end

%This will show reconstructed test image.
imshow(uint8(reshape(reconstlap(:,1),750,750)))

%project is 84 x 9
reconsteig = normc(lapeigenface);
projectplot = transpose(reconsteig)*testsubmean;	%projection of testing images into eigenspace
reconst1 = projectplot(1,1)*reconsteig(:,1);		% This initializez the reconst vector of 110592 x 1.
reconst1 = reconst1 - reconst1;
testplot=0;
testplot(1:9,1:84)=0;
tempreconst=0;
tempreconst(1:84)=0;
reconstplot5=0;
reconstplot5(1:84)=0;
for j=1:9
    for k = 1:84
        sumR=0;
        for i=0:k-1
            calc = projectplot(i+1,j)*reconsteig(:,i+1);	% Calculates the reconstructed image
            sumR = sumR + calc;	
        end
        tempreconst(k)= norm(testsubmean(:,j)-sumR);
    end
    testplot(j,:)=tempreconst;
end
reconstplot5 = mean(testplot);
plot((1:84),reconstplot5,'c')
title('Reconstruction Error Plot for Laplacian Kernel')
legend('Avg Error = 1.2854 x 10^4')


%log kernel (contains euclidean squared norm)
logkernel(1:84,1:84)=0;
calc = 0;

% %tuning logkernel
 tuneresults = 0;
%tuneresults(1:2)=0;
 for q = 1:1
%     if q=1 
%     tunecalc = q*10000000;
%      varrbf=tunecalc;

for i=1:84
    for j=1:84
       %calc = -log((norm(doubleA(:,i)-doubleA(:,j)))+1);
        calc = -log(dot(doubleA(:,i),doubleA(:,i))+dot(doubleA(:,j),doubleA(:,j))-2*dot(doubleA(:,i),doubleA(:,j))+1);
       logkernel(i,j)=calc;
    end
end

%centering the kernel matrix
% 1N = scalarmatrix NxN matrix with all values equal to 1/N
sizeN=size(logkernel);
sizeN=sizeN(1);
scalarmatrix(1:sizeN,1:sizeN)=1/sizeN;
centermatrix = logkernel-scalarmatrix*logkernel-logkernel*scalarmatrix+scalarmatrix*logkernel*scalarmatrix;

%compute eigenvectors for exponential kernel
[logV,logD] = eig(centermatrix);
logeigenface = submeanface*logV; %try this for now

%Logarithm reconstruction

%These are just the eigenvectors without the mean face added
keigenfacelog = logeigenface(:,65);  		%This saves the best k=20 Eigenfaces.
calc=[];
for t = 66:84
calc= logeigenface(:,t);
keigenfacelog = [keigenfacelog,calc];
end

%normalizes the eigenvectors
%keigenface is 90000 x 20
for i=1:20
keigenfacelog(:,i)=keigenfacelog(:,i)/(norm(keigenfacelog(:,i)));	% This normalizes the “eigenfaces”
end

%project is 20 x 10
projectlog = transpose(keigenfacelog)*testsubmean;	%projection of testing images into eigenspace


reconstlog = projectlog(1,1)*keigenfacelog(:,1);		% This initializez the reconst vector of 110592 x 1.
reconstlog = reconstlog - reconstlog;
for j = 1:9					%This creates matrix of reconstructed test images.
sumR = 0;
for i = 1:20
calc = projectlog(i,j)*keigenfacelog(:,i)	;	% Calculates the reconstructed image
sumR = sumR + calc;		% Sums up entire eigenfaces and weights, obtains reconstructed image
end
reconstlog = [reconstlog, sumR];
end;
reconstlog(:,1)=[];			%This omits the first column, which was used to initialize matrix.

% adds the mean image value to the reconstructed images (using the eigenfaces)
for i=1:9
reconstlog(:,i) = reconstlog(:,i)+(meanface);
end

normtune(1:9)=0;
for i=1:9
    normtune(i) = norm(reconstlog(:,i)-doubtestimages(:,i));
end
tuneresults = mean(normtune);
end

%This will show reconstructed test image.
imshow(uint8(reshape(reconstlog(:,1),750,750)))

%project is 84 x 9
reconsteig = normc(logeigenface);
projectplot = transpose(reconsteig)*testsubmean;	%projection of testing images into eigenspace
reconst1 = projectplot(1,1)*reconsteig(:,1);		% This initializez the reconst vector of 110592 x 1.
reconst1 = reconst1 - reconst1;
testplot=0;
testplot(1:9,1:84)=0;
tempreconst=0;
tempreconst(1:84)=0;
reconstplot6=0;
reconstplot6(1:84)=0;
for j=1:9
    for k = 1:84
        sumR=0;
        for i=0:k-1
            calc = projectplot(84-i,j)*reconsteig(:,84-i);	% Calculates the reconstructed image
            sumR = sumR + calc;	
        end
        tempreconst(k)= norm(testsubmean(:,j)-sumR);
    end
    testplot(j,:)=tempreconst;
end
reconstplot6 = mean(testplot);
plot((1:82),reconstplot6(1:82),'green') %omit last two for inconsistencies
title('Reconstruction Error Plot for Logarithmic Kernel')
legend('Avg Error = 1.3938 x 10^4')

%power kernel (using euclidean squared norm)
powkernel(1:84,1:84)=0;
calc = 0;

for i=1:84
    for j=1:84
       calc= norm(doubleA(:,i)-doubleA(:,j),10);
        %calc = -(dot(doubleA(:,i),doubleA(:,i))+dot(doubleA(:,j),doubleA(:,j))-2*dot(doubleA(:,i),doubleA(:,j)));
       powkernel(i,j)=calc;
    end
end

%centering the kernel matrix
% 1N = scalarmatrix NxN matrix with all values equal to 1/N
sizeN=size(powkernel);
sizeN=sizeN(1);
scalarmatrix(1:sizeN,1:sizeN)=1/sizeN;
centermatrix = powkernel-scalarmatrix*powkernel-powkernel*scalarmatrix+scalarmatrix*powkernel*scalarmatrix;

%compute eigenvectors for exponential kernel
[powV,powD] = eig(centermatrix);
poweigenface = submeanface*powV; %try this for now

%Power kernel reconstruction

%These are just the eigenvectors without the mean face added
keigenfacepow = poweigenface(:,1);  		%This saves the best k=20 Eigenfaces.
calc=[];
for t = 2:20
calc= poweigenface(:,t);
keigenfacepow = [keigenfacepow,calc];
end

%normalizes the eigenvectors
%keigenface is 90000 x 20
for i=1:20
keigenfacepow(:,i)=keigenfacepow(:,i)/(norm(keigenfacepow(:,i)));	% This normalizes the “eigenfaces”
end

%project is 20 x 10
projectpow = transpose(keigenfacepow)*testsubmean;	%projection of testing images into eigenspace


reconstpow = projectpow(1,1)*keigenfacepow(:,1);		% This initializez the reconst vector of 110592 x 1.
reconstpow = reconstpow - reconstpow;
for j = 1:9					%This creates matrix of reconstructed test images.
sumR = 0;
for i = 1:20
calc = projectpow(i,j)*keigenfacepow(:,i)	;	% Calculates the reconstructed image
sumR = sumR + calc;		% Sums up entire eigenfaces and weights, obtains reconstructed image
end
reconstpow = [reconstpow, sumR];
end;
reconstpow(:,1)=[];			%This omits the first column, which was used to initialize matrix.

% adds the mean image value to the reconstructed images (using the eigenfaces)
for i=1:9
reconstpow(:,i) = reconstpow(:,i)+(meanface);
end

%This will show reconstructed test image.
imshow(uint8(reshape(reconstpow(:,1),750,750)))


%This is the hyperbolic tangent kernel

tanhkernel(1:135,1:135)=0;
calc = 0;
for i=1:135
    for j=1:135
       calc = tanh(dot(doubleA(:,i),doubleA(:,j)));
       tanhkernel(i,j)=calc;
    end
end

%centering the kernel matrix
% 1N = scalarmatrix NxN matrix with all values equal to 1/N
sizeN=size(tanhkernel);
sizeN=sizeN(1);
scalarmatrix(1:sizeN,1:sizeN)=1/sizeN;
centermatrix = tanhkernel-scalarmatrix*tanhkernel-tanhkernel*scalarmatrix+scalarmatrix*tanhkernel*scalarmatrix;

%compute eigenvectors for tanh kernel
[tanhV,tanhD] = eig(centermatrix);
tanheigenface = submeanface*tanhV; %try this for now

%tanh reconstruction

%These are just the eigenvectors without the mean face added
keigenfacetanh = tanheigenface(:,51);  		%This saves the best k=20 Eigenfaces.
calc=[];
for t = 52:70
calc= tanheigenface(:,t);
keigenfacetanh = [keigenfacetanh,calc];
end

%normalizes the eigenvectors
%keigenface is 90000 x 20
for i=1:20
keigenfacetanh(:,i)=keigenfacetanh(:,i)/(norm(keigenfacetanh(:,i)));	% This normalizes the “eigenfaces”
end

%project is 20 x 10
projecttanh = transpose(keigenfacetanh)*testsubmean;	%projection of testing images into eigenspace


reconsttanh = projecttanh(1,1)*keigenfacetanh(:,1);		% This initializez the reconst vector of 110592 x 1.
reconsttanh = reconsttanh - reconsttanh;
for j = 1:10					%This creates matrix of reconstructed test images.
sumR = 0;
for i = 1:20
calc = projecttanh(i,j)*keigenfacetanh(:,i)	;	% Calculates the reconstructed image
sumR = sumR + calc;		% Sums up entire eigenfaces and weights, obtains reconstructed image
end
reconsttanh = [reconsttanh, sumR];
end;
reconsttanh(:,1)=[];			%This omits the first column, which was used to initialize matrix.

% adds the mean image value to the reconstructed images (using the eigenfaces)
for i=1:10
reconsttanh(:,i) = reconsttanh(:,i)+(meanface);
end

%This will show reconstructed test image.
imshow(uint8(reshape(reconsttanh(:,1),300,300)))




%% good reconstruction

projectplot1 = transpose(normc(eigenvect))*testsubmean; %70 x 10 reconstplot(1:150)=0;
reconstimage(1:90000,1:10)=0;
difference(1:10)=0;
reconstplot(1:70)=0;

for k = 1:70
for j = 1:10					%This creates matrix of reconstructed test images.
sumR(1:90000,1)= 0;
for i = 70:-1:(70-k);
calc = projectplot1(i,j)*(normc(eigenface(:,i)));	% Calculates the reconstructed image
sumR = calc + sumR;		% Sums up entire eigenfaces and weights, obtains rec image
end
reconstimage(:,j) = sumR;            %65536x1 vector of image j

difference(j) = mean(mean((abs(reshape(doubtestimages(:,j),384,288)-reshape(reconstimage(:,j)+meanface,300,300))).^2,2));
end
reconstplot(k) = mean(difference);
end
plot((1:70),reconstplot/2000);

