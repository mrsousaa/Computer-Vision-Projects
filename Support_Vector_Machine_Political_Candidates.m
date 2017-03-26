bin_trait_annotation=2*(trait_annotation > 0)-1;

%This reads in all HoG features of 491 images
saveHoG(1:491,1:119072)=0;
for i=1:9
    im = imread(['C:\Users\Luis\Documents\Graduate Statistics UCLA\Pattern Recognition and Machine Learning - Stats 231\Project 3\stat-proj\img\M000',num2str(i),'.jpg']);
    im = double(im);
    hogfeat = HoGfeatures(im);
    saveHoG(i,:) = reshape(hogfeat,119072,1);
end

for i=10:99
    im = imread(['C:\Users\Luis\Documents\Graduate Statistics UCLA\Pattern Recognition and Machine Learning - Stats 231\Project 3\stat-proj\img\M00',num2str(i),'.jpg']);
    im = double(im);
    hogfeat = HoGfeatures(im);
    saveHoG(i,:) = reshape(hogfeat,119072,1);
end

for i=100:491
    im = imread(['C:\Users\Luis\Documents\Graduate Statistics UCLA\Pattern Recognition and Machine Learning - Stats 231\Project 3\stat-proj\img\M0',num2str(i),'.jpg']);
    im = double(im);
    hogfeat = HoGfeatures(im);
    saveHoG(i,:) = reshape(hogfeat,119072,1);
end

%Support Vector Regression
model1 = svmtrain(trait_annotation(1:250,1),face_landmark(1:250,:),'-s 3 -t 2 -c 20 -g 64 -p 1');
[a,b,c]=svmpredict(trait_annotation(251:400,1),face_landmark(251:400,:),model1);

%Support Vector Regression
model1 = svmtrain(trait_annotation(1:250,1),face_landmark(1:250,:),'-s 3 -t 2 -c 5 -g 0.00000838');
[a,b,c]=svmpredict(trait_annotation(251:400,1),face_landmark(251:400,:),model1);

%Support Vector Regression
model1 = svmtrain(trait_annotation(1:250,1),face_landmark(1:250,:),'-s 3');
[a,b,c]=svmpredict(trait_annotation(251:400,1),face_landmark(251:400,:),model1);


%Binary Support Vector Machine accuracy, for Landmark only
accuracy(1:10,1:14)=0;
for i = 3:3
    model1 = svmtrain(bin_trait_annotation(50:491,i),face_landmark(50:491,:),'-t 0');
    [a,b,c]=svmpredict(bin_trait_annotation(1:49,i),face_landmark(1:49,:),model1);
    accuracy(1,i)=b(1);
    
    model1 = svmtrain(bin_trait_annotation([1:49,99:491],i),face_landmark([1:49,99:491],:),'-t 0');
    [a,b,c]=svmpredict(bin_trait_annotation(50:98,i),face_landmark(50:98,:),model1);
    accuracy(2,i)=b(1);
    
    model1 = svmtrain(bin_trait_annotation([1:98,148:491],i),face_landmark([1:98,148:491],:),'-t 0');
    [a,b,c]=svmpredict(bin_trait_annotation(99:147,i),face_landmark(99:147,:),model1);
    accuracy(3,i)=b(1);
    
    model1 = svmtrain(bin_trait_annotation([1:147,197:491],i),face_landmark([1:147,197:491],:),'-t 0');
    [a,b,c]=svmpredict(bin_trait_annotation(148:196,i),face_landmark(148:196,:),model1);
    accuracy(4,i)=b(1);
    
    model1 = svmtrain(bin_trait_annotation([1:196,246:491],i),face_landmark([1:196,246:491],:),'-t 0');
    [a,b,c]=svmpredict(bin_trait_annotation(197:245,i),face_landmark(197:245,:),model1);
    accuracy(5,i)=b(1);
    
    model1 = svmtrain(bin_trait_annotation([1:245,295:491],i),face_landmark([1:245,295:491],:),'-t 0');
    [a,b,c]=svmpredict(bin_trait_annotation(246:294,i),face_landmark(246:294,:),model1);
    accuracy(6,i)=b(1);
    
    model1 = svmtrain(bin_trait_annotation([1:294,344:491],i),face_landmark([1:294,344:491],:),'-t 0');
    [a,b,c]=svmpredict(bin_trait_annotation(295:343,i),face_landmark(295:343,:),model1);
    accuracy(7,i)=b(1);
    
    model1 = svmtrain(bin_trait_annotation([1:343,393:491],i),face_landmark([1:343,393:491],:),'-t 0');
    [a,b,c]=svmpredict(bin_trait_annotation(344:392,i),face_landmark(344:392,:),model1);
    accuracy(8,i)=b(1);
    
    model1 = svmtrain(bin_trait_annotation([1:393,442:491],i),face_landmark([1:393,442:491],:),'-t 0');
    [a,b,c]=svmpredict(bin_trait_annotation(393:441,i),face_landmark(393:441,:),model1);
    accuracy(9,i)=b(1);
    
    model1 = svmtrain(bin_trait_annotation(1:441,i),face_landmark(1:441,:),'-t 0');
    [a,b,c]=svmpredict(bin_trait_annotation(442:491,i),face_landmark(442:491,:),model1);
    accuracy(10,i)=b(1); 
end


%Binary Support Vector Machine accuracy, for HoG only

for i = 1:14
    model1 = svmtrain(bin_trait_annotation(100:450,i),saveHoG(100:450,:),'-t 0');
    [a,b,c]=svmpredict(bin_trait_annotation(1:99,i),saveHoG(1:99,:),model1);
    accuracy(i)=b(1);
end

%Binary Support Vector Machine accuracy for Landmark and HoG
landmarkHoG = [face_landmark, saveHoG];
accuracy3(1:10,1:14)=0;
for i = 1:14
    model1 = svmtrain(bin_trait_annotation(50:491,i),landmarkHoG(50:491,:),'-t 0');
    [a,b,c]=svmpredict(bin_trait_annotation(1:49,i),landmarkHoG(1:49,:),model1);
    accuracy3(1,i)=b(1);
    
    model1 = svmtrain(bin_trait_annotation([1:49,99:491],i),landmarkHoG([1:49,99:491],:),'-t 0');
    [a,b,c]=svmpredict(bin_trait_annotation(50:98,i),landmarkHoG(50:98,:),model1);
    accuracy3(2,i)=b(1);
    
    model1 = svmtrain(bin_trait_annotation([1:98,148:491],i),landmarkHoG([1:98,148:491],:),'-t 0');
    [a,b,c]=svmpredict(bin_trait_annotation(99:147,i),landmarkHoG(99:147,:),model1);
    accuracy3(3,i)=b(1);
    
    model1 = svmtrain(bin_trait_annotation([1:147,197:491],i),landmarkHoG([1:147,197:491],:),'-t 0');
    [a,b,c]=svmpredict(bin_trait_annotation(148:196,i),landmarkHoG(148:196,:),model1);
    accuracy3(4,i)=b(1);
    
    model1 = svmtrain(bin_trait_annotation([1:196,246:491],i),landmarkHoG([1:196,246:491],:),'-t 0');
    [a,b,c]=svmpredict(bin_trait_annotation(197:245,i),landmarkHoG(197:245,:),model1);
    accuracy3(5,i)=b(1);
    
    model1 = svmtrain(bin_trait_annotation([1:245,295:491],i),landmarkHoG([1:245,295:491],:),'-t 0');
    [a,b,c]=svmpredict(bin_trait_annotation(246:294,i),landmarkHoG(246:294,:),model1);
    accuracy3(6,i)=b(1);
    
    model1 = svmtrain(bin_trait_annotation([1:294,344:491],i),landmarkHoG([1:294,344:491],:),'-t 0');
    [a,b,c]=svmpredict(bin_trait_annotation(295:343,i),landmarkHoG(295:343,:),model1);
    accuracy3(7,i)=b(1);
    
    model1 = svmtrain(bin_trait_annotation([1:343,393:491],i),landmarkHoG([1:343,393:491],:),'-t 0');
    [a,b,c]=svmpredict(bin_trait_annotation(344:392,i),landmarkHoG(344:392,:),model1);
    accuracy3(8,i)=b(1);
    
    model1 = svmtrain(bin_trait_annotation([1:393,442:491],i),landmarkHoG([1:393,442:491],:),'-t 0');
    [a,b,c]=svmpredict(bin_trait_annotation(393:441,i),landmarkHoG(393:441,:),model1);
    accuracy3(9,i)=b(1);
    
    model1 = svmtrain(bin_trait_annotation(1:441,i),landmarkHoG(1:441,:),'-t 0');
    [a,b,c]=svmpredict(bin_trait_annotation(442:491,i),landmarkHoG(442:491,:),model1);
    accuracy3(10,i)=b(1); 
end



% Finding best parameter for SVR
accuracyresult(1:100,1:100)=0;
for i = 0.1:0.1:10
    for j = 0.1:0.1:10
        cmd = ['-s 3 -t 2 -c ', num2str(i),' -g ', num2str(j),' -p 1'];
        model1 = svmtrain(trait_annotation(1:250,1),face_landmark(1:250,:),cmd);
        [a,b,c]=svmpredict(trait_annotation(251:400,1),face_landmark(251:400,:),model1);
        index1=int8(10*i);
        index2=int8(10*j);
        accuracyresult(index1,index2)=b(1);
    end
end


%Part 2

%This reads in all HoG features for senators
senatorHoG(1:116,1:119072)=0;
for i=1:9
    im = imread(['C:\Users\Luis\Documents\Graduate Statistics UCLA\Pattern Recognition and Machine Learning - Stats 231\Project 3\stat-proj\img-elec\senator\S000',num2str(i),'.jpg']);
    im = double(im);
    hogfeat = HoGfeatures(im);
    senatorHoG(i,:) = reshape(hogfeat,119072,1);
end

for i=10:99
    im = imread(['C:\Users\Luis\Documents\Graduate Statistics UCLA\Pattern Recognition and Machine Learning - Stats 231\Project 3\stat-proj\img-elec\senator\S00',num2str(i),'.jpg']);
    im = double(im);
    hogfeat = HoGfeatures(im);
    senatorHoG(i,:) = reshape(hogfeat,119072,1);
end

for i=100:116
    im = imread(['C:\Users\Luis\Documents\Graduate Statistics UCLA\Pattern Recognition and Machine Learning - Stats 231\Project 3\stat-proj\img-elec\senator\S0',num2str(i),'.jpg']);
    im = double(im);
    hogfeat = HoGfeatures(im);
    senatorHoG(i,:) = reshape(hogfeat,119072,1);
end



%This reads in all HoG features for governors
governorHoG(1:112,1:119072)=0;
for i=1:9
    im = imread(['C:\Users\Luis\Documents\Graduate Statistics UCLA\Pattern Recognition and Machine Learning - Stats 231\Project 3\stat-proj\img-elec\governor\G000',num2str(i),'.jpg']);
    im = double(im);
    hogfeat = HoGfeatures(im);
    governorHoG(i,:) = reshape(hogfeat,119072,1);
end

for i=10:99
    im = imread(['C:\Users\Luis\Documents\Graduate Statistics UCLA\Pattern Recognition and Machine Learning - Stats 231\Project 3\stat-proj\img-elec\governor\G00',num2str(i),'.jpg']);
    im = double(im);
    hogfeat = HoGfeatures(im);
    governorHoG(i,:) = reshape(hogfeat,119072,1);
end

for i=100:112
    im = imread(['C:\Users\Luis\Documents\Graduate Statistics UCLA\Pattern Recognition and Machine Learning - Stats 231\Project 3\stat-proj\img-elec\governor\G0',num2str(i),'.jpg']);
    im = double(im);
    hogfeat = HoGfeatures(im);
    governorHoG(i,:) = reshape(hogfeat,119072,1);
end


senatorlandmarkHoG = [face_landmark, senatorHoG];
%These are the "trait" results for senators
senpredicttraits(1:116,1:14)=0;
senaccuracy(1:14)=0;
for i = 1:14
    %model1 = svmtrain(bin_trait_annotation(1:491,i),landmarkHoG(1:491,:),'-s 3 -c 5 -g 0.00000836');
    model1 = svmtrain(bin_trait_annotation(1:491,i),landmarkHoG(1:491,:),'t 0')
    [a,b,c]=svmpredict(bin_trait_annotation(1:116,i),senatorlandmarkHoG(1:116,:),model1);
    senpredicttraits(:,i)=a;
    senaccuracy(i)=b(1);
end

%Use this to compute difference in traits between candidates
binsenvotediff=2*(vote_diff > 0)-1;
diffsenpredicttraits(1:116,1:14)=0;
for i=1:2:115
    diffsenpredicttraits(i,:)= senpredicttraits(i,:)-senpredicttraits(i+1,:);
    diffsenpredicttraits(i+1,:)= senpredicttraits(i+1,:)-senpredicttraits(i,:);
end

%training the Senator Election winner classifier
senwinaccuracy(1:10)=0;
model1 = svmtrain(binsenvotediff(12:116),diffsenpredicttraits(12:116,:),'-t 0');
[a,b,c]=svmpredict(binsenvotediff(1:11),diffsenpredicttraits(1:11,:),model1);
senwinaccuracy(1)=b(1);
model1 = svmtrain(binsenvotediff([1:11,23:116]),diffsenpredicttraits([1:11,23:116],:),'-t 0');
[a,b,c]=svmpredict(binsenvotediff(12:22),diffsenpredicttraits(12:22,:),model1);
senwinaccuracy(2)=b(1);
model1 = svmtrain(binsenvotediff([1:22,34:116]),diffsenpredicttraits([1:22,34:116],:),'-t 0');
[a,b,c]=svmpredict(binsenvotediff(23:33),diffsenpredicttraits(23:33,:),model1);
senwinaccuracy(3)=b(1);
model1 = svmtrain(binsenvotediff([1:33,45:116]),diffsenpredicttraits([1:33,45:116],:),'-t 0');
[a,b,c]=svmpredict(binsenvotediff(34:44),diffsenpredicttraits(34:44,:),model1);
senwinaccuracy(4)=b(1);
model1 = svmtrain(binsenvotediff([1:44,56:116]),diffsenpredicttraits([1:44,56:116],:),'-t 0');
[a,b,c]=svmpredict(binsenvotediff(45:55),diffsenpredicttraits(45:55,:),model1);
senwinaccuracy(5)=b(1);
model1 = svmtrain(binsenvotediff([1:55,67:116]),diffsenpredicttraits([1:55,67:116],:),'-t 0');
[a,b,c]=svmpredict(binsenvotediff(56:66),diffsenpredicttraits(56:66,:),model1);
senwinaccuracy(6)=b(1);
model1 = svmtrain(binsenvotediff([1:66,78:116]),diffsenpredicttraits([1:66,78:116],:),'-t 0');
[a,b,c]=svmpredict(binsenvotediff(67:77),diffsenpredicttraits(67:77,:),model1);
senwinaccuracy(7)=b(1);
model1 = svmtrain(binsenvotediff([1:77,89:116]),diffsenpredicttraits([1:77,89:116],:),'-t 0');
[a,b,c]=svmpredict(binsenvotediff(78:88),diffsenpredicttraits(78:88,:),model1);
senwinaccuracy(8)=b(1);
model1 = svmtrain(binsenvotediff([1:88,101:116]),diffsenpredicttraits([1:88,101:116],:),'-t 0');
[a,b,c]=svmpredict(binsenvotediff(89:100),diffsenpredicttraits(89:100,:),model1);
senwinaccuracy(9)=b(1);
model1 = svmtrain(binsenvotediff(1:100),diffsenpredicttraits(1:100,:),'-t 0');
[a,b,c]=svmpredict(binsenvotediff(101:116),diffsenpredicttraits(101:116,:),model1);
senwinaccuracy(10)=b(1);

governorlandmarkHoG = [face_landmark, governorHoG];
%These are the "trait" results for governors
govpredicttraits(1:112,1:14)=0;
govaccuracy(1:14)=0;
for i = 1:14
    model1 = svmtrain(bin_trait_annotation(1:491,i),landmarkHoG(1:491,:),'-t 0');
    %model1 = svmtrain(trait_annotation(1:491,i),landmarkHoG(1:491,:),'-s 3 -c 5 -g 0.00000836');
    [a,b,c]=svmpredict(bin_trait_annotation(1:112,i),governorlandmarkHoG(1:112,:),model1);
    govpredicttraits(:,i)=a;
    govaccuracy(i)=b(1);
end

%Use this to compute difference in traits between candidates
bingovvotediff=2*(vote_diff > 0)-1;
diffgovpredicttraits(1:112,1:14)=0;
for i=1:2:111
    diffgovpredicttraits(i,:)= govpredicttraits(i,:)-govpredicttraits(i+1,:);
    diffgovpredicttraits(i+1,:)= govpredicttraits(i+1,:)-govpredicttraits(i,:);
end

%training the Governor Election winner classifier
govwinaccuracy(1:10)=0;
model1 = svmtrain(bingovvotediff(12:112),diffgovpredicttraits(12:112,:),'-t 0');
[a,b,c]=svmpredict(bingovvotediff(1:11),diffgovpredicttraits(1:11,:),model1);
govwinaccuracy(1)=b(1);
model1 = svmtrain(bingovvotediff([1:11,23:112]),diffgovpredicttraits([1:11,23:112],:),'-t 0');
[a,b,c]=svmpredict(bingovvotediff(12:22),diffgovpredicttraits(12:22,:),model1);
govwinaccuracy(2)=b(1);
model1 = svmtrain(bingovvotediff([1:22,34:112]),diffgovpredicttraits([1:22,34:112],:),'-t 0');
[a,b,c]=svmpredict(bingovvotediff(23:33),diffgovpredicttraits(23:33,:),model1);
govwinaccuracy(3)=b(1);
model1 = svmtrain(bingovvotediff([1:33,45:112]),diffgovpredicttraits([1:33,45:112],:),'-t 0');
[a,b,c]=svmpredict(bingovvotediff(34:44),diffgovpredicttraits(34:44,:),model1);
govwinaccuracy(4)=b(1);
model1 = svmtrain(bingovvotediff([1:44,56:112]),diffgovpredicttraits([1:44,56:112],:),'-t 0');
[a,b,c]=svmpredict(bingovvotediff(45:55),diffgovpredicttraits(45:55,:),model1);
govwinaccuracy(5)=b(1);
model1 = svmtrain(bingovvotediff([1:55,67:112]),diffgovpredicttraits([1:55,67:112],:),'-t 0');
[a,b,c]=svmpredict(bingovvotediff(56:66),diffgovpredicttraits(56:66,:),model1);
govwinaccuracy(6)=b(1);
model1 = svmtrain(bingovvotediff([1:66,78:112]),diffgovpredicttraits([1:66,78:112],:),'-t 0');
[a,b,c]=svmpredict(bingovvotediff(67:77),diffgovpredicttraits(67:77,:),model1);
govwinaccuracy(7)=b(1);
model1 = svmtrain(bingovvotediff([1:77,89:112]),diffgovpredicttraits([1:77,89:112],:),'-t 0');
[a,b,c]=svmpredict(bingovvotediff(78:88),diffgovpredicttraits(78:88,:),model1);
govwinaccuracy(8)=b(1);
model1 = svmtrain(bingovvotediff([1:88,101:112]),diffgovpredicttraits([1:88,101:112],:),'-t 0');
[a,b,c]=svmpredict(bingovvotediff(89:100),diffgovpredicttraits(89:100,:),model1);
govwinaccuracy(9)=b(1);
model1 = svmtrain(bingovvotediff(1:100),diffgovpredicttraits(1:100,:),'-t 0');
[a,b,c]=svmpredict(bingovvotediff(101:112),diffgovpredicttraits(101:112,:),model1);
govwinaccuracy(10)=b(1);



%test
accuracytest(1:10,1:14)=0;
for i = 1:1
    model1 = svmtrain(bin_trait_annotation(1:491,i),landmarkHoG(1:491,:),'-t 0');
    [a1,b1,c1]=svmpredict(bin_trait_annotation(117:232,i),senatorlandmarkHoG(1:116,:),model1);
    accuracytest(1,i)=b(1);
end

%test correlation
govcorr =corrcoef([bingovvotediff,diffgovpredicttraits]);
sencorr = corrcoef([binsenvotediff,diffsenpredicttraits]);
max(govsen(1));
max(govcorr(1));

%test average%
averageacc(1:14)=0;
for i=1:14
    averageacc(i)=mean(accuracy3(:,i));
end
