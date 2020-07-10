% load 3D knee data
fixed = niftiinfo('./data/knee1FS time1 337172.nii');

fix = niftiread(fixed); % mov and fix are integer arrays
fix_n=double(fix)./(max(max(max(double(fix)))));


for c = 1:10
     T = randomAffine3d('Rotation',[-30 30]);
     sameAsInput = affineOutputView(size(fix),T,'BoundsStyle','SameAsInput');
     fix2_n = imwarp(fix_n,T,'OutputView',sameAsInput);
     
     save(['new_' num2str(c) '.mat'],'fix2_n');
end

fix2 = imwarp(fix,T,'OutputView',sameAsInput);

%% show image pairs %%
figure(1), imshowpair(fix(:,:,100),fix2(:,:,100))
imshowpair(fix_n(:,:,100),fix2_n(:,:,100))

% %%%% 3D Affine REGISTRATION on MATLAB %%%
% %%%% these are hyperparameters for registration %%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% optimizer 
% [optimizer, metric] = imregconfig('multimodal');
% optimizer.InitialRadius = 0.001;
% optimizer.Epsilon = 1e-4;
% optimizer.GrowthFactor = 1.01;
% optimizer.MaximumIterations = 1000;
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
% % matlab registration (should take 5-10 min)
% 
% T2 = imregtform(fix2,fix,"affine",optimizer,metric);
% 
% fix2reg = imwarp(fix2,T2,'OutputView',imref3d(size(fix)));
% %fix2bigreg = imwarp(fix2big,T2,'OutputView',imref3d(size(fix)));
% 
% figure(2),imshowpair(fix(:,:,90),fix2reg(:,:,90))