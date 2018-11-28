%% load data
M = csvread('meatspec.csv',1);
fat = M(:,101);
spec = M(:,1:100);
M0= csvread('meatspec.csv')
summary(M)
%% Ridge Regression
lamda = 0:1e-5:5e-3;
b = ridge(fat,spec,lamda);
c= ridge(fat,spec,0,1) 
e=ridge(fat,spec,0,0)
d=regress(fat,spec)
mean((fat-spec*d).^2)
fat-spec*c
ridge2=ridge(fat,spec,0,0)
ridge2(2:end)
%% Ridge Trace
figure
plot(lamda,b,'LineWidth',2)
ylim([-100 100])
grid on
xlabel('Ridge Parameter')
ylabel('Standardized Coefficient')
title('{\bf Ridge Trace}')
%%
%spec=[ones(length(spec),1) spec] ��ʹridgeϵ��102��
[ones(length(fat),1),fat]
%ridgef=@(XTRAIN,ytrain,XTEST)(XTEST*ridge(ytrain,XTRAIN,0,0));
ridgef=@(XTRAIN,ytrain,XTEST)([ones(length(XTEST),1),XTEST]*ridge(ytrain,XTRAIN,0,0));
%ridgef(spec(1:10,:),fat(1:10,:),spec)������
%ridgef2(spec,fat,spec)���ᱨ��
%size([ones(length(spec),1) spec]*ridge(fat,spec,0,0))���ᱨ��
%ridgef=@(XTRAIN,ytrain,XTEST)(XTEST*(ridge(ytrain,XTRAIN,0,0)))
ridgef=@(XTRAIN,ytrain,XTEST)([ones(length(XTEST),101)]*ridge(ytrain,XTRAIN,0,0))


cvMse(:,i) = crossval('mse',spec,fat,'predfun',ridgef2,'kfold',10)
%% дһ������lamda��ѭ��,plot
%cross validation�ô�֮һ����ѡȡ������
i=1
for lamda=0:1e-11:1e-9
ridgef=@(XTRAIN,ytrain,XTEST)(XTEST*ridge(ytrain,XTRAIN,lamda,1));
cvMse(:,i) = crossval('mse',spec,fat,'predfun',ridgef,'kfold',10)
i=i+1
end
[B,I]=min(cvMse)
plot(lamda,cvMse,'o-')
%% debug XTESTά�� 
% lamda=0:1e-7:10e-5
i=1
for lamda=0:1e-7:10e-5
ridgef=@(XTRAIN,ytrain,XTEST)([ones(min(size(XTEST)),1) XTEST]*ridge(ytrain,XTRAIN,lamda,0));
cvMse(:,i) = crossval('mse',spec,fat,'predfun',ridgef,'kfold',10)
i=i+1
end
[B,I]=min(cvMse)
lamda=0:1e-7:10e-5
plot(lamda,cvMse,'o-')
xlabel('Ridge Parameter')
ylabel('MSE on the validation set')
%% lamda=0:1e-10:3e-8
%not choose mse, GCV; split data with; set seed
i=1
for lamda=0:1e-10:3e-8
ridgef=@(XTRAIN,ytrain,XTEST)([ones(min(size(XTEST)),1) XTEST]*ridge(ytrain,XTRAIN,lamda,0));
cvMse(:,i) = crossval('mse',spec,fat,'predfun',ridgef,'kfold',10)
i=i+1
end
[B,I]=min(cvMse)
lamda=0:1e-10:3e-8
plot(lamda,cvMse,'o-')
%% Ridge Not rescaled
i=1
for lamda=0:1e-10:3e-8
ridgef=@(XTRAIN,ytrain,XTEST)(XTEST*ridge(ytrain,XTRAIN,lamda,1));
cvMse = crossval('mse',spec,fat,'predfun',ridgef,'kfold',10)
cvRmse(:,i) = sqrt(cvMse)
i=i+1
end
[B,I]=min(cvRmse)
Lamda=0:1e-10:3e-8
plot(Lamda,cvRmse,'o-')
axis([0 1.2e-9 6 8])
xlim([0,1.2e-9])
%% ��ʹ��cross
i=1
spec_Train=spec(1:172,:)
spec_Test=spec(173:end,:)
fat_Train=fat(1:172,:)
fat_Test=fat(173:end,:)
for lamda=0:1e-7:10e-5
    fathat=[ones(43,1) spec_Test]*ridge(fat_Train,spec_Train,lamda,0)
    cvMse=mean((fat_Test-fathat).^2)
    cvRmse(:,i) = sqrt(cvMse)
i=i+1
end
[B,I]=min(cvRmse)
[a,b]=max(cvRmse)
%66e-10Ϊ���ֵ��

Lamda=0:1e-7:10e-5
plot(Lamda,cvRmse,'o-')
%% Try GCV
i=1
spec_Train=spec(1:172,:)
spec_Test=spec(173:end,:)
fat_Train=fat(1:172,:)
fat_Test=fat(173:end,:)
for lamda=0:1e-7:10e-5
    fathat=[ones(43,1) spec_Test]*ridge(fat_Train,spec_Train,lamda,0)
    gcv=43*mean((fat_Test-fathat).^2)/(43-)
    
i=i+1
end
[B,I]=min(cvRmse)
[a,b]=max(cvRmse)
%66e-10Ϊ���ֵ��

Lamda=0:1e-7:10e-5
plot(Lamda,cvRmse,'o-')

%%
x = M(:,101);
y = M(:,1:100)
N = length(x)
sse = 0;
indices = crossvalind('Kfold',N,10)
for i = 1:100
    [train,test] = crossvalind('Kfold',20,5);
    yhat = ridge(x(train),y(train),2)* x(test);
    sse = sse + sum((yhat - y(test)).^2);
end
CVerr = sse / 100
%%
load('fisheriris')
y = meas(:,1)
X = [ones(size(y,1),1),meas(:,2:4)]
regf=@(XTRAIN,ytrain,XTEST)(XTEST*regress(ytrain,XTRAIN));
%cvMse = crossval('mse',X,y,'predfun',regf)