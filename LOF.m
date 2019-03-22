function lof = LOF(dist)
%% 输入参数说明
% dist :维度m-by-m，表示每个点到另外所有点之间的距离的矩阵
%       每一行(列)表示该第i个样本点，到另外所有样本点的行(列)向量
%       dist矩阵为对称矩阵
% 
%% 输出参数说明：
% lof：表示local outlier factor，局部离群点系数

%% 
K=10;
m=size(dist,1);
reach_distance=zeros(m,m);
lrd=zeros(m,1);
lof=zeros(m,1);
%%
%计算k_distance
distance=sort(dist,2,'ascend');
k_distance=distance(:,K+1);
index=dist<k_distance;
num_k=sum(index,2);
%%
%计算可达距离
for i=1:m
    for j=i+1:m
        reach_distance(i,j)=max(k_distance(i),dist(i,j));
        reach_distance(j,i)=reach_distance(i,j);
    end
end
%% 
% 计算lrd
for i=1:m
    lrd(i)=num_k(i)/sum(reach_distance(i,index(i,:)));
end
%%
%计算lof
for i=1:m
    lof(i)=sum(lrd(index(i,:)))/(num_k(i).*lrd(i));
end
end

