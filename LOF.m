function lof = LOF(dist)
%% �������˵��
% dist :ά��m-by-m����ʾÿ���㵽�������е�֮��ľ���ľ���
%       ÿһ��(��)��ʾ�õ�i�������㣬�������������������(��)����
%       dist����Ϊ�Գƾ���
% 
%% �������˵����
% lof����ʾlocal outlier factor���ֲ���Ⱥ��ϵ��

%% 
K=10;
m=size(dist,1);
reach_distance=zeros(m,m);
lrd=zeros(m,1);
lof=zeros(m,1);
%%
%����k_distance
distance=sort(dist,2,'ascend');
k_distance=distance(:,K+1);
index=dist<k_distance;
num_k=sum(index,2);
%%
%����ɴ����
for i=1:m
    for j=i+1:m
        reach_distance(i,j)=max(k_distance(i),dist(i,j));
        reach_distance(j,i)=reach_distance(i,j);
    end
end
%% 
% ����lrd
for i=1:m
    lrd(i)=num_k(i)/sum(reach_distance(i,index(i,:)));
end
%%
%����lof
for i=1:m
    lof(i)=sum(lrd(index(i,:)))/(num_k(i).*lrd(i));
end
end

