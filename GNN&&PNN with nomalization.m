clear all
clc
%% 导入数据集
load BreastTissue_data.mat
% 从数据集可以看出一共是106个样本，9个特征
% 本例属于分类问题
P_train=[];
T_train=[];
P_test=[];
T_test=[];
idx=cell(1,6);
% GRNN和PNN不需要数据归一化


%% 划分训练集和测试集
% 注意本例的各类样本个数不一样

for i=1:6
    idx(1,i)={find(label==i)};  %找到每种样本对应的下标，存入一个元胞数组，方便后面划分数据集的时候调用
end

for i=1:6
    index=idx{1,i}; %得到当前这个种类的样本在数据集中的下标（向量）
    len=length(index);
    n=round(len*0.8); %按照一定比例划分数据集
    temp=randperm(len);
    P_train=[P_train,matrix(index(temp(1:n)),:)'];  %直接把训练集转化成矩阵的列表示样本数目的形式了
    T_train=[T_train,label(index(temp(1:n)),:)'];  %也可以采用[P_train;matrix(...)]的形式
    P_test=[P_test,matrix(index(temp(n+1:end)),:)'];
    T_test=[T_test,label(index(temp(n+1:end)),:)'];
end
%% 归一化数据
[P_train,ps_input]=mapminmax(P_train);
P_test=mapminmax('apply',P_test,ps_input);
[T_train,ps_output]=mapminmax(T_train);

%% 创建神经网络
result_grnn=[];
result_pnn=[];
for spread=0.01:0.03:1
    net_grnn=newgrnn(P_train,T_train,spread);
    Tc_train=ind2vec(mapminmax('reverse',T_train,ps_output));
    net_pnn=newpnn(P_train,Tc_train,spread);

    %% 仿真测试
    T_pred_grnn=sim(net_grnn,P_test);
    Tc_pred_pnn=sim(net_pnn,P_test);
    T_pred_pnn=vec2ind(Tc_pred_pnn);
    %反向归一化
    T_pred_grnn=round(mapminmax('reverse',T_pred_grnn,ps_output));
    %T_pred_pnn=mapminmax('reverse',T_pred_pnn,ps_output);
    result_grnn=[result_grnn,T_pred_grnn'];
    result_pnn=[result_pnn,T_pred_pnn'];
end
%% 计算准确率
accuracy_grnn=[];
accuracy_pnn=[];
for i=1:length(result_grnn)
    accuracy_1=length(find(result_grnn(:,i)==T_test'))/length(T_test);
    accuracy_2=length(find(result_pnn(:,i)==T_test'))/length(T_test);
    accuracy_grnn=[accuracy_grnn,accuracy_1];
    accuracy_pnn=[accuracy_pnn,accuracy_2];
end
%% 下面进行画图操作
figure(1);
N=length(T_test);
plot(1:N,T_test,'bo-',1:N,result_grnn(:,4),'g*-',1:N,result_pnn(:,4),'r-^')
xlabel('预测样本编号')
ylabel('预测值')
legend('真实值','GRNN预测值','PNN预测值')
string={'测试集预测结果对比GRNN vs PNN',['正确率',num2str(accuracy_grnn(4)*100) '%(GRNN)vs' num2str(accuracy_pnn(4)*100) '%PNN']};
title(string)
figure(2);
spread=0.01:0.03:1;
plot(spread,accuracy_grnn,'b-o',spread,accuracy_pnn,'r-*');
legend('GRNN','PNN')
xlabel('spread')
ylabel('准确率')
title('准确率随spread变化图')