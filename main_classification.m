%% iris数据集识别
%%
clear all;
clc;
%% 
% 加载数据集
load iris_data.mat

%%
%测试集、训练集的划分
%% 注意这里分类数据集的划分方式
p_train=[];
t_train=[];
p_test=[];
t_test=[];
n=randperm(size(classes,1));
for i=1:3  %for 循环分段划分，然后再拼接到一起
    temp_input=features((i-1)*50+1:i*50,:);
    temp_output=classes((i-1)*50+1:i*50,:);
    n=randperm(50);
    % 训练集--120个样本
    p_train=[p_train,temp_input(n(1:40),:)'];
    t_train=[t_train,temp_output(n(1:40),:)'];
    % 测试集--30个样本
    p_test=[p_test,temp_input(n(41:50),:)'];
    t_test=[t_test,temp_output(n(41:50),:)'];
    %仔细体会一下这一段关于多标签分类的数据集的划分
end


%% 
% 记录测试集样本的数量
N=size(t_test,2);
%%
% 数据归一化处理
[P_train,ps_input]=mapminmax(p_train);
P_test=mapminmax('apply',p_test,ps_input);
T_train=t_train;
T_test=t_test;

%% 创建ELM神经网络
[IW,B,LW,TF,TYPE]=ELMtrain(P_train,T_train,20,'sig',1);  %这里需要编程寻找最佳的隐含层神经元的个数
%也可以对比一下这里激活函数的区别
T_sim=ELMpredict(P_test,IW,B,LW,TF,TYPE);

%%
%结果对比
result=[T_test',T_sim'];

%%
%计算正确率
accuracy=length(find(T_test==T_sim))/length(T_test);

%% 画图
figure(1);
plot(1:N,T_test,'r-*',1:N,T_sim,'b-o');
legend('真实值','ELM预测值')
xlabel('测试集样本编号')
ylabel('预测值')
string={'测试集预测效果对比',['accuracy= ',num2str(accuracy*100),'%']};
title(string)

