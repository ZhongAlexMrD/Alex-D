%% 汽油辛烷值的预测
%%
clear all;
clc;
%% 
% 加载数据集
load spectra_data.mat

%%
%测试集、训练集的划分
temp=randperm(size(NIR,1));
p_train=NIR(temp(1:50),:)';
t_train=octane(temp(1:50),:)';
p_test=NIR(temp(51:end),:)';
t_test=octane(temp(51:end),:)';
%% 
% 记录测试集样本的数量
N=size(t_test,2);
%%
% 数据归一化处理
[P_train,ps_input]=mapminmax(p_train);
P_test=mapminmax('apply',p_test,ps_input);
[T_train,ps_output]=mapminmax(t_train);
T_test=t_test;
%% 创建ELM神经网络
[IW,B,LW,TF,TYPE]=ELMtrain(P_train,T_train,30,'sig',0);  %这里需要编程寻找最佳的隐含层神经元的个数
%也可以对比一下这里激活函数的区别
t_sim=ELMpredict(P_test,IW,B,LW,TF,TYPE);

%% 数据反归一化
T_sim=mapminmax('reverse',t_sim,ps_output);

%%
%结果对比
result=[T_test',T_sim'];

%%
%计算均方误差
E=mse(T_sim-T_test);

%%
%计算决定系数
R2=(N*sum(T_test.*T_sim)-sum(T_test)*sum(T_sim))^2/((N*sum(T_sim.^2)-sum(T_sim)^2)*(N*sum(T_test.^2)-sum(T_test)^2));
error=abs(T_test-T_sim)./T_test;
%% 画图
figure(1);
plot(1:N,T_test,'r-*',1:N,T_sim,'b-o');
legend('真实值','ELM预测值')
xlabel('测试集样本编号')
ylabel('预测值')
string={'测试集预测效果对比',['R^2= ',num2str(R2)]};
title(string)

figure(2);
plot(1:N,error*100,'b-o');
xlabel('测试集样本编号')
ylabel('相对误差%')
string={'测试集预测相对误差'};
title(string)