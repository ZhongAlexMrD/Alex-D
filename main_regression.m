%% ��������ֵ��Ԥ��
%%
clear all;
clc;
%% 
% �������ݼ�
load spectra_data.mat

%%
%���Լ���ѵ�����Ļ���
temp=randperm(size(NIR,1));
p_train=NIR(temp(1:50),:)';
t_train=octane(temp(1:50),:)';
p_test=NIR(temp(51:end),:)';
t_test=octane(temp(51:end),:)';
%% 
% ��¼���Լ�����������
N=size(t_test,2);
%%
% ���ݹ�һ������
[P_train,ps_input]=mapminmax(p_train);
P_test=mapminmax('apply',p_test,ps_input);
[T_train,ps_output]=mapminmax(t_train);
T_test=t_test;
%% ����ELM������
[IW,B,LW,TF,TYPE]=ELMtrain(P_train,T_train,30,'sig',0);  %������Ҫ���Ѱ����ѵ���������Ԫ�ĸ���
%Ҳ���ԶԱ�һ�����Ｄ���������
t_sim=ELMpredict(P_test,IW,B,LW,TF,TYPE);

%% ���ݷ���һ��
T_sim=mapminmax('reverse',t_sim,ps_output);

%%
%����Ա�
result=[T_test',T_sim'];

%%
%����������
E=mse(T_sim-T_test);

%%
%�������ϵ��
R2=(N*sum(T_test.*T_sim)-sum(T_test)*sum(T_sim))^2/((N*sum(T_sim.^2)-sum(T_sim)^2)*(N*sum(T_test.^2)-sum(T_test)^2));
error=abs(T_test-T_sim)./T_test;
%% ��ͼ
figure(1);
plot(1:N,T_test,'r-*',1:N,T_sim,'b-o');
legend('��ʵֵ','ELMԤ��ֵ')
xlabel('���Լ��������')
ylabel('Ԥ��ֵ')
string={'���Լ�Ԥ��Ч���Ա�',['R^2= ',num2str(R2)]};
title(string)

figure(2);
plot(1:N,error*100,'b-o');
xlabel('���Լ��������')
ylabel('������%')
string={'���Լ�Ԥ��������'};
title(string)