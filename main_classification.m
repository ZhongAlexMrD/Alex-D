%% iris���ݼ�ʶ��
%%
clear all;
clc;
%% 
% �������ݼ�
load iris_data.mat

%%
%���Լ���ѵ�����Ļ���
%% ע������������ݼ��Ļ��ַ�ʽ
p_train=[];
t_train=[];
p_test=[];
t_test=[];
n=randperm(size(classes,1));
for i=1:3  %for ѭ���ֶλ��֣�Ȼ����ƴ�ӵ�һ��
    temp_input=features((i-1)*50+1:i*50,:);
    temp_output=classes((i-1)*50+1:i*50,:);
    n=randperm(50);
    % ѵ����--120������
    p_train=[p_train,temp_input(n(1:40),:)'];
    t_train=[t_train,temp_output(n(1:40),:)'];
    % ���Լ�--30������
    p_test=[p_test,temp_input(n(41:50),:)'];
    t_test=[t_test,temp_output(n(41:50),:)'];
    %��ϸ���һ����һ�ι��ڶ��ǩ��������ݼ��Ļ���
end


%% 
% ��¼���Լ�����������
N=size(t_test,2);
%%
% ���ݹ�һ������
[P_train,ps_input]=mapminmax(p_train);
P_test=mapminmax('apply',p_test,ps_input);
T_train=t_train;
T_test=t_test;

%% ����ELM������
[IW,B,LW,TF,TYPE]=ELMtrain(P_train,T_train,20,'sig',1);  %������Ҫ���Ѱ����ѵ���������Ԫ�ĸ���
%Ҳ���ԶԱ�һ�����Ｄ���������
T_sim=ELMpredict(P_test,IW,B,LW,TF,TYPE);

%%
%����Ա�
result=[T_test',T_sim'];

%%
%������ȷ��
accuracy=length(find(T_test==T_sim))/length(T_test);

%% ��ͼ
figure(1);
plot(1:N,T_test,'r-*',1:N,T_sim,'b-o');
legend('��ʵֵ','ELMԤ��ֵ')
xlabel('���Լ��������')
ylabel('Ԥ��ֵ')
string={'���Լ�Ԥ��Ч���Ա�',['accuracy= ',num2str(accuracy*100),'%']};
title(string)

