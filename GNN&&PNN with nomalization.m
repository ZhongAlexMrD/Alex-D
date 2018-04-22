clear all
clc
%% �������ݼ�
load BreastTissue_data.mat
% �����ݼ����Կ���һ����106��������9������
% �������ڷ�������
P_train=[];
T_train=[];
P_test=[];
T_test=[];
idx=cell(1,6);
% GRNN��PNN����Ҫ���ݹ�һ��


%% ����ѵ�����Ͳ��Լ�
% ע�Ȿ���ĸ�������������һ��

for i=1:6
    idx(1,i)={find(label==i)};  %�ҵ�ÿ��������Ӧ���±꣬����һ��Ԫ�����飬������滮�����ݼ���ʱ�����
end

for i=1:6
    index=idx{1,i}; %�õ���ǰ�����������������ݼ��е��±꣨������
    len=length(index);
    n=round(len*0.8); %����һ�������������ݼ�
    temp=randperm(len);
    P_train=[P_train,matrix(index(temp(1:n)),:)'];  %ֱ�Ӱ�ѵ����ת���ɾ�����б�ʾ������Ŀ����ʽ��
    T_train=[T_train,label(index(temp(1:n)),:)'];  %Ҳ���Բ���[P_train;matrix(...)]����ʽ
    P_test=[P_test,matrix(index(temp(n+1:end)),:)'];
    T_test=[T_test,label(index(temp(n+1:end)),:)'];
end
%% ��һ������
[P_train,ps_input]=mapminmax(P_train);
P_test=mapminmax('apply',P_test,ps_input);
[T_train,ps_output]=mapminmax(T_train);

%% ����������
result_grnn=[];
result_pnn=[];
for spread=0.01:0.03:1
    net_grnn=newgrnn(P_train,T_train,spread);
    Tc_train=ind2vec(mapminmax('reverse',T_train,ps_output));
    net_pnn=newpnn(P_train,Tc_train,spread);

    %% �������
    T_pred_grnn=sim(net_grnn,P_test);
    Tc_pred_pnn=sim(net_pnn,P_test);
    T_pred_pnn=vec2ind(Tc_pred_pnn);
    %�����һ��
    T_pred_grnn=round(mapminmax('reverse',T_pred_grnn,ps_output));
    %T_pred_pnn=mapminmax('reverse',T_pred_pnn,ps_output);
    result_grnn=[result_grnn,T_pred_grnn'];
    result_pnn=[result_pnn,T_pred_pnn'];
end
%% ����׼ȷ��
accuracy_grnn=[];
accuracy_pnn=[];
for i=1:length(result_grnn)
    accuracy_1=length(find(result_grnn(:,i)==T_test'))/length(T_test);
    accuracy_2=length(find(result_pnn(:,i)==T_test'))/length(T_test);
    accuracy_grnn=[accuracy_grnn,accuracy_1];
    accuracy_pnn=[accuracy_pnn,accuracy_2];
end
%% ������л�ͼ����
figure(1);
N=length(T_test);
plot(1:N,T_test,'bo-',1:N,result_grnn(:,4),'g*-',1:N,result_pnn(:,4),'r-^')
xlabel('Ԥ���������')
ylabel('Ԥ��ֵ')
legend('��ʵֵ','GRNNԤ��ֵ','PNNԤ��ֵ')
string={'���Լ�Ԥ�����Ա�GRNN vs PNN',['��ȷ��',num2str(accuracy_grnn(4)*100) '%(GRNN)vs' num2str(accuracy_pnn(4)*100) '%PNN']};
title(string)
figure(2);
spread=0.01:0.03:1;
plot(spread,accuracy_grnn,'b-o',spread,accuracy_pnn,'r-*');
legend('GRNN','PNN')
xlabel('spread')
ylabel('׼ȷ��')
title('׼ȷ����spread�仯ͼ')