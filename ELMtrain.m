function [IW,B,LW,TF,TYPE] = ELMtrain(P,T,N,TF,TYPE)
%ELMTRAIN �˴���ʾ�йش˺�����ժҪ
%   ����˵��
%  P  ��ʾ������������R��Qά
%  T  ��ʾ�����������S��Qά
%  N  ��ʾ�м����������Ԫ������Ĭ��ΪQ
%  TF ��ʾ������������֮��Ĵ��ݺ�������ѡֵ���£�
%        ��sig������ʾsigmoid������Ĭ��Ϊ��sig��
%        ��sin������ʾ���Һ���
%        ��hardlim������ʾhardlim ����
%  TYPE ��ʾ���ͣ�0��ʾ�ع飬1��ʾ���ࡣĬ��Ϊ0
%%
if nargin<2
    error('ELM��������Ĳ�������');
end

if nargin<3
    N=size(P,2);
end

if nargin<4
    TF='sig';
end

if nargin<5
    TPYE=0;
end
%%
% ע���������ǰ���if�ж����
%%
if size(P,2)~=size(T,2)
    error('ELM����������������������ͬ')
end

[R,Q]=size(P);
if TYPE==1
    T=ind2vec(T);     %�����ת�����������������ʽ�йء�����������ʼǣ�
                     %Ҫע������ļ���
end
[S,Q]=size(T);
% �����������������Ȩֵ
IW=rand(N,R)*2-1;
%�������BiasMatrix
B=rand(N,1);
BiasMatrix=repmat(B,1,Q);
%�����������������H
tempH=IW*P+BiasMatrix;
switch TF
    case 'sig'
        H=1./(1+exp(-tempH));
    case 'sin'
        H=sin(tempH);
    case 'hardlim'
        hardlim(tempH)
end
%�����������������֮�������Ȩֵ
LW=pinv(H')*T';

end

