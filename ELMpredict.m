function Y= ELMpredict(P,IW,B,LW,TF,TYPE)
%ELMPREDICT �˴���ʾ�йش˺�����ժҪ
%     ����˵��
%  P  ��ʾ������������R��Qά
%  T  ��ʾ�����������S��Qά
%  N  ��ʾ�м����������Ԫ������Ĭ��ΪQ
%  TF ��ʾ������������֮��Ĵ��ݺ�������ѡֵ���£�
%        ��sig������ʾsigmoid������Ĭ��Ϊ��sig��
%        ��sin������ʾ���Һ���
%        ��hardlim������ʾhardlim ����
%  TYPE ��ʾ���ͣ�0��ʾ�ع飬1��ʾ���ࡣĬ��Ϊ0
%%
if nargin<6
    error('ELM��������Ĳ�������')
end

Q = size(P,2);
BiasMatrix = repmat(B,1,Q);
tempH = IW * P + BiasMatrix;

switch TF
    case 'sig'
        H=1./(1+exp(-tempH));
    case 'sin'
        H=sin(tempH);
    case 'hardlim'
        H=hardlim(tempH);
end
Y=LW'*H;
if TYPE==1
    tempY=zeros(size(Y));
    for i=1:size(Y,2)
        [~,index]=max(Y(:,i));
        tempY(index,i)=1;      %�����е������ʵ���˼��ѡ���������λ��Ԥ��Ϊ1,����λ��Ԥ��Ϊ0
%         �˴�Ҳ������compet����������
%         index=compet(Y(:,i));
%         tempY(logical(index),i)=1
    end
    Y=vec2ind(tempY);
end
end

