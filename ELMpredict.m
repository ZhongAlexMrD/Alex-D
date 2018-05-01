function Y= ELMpredict(P,IW,B,LW,TF,TYPE)
%ELMPREDICT 此处显示有关此函数的摘要
%     参数说明
%  P  表示输入样本矩阵，R×Q维
%  T  表示输出样本矩阵，S×Q维
%  N  表示中间隐含层的神经元个数，默认为Q
%  TF 表示输入层和隐含层之间的传递函数，可选值如下：
%        ‘sig’，表示sigmoid函数，默认为‘sig’
%        ‘sin’，表示正弦函数
%        ‘hardlim’，表示hardlim 函数
%  TYPE 表示类型，0表示回归，1表示分类。默认为0
%%
if nargin<6
    error('ELM网络输入的参数不足')
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
        tempY(index,i)=1;      %这里有点计算概率的意思，选择概率最大的位置预测为1,其他位置预测为0
%         此处也可以用compet函数来代替
%         index=compet(Y(:,i));
%         tempY(logical(index),i)=1
    end
    Y=vec2ind(tempY);
end
end

