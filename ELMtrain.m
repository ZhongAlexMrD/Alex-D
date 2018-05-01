function [IW,B,LW,TF,TYPE] = ELMtrain(P,T,N,TF,TYPE)
%ELMTRAIN 此处显示有关此函数的摘要
%   参数说明
%  P  表示输入样本矩阵，R×Q维
%  T  表示输出样本矩阵，S×Q维
%  N  表示中间隐含层的神经元个数，默认为Q
%  TF 表示输入层和隐含层之间的传递函数，可选值如下：
%        ‘sig’，表示sigmoid函数，默认为‘sig’
%        ‘sin’，表示正弦函数
%        ‘hardlim’，表示hardlim 函数
%  TYPE 表示类型，0表示回归，1表示分类。默认为0
%%
if nargin<2
    error('ELM网络输入的参数过少');
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
% 注意体会这里前面的if判断语句
%%
if size(P,2)~=size(T,2)
    error('ELM网络的样本矩阵个数必须相同')
end

[R,Q]=size(P);
if TYPE==1
    T=ind2vec(T);     %这里的转换和神经网络多分类的形式有关。（详见吴恩达笔记）
                     %要注意这里的技巧
end
[S,Q]=size(T);
% 随机产生输入层的连接权值
IW=rand(N,R)*2-1;
%随机产生BiasMatrix
B=rand(N,1);
BiasMatrix=repmat(B,1,Q);
%计算隐含层输出矩阵H
tempH=IW*P+BiasMatrix;
switch TF
    case 'sig'
        H=1./(1+exp(-tempH));
    case 'sin'
        H=sin(tempH);
    case 'hardlim'
        hardlim(tempH)
end
%计算隐含层与输出层之间的连接权值
LW=pinv(H')*T';

end

