A = rand(100,2);
A(100,1)=2;
A(100,2)=2;
dist = zeros(100,100);
for i = 1:100
    for j = 1:100
    dist(i,j) = norm(A(i,:)-A(j,:));
    end
end
lof = LOF(dist);
% ��ͼ
x=A;
subplot(2,1,1)
plot(1:size(x,1),x,'-b.','linewidth',2,'markersize',14.5);
legend('ԭʼ����');
title('LOF����Ч��ͼ');
set(legend,'location','best');
subplot(2,1,2)
plot(1:size(lof,2),lof,'-r.','linewidth',2,'markersize',14.5);
legend('LOFֵ');
set(legend,'location','best');