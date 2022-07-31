function [P,T,Pb,Tb,element_order]=information_input(a,b,N,element_order)
%information_input(a,b,N)  用于读取坐标和单元信息
%a:左端点坐标
%b:右端点坐标
%N:单元个数
%P:网格节点坐标信息矩阵for trial function
%T:网格单元信息矩阵for trial function
%Pb:有限元节点坐标信息矩阵for test function
%Tb:有限元单元信息矩阵for test function
%根据是否为线性单元判断P、T是否等于Pb、Tb
%element_order:单元阶次
%  张嘉林

%线性单元
if element_order==1
P=sparse(1,N+1);
T=sparse(2,N);
Pb=sparse(1,N+1);
Tb=sparse(2,N);
P=linspace(a,b,N+1);
T=zeros(2,N);
Pb=P;
T(1,:)=1:N;
T(2,:)=2:N+1;
Tb=T;
else    %暂时没有编写
    P=sparse(1,N+1);
    T=sparse(2,N);
    Pb=sparse(1,(N+1)*element_order);
    T=sparse(2,N);
    P=linspace(a,b,N+1);
    T=zeros(2,N);
    Pb=P;
    Tb=T;
    Tb(1,:)=1:N;
    Tb(2,:)=2:N+1;
end
%Draw node label
xx=Pb;
yy=zeros(1,N+1);
plot(xx,yy,'-o','LineWidth',2)
for i=1:size(Pb,2);
    tempstr=['' int2str(i)];
    text(xx(i),yy(i),tempstr,'Color',[1 0 0],'FontSize',14,'HorizontalAlignment','right');
end
end
