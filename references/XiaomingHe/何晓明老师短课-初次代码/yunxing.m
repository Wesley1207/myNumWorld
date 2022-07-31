%运算脚本文件
clear all;
clc;
a=0;b=1;N=128;element_order=1;r=1;s=1;
[P,T,Pb,Tb,element_order]=information_input(a,b,N,element_order);
[A]=stiffness_Algorithm4(Pb,Tb,r,s,element_order,N);
 %A_full=full(A)  %全矩阵显示
%A_full_1=full(A);
 [B]=force_Algorithm5(Pb,Tb,r,s,element_order,N);
 B2=B
 [A,B]=boundarynodes_Algorithm6(N,A,B);
 % A_full=full(A);
 u_h=A\B;
u_exat=Pb.*cos(Pb)
u_exat1=reshape(u_exat,size(u_exat,2),1);
A_error=max(abs(u_exat1-u_h))
 plot(Pb,u_h)