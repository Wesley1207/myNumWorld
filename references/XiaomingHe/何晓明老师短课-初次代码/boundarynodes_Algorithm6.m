function [A,B]=boundarynodes_Algorithm6(N,A,B)
%使用何老师更新之后的算法6来引入约束
%boundarynodes，约束节点列表
%1:表示狄利克雷边界条件
%2:纽曼边界条件
%3：罗宾边界条件
%张嘉林
boundarynodes=[1,1;1,N+1;0,cos(1)];
nbn=size(boundarynodes,2);
for k=1:nbn
if boundarynodes(1,k)==1
    i=boundarynodes(2,k);
    A(i,:)=0;
    A(i,i)=1;
    B(i)=boundarynodes(3,k);
end
end
end