function [B]=force_Algorithm5(Pb,Tb,r,s,element_order,N)
%使用何老师更新之后的算法5来形成右侧载荷矩阵
%Pb:有限元节点坐标信息矩阵for test function
%Tb:有限元单元信息矩阵for test function
%r:试探函数的求导阶次
%s:测试函数的求导阶次
%element_order:单元阶次
%Pb和Tb由前面的information_input输入
%K为输出的全局刚度矩阵
%张嘉林
%syms x;
%v=int(-exp(x)*(cos(x)-2*sin(x)-x*cos(x)-x*sin(x))*x,0,1)
%vpa(v)
%直接不定积分symsx;G=int(-exp(x)*(cos(x)-2*sin(x)-x*cos(x)-x*sin(x))*(1-x))
%H=int(-exp(x)*(cos(x)-2*sin(x)-x*cos(x)-x*sin(x))*(x))
%结果G =-(exp(x)*(2*cos(x) + sin(x) + 2*x^2*sin(x) - x*cos(x) - 3*x*sin(x)))/2
%结果H=(exp(x)*(sin(x) + 2*x^2*sin(x) - x*cos(x) - x*sin(x)))/2
if r==1 && s==1    %暂时没有想好怎么写
if element_order==1
    Nb_force=size(Pb,2);%所有未知节点或者基函数个数
    B=zeros(Nb_force,1);
    for i=1:N
        T_current=Tb(:,i);
        eta1=-0.577350269;           %高斯积分点1
        eta2=0.577350269;            %高斯积分点2
        eta=[eta1,eta2];
        L=size(eta,2);
        w1=1;                     %高斯积分点权重
        w2=1;                     %高斯积分点权重
        w=[w1,w2];
        Nb_test=size(Tb,1); %每个单元节点决定的基函数个数，可以由Tb矩阵行提取
            for n=1: Nb_test
                r=0;%由于在高斯点用到了求和，所以要对r进行初始化，从而可以用于迭代             
             for g=1:L
        x=(Pb(T_current(2))+Pb(T_current(1)))/2+...
            ((Pb(T_current(2))-Pb(T_current(1)))*eta(g))/2; %eta为自然坐标点
        f=-exp(x)*(cos(x)-2*sin(x)-x*cos(x)-x*sin(x))*...
            ((Pb(T_current(2))-Pb(T_current(1)))/2)*w(g);%积分坐标变为[-1,1]，从而可以使用高斯积分
        N_shape1=(Pb(T_current(2))-x)/(Pb(T_current(2))-Pb(T_current(1)));
        N_shape2=(x-Pb(T_current(1)))/(Pb(T_current(2))-Pb(T_current(1)));
        N=[N_shape1,N_shape2];
        r=r+f*N(n);%注意形函数应该要和N_test对应起来才可以
              end
            B(Tb(n,i),1)=r+B(Tb(n,i),1); 
            end
    end
end
end 
end