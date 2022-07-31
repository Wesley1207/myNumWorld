function [A]=stiffness_Algorithm4(Pb,Tb,r,s,element_order,N)
%使用何老师更新之后的算法4来形成总体刚度矩阵
%Pb:有限元节点坐标信息矩阵for test function
%Tb:有限元单元信息矩阵for test function
%r:试探函数的求导阶次
%s:测试函数的求导阶次
%element_order:单元阶次
%Pb和Tb由前面的information_input输入
%K为输出的全局刚度矩阵
%张嘉林
if r==1 && s==1    %暂时没有想好怎么写
if element_order==1
    Nb_trial=size(Tb,1);%每个单元节点决定的基函数个数，可以由Tb矩阵行提取
    Nb_test=size(Tb,1); %每个单元节点决定的基函数个数，可以由Tb矩阵行提取
    A=sparse(size(Pb,2),size(Pb,2));
    for i=1:N
        T_current=Tb(:,i);
        phi_n1=-1/(Pb(T_current(2))-Pb(T_current(1)));
        phi_n2=1/(Pb(T_current(2))-Pb(T_current(1)));
        N_shape_matrix_deriv=[phi_n1,phi_n2];
        c_e=exp(Pb(T_current(2)))-exp(Pb(T_current(1)));
        for m=1: Nb_trial
            for n=1: Nb_test
            r=N_shape_matrix_deriv(m)*N_shape_matrix_deriv(n)*c_e;
            A(Tb(m,i),Tb(n,i))=r+A(Tb(m,i),Tb(n,i)); % A(Tb(n,i),Tb(m,i))也可行
            end
        end
    end
end
end 
if element_order==2
    Nb_trial=size(Pb,2);
    Nb_test=size(Pb,2);
    A=sparse(Nb_trial,Nb_test);
    for i=1:N
        T_current=Tb(:,i);
        phi_n1=-1/(Pb(T_current(2))-Pb(T_current(1)));
        phi_n2=1/(Pb(T_current(2))-Pb(T_current(1)));
        N_shape_matrix_deriv=[phi_n1,phi_n2];
        c_e=-(exp(T_current(2))-exp(T_current(1)));
        for m=1: Nb_trial
            for n=1: Nb_test
            r=N_shape_matrix_deriv(Nb_trial)*N_shape_matrix_deriv(Nb_test)*c_e;
            A(Tb(m,i),Tb(n,i))=r
            end
        end
    end
end
end
 