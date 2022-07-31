# 宝贵的经验

## 程序

当你想交换数组的值的时候, 比如想把 `u` 给 `u_n`, `u_n` 给 `u_nm1`, 永远使用:
```
u_nm1[:] = u_n
u_n[:] = u
```
而不是:
```
u_nm1 = u_n
u_n = u
```
原因是, for the method1, `u_n[:]=u`, which means that without creating a new reference, we overwrite `u_n` with `u`'s value.

for the method2, 'u_n=u', the `u_n` and `u` are different refs to the same data, so when data is changed by `u` the `u_n` will change also. This is very bad when we swap arrays during time advancing !!!
