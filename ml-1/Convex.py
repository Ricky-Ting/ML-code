import numpy as np

def x4(para_list):
    ret = 0;
    for i in para_list:
        ret += i*i*i*i
    return ret

def x3(para_list):
    ret = 0;
    for i in para_list:
        ret += i*i*i
    return ret

def x2(para_list):
    ret = 0;
    for i in para_list:
        ret += i*i
    return ret

def x1(para_list):
    ret = 0;
    for i in para_list:
        ret += i
    return ret

def x0(para_list):
    return len(para_list)


def judge(para_list):
    A = x4(para_list)
    B = x3(para_list)
    C = x2(para_list)
    D = x1(para_list)
    E = x0(para_list)
    if A <= 0:
        return 0
    if A*C - B*B <= 0:
        return 0
    if A*C*E + 2*B*C*D -C*C*C - B*B*E - A*D*D <= 0:
        return 0
    return 1

ssize = int(input())
print("ssize=",ssize)
x = np.random.randint(20, size=ssize) 
while judge(x) == 0:
    x = np.random.randint(20, size=ssize)
print(x)
print("A=",x4(x))
print("B=",x3(x))
print("C=",x2(x))
print("D=",x1(x))
print("E=",x0(x))






