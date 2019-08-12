import math
import numpy as np
import matplotlib.pyplot as plt

def ent(p):
    return  - p * math.log(p,2)

def cal(cnt, one, tot):
    if(cnt ==0):
        return 0
    if(one ==0 or one==cnt):
        return 0 
    return (cnt/tot) * ( ent(one/cnt) + ent(1 - one/cnt)  )

def eval(mid, x, y):
    left_cnt = 0
    right_cnt = 0
    left_one = 0; right_one = 0;
    ret = 0
    for i in range(len(x)):
        if(x[i]<=mid):
            left_cnt += 1
            if(y[i]==1):
                left_one += 1
        else:
            right_cnt += 1
            if(y[i]==1):
                right_one += 1
    return cal(left_cnt, left_one, left_cnt + right_cnt) + cal(right_cnt, right_one, left_cnt + right_cnt)
        

x1 = [24, 53, 23, 25, 32, 52, 22, 43, 52, 48]
y = [1, 0, 0, 1, 1, 1, 1, 0, 0, 1]

x2 = [40, 52, 25, 77, 48, 110, 38, 44, 27, 65]

#x1 = [53, 52, 43, 48]
#x2 = [52, 110, 44, 65]
#y = [0, 1, 0, 1]

#x1.pop(3-1); x2.pop(3-1); y.pop(3-1); 
#x1.pop( 9 - 1 - 1); x2.pop(9-1 - 1); y.pop(9-1-1); 

s1 = x1.copy() 
s2 = x2.copy() 

#print(s2)
cnt = 0
for i in range(len(y)):
    if(y[i]==1):
        cnt+= 1;
print( cal(len(y), cnt, len(y))  )

s1.sort();
s2.sort();
for i in range(len(s2)-1):
    mid =  (s2[i]+s2[i+1])/2  
    print(mid," :", eval(mid, x2, y))

for i in range(len(y)):
    if(y[i]==0):
        plt.scatter( x1[i], x2[i], color='k')
    else:
        plt.scatter( x1[i], x2[i], color ='r' )


x = np.linspace(0,100)
y1 = 1.25*x - 1
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.text(80,80,"$1.25x_1 - x_2 - 1 = 0$")

plt.plot(x,y1)

plt.show()
