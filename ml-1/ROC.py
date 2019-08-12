import matplotlib.pyplot as plt

def TP(para_list, cut_point):
    counter=0
    for i in range(0,cut_point):
        (r,s) = para_list[i];
        if s==1:
            counter=counter+1
    return counter

def FP(para_list, cut_point):
    return cut_point - TP(para_list, cut_point)
    
def FN(para_list, cut_point):
    counter=0;
    for i in range(cut_point, len(para_list)):
        (r,s) = para_list[i]
        if s==1:
            counter=counter+1;
    return counter

def TN(para_list, cut_point):
    return len(para_list) - cut_point - FN(para_list, cut_point)   

def TPR(para_list, cut_point):
    return (TP(para_list, cut_point) ) / (TP(para_list, cut_point) + FN(para_list, cut_point))

def FPR(para_list, cut_point):
    return (FP(para_list, cut_point)) / (TN(para_list, cut_point) + FP(para_list, cut_point))

def TPR_FPR(para_list, cut_point):
    return (FPR(para_list, cut_point), TPR(para_list, cut_point))


predicted=[(0.6,1), (0.51,0), (0.5,1), (0.4,1), (0.33,0), (0.3,0), (0.22,1), (0.2,0)  ]
predicted2=[(0.8,0), (0.68,1), (0.53,0),(0.4,1), (0.22,1), (0.11,0), (0.1,0), (0.04, 1)]
print(predicted2)
#print(FP(predicted2, 3))
points = []
for cut_point in range(0,len(predicted)+1):
   points.append((TPR_FPR(predicted2,cut_point))) 
print(points)

AUC =0
for i in range(0, len(points)-1):
    (x1,y1) = points[i]
    (x2,y2) = points[i+1]
    AUC = AUC + 0.5 * (x2-x1)*(y1+y2)
print("The AUC is", AUC)



plt.axis([0,1,0,1])
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC of y$C_2$")
plt.text(0.6,0.4,"AUC of y$C_2=$"+str(AUC))
for i in range(0, len(points)):
    (x,y) = points[i] 
    plt.scatter(x,y,color='k')
for i in range(0, len(points)-1):
    (x1,y1) = points[i]
    (x2, y2) = points[i+1]
    plt.plot((x1,x2), (y1,y2),color='k')

plt.show()
