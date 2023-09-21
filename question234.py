'''---------------------------------------以下为第二，三，四问----------------------------------------'''
import numpy as np
from numpy import sin,cos,tan,arctan,pi
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fsolve


plt.rc('font',size=16,family='SimHei')
plt.rcParams['axes.unicode_minus'] = False 
'''---------------------------------------第二问----------------------------------------'''
#角度
theta=120*np.pi/180
alpha=1.5*np.pi/180
beta=np.array([0,45,90,135,180,225,270,315])*pi/180
gama=arctan(tan(alpha)*sin(np.abs(pi-beta)))

#1海里=1852米
nm=1852

#距离中心点的距离，单位为米
distance=np.array([0,0.3,0.6,0.9,1.2,1.5,1.8,2.1])
distance=distance*nm

#海水深度
D=np.zeros((8,8))
for i in range(8):
    D[i,:]=120+distance*tan(alpha)*cos(beta[i])

#覆盖宽度
w1=np.zeros((8,8))
for i in range(8):
    dL_left=D[i,:]*cos(gama[i])*sin(theta/2)/cos(theta/2+gama[i])
    dL_right=D[i,:]*cos(gama[i])*sin(theta/2)/cos(theta/2-gama[i])

    w1[i,:]=dL_left+dL_right

#保存到excel文件中
f1=pd.read_excel('result2.xlsx')
f1.iloc[1:9,2:10]=w1
r1=pd.ExcelWriter('result2.xlsx')
f1.to_excel(r1,'Sheet1',index=False)
r1.save()

#整理输出数据
w1=pd.DataFrame(w1)
print('\n--------------------------第二问----------------------------\n')
print('覆盖宽度为:\n',w1)

'''---------------------------------------第三问----------------------------------------'''
print('\n--------------------------第三问----------------------------\n')
#中心点深度
dd=110/nm

#计算测线总长度
D=np.zeros((1,100))
D[0,0]=(dd*cos(alpha)*sin(theta/2)-2*cos(alpha+theta/2))/(sin(alpha)*sin(theta/2)+cos(alpha+theta/2))
i=0
while D[0,i] <= (-dd*cos(alpha)*sin(theta/2)+2*cos(alpha-theta/2))/(cos(alpha-theta/2)-sin(alpha)*sin(theta/2)):
    D[0,i+1]=((cos(alpha-theta/2)*cos(alpha+theta/2)-0.9*cos(alpha-theta/2)*sin(alpha)*sin(theta/2))*D[0,i]+0.9*dd*(cos(alpha-theta/2)+cos(alpha+theta/2))*cos(alpha)*sin(theta/2))/(cos(alpha-theta/2)*cos(alpha+theta/2)+0.9*cos(alpha+theta/2)*sin(alpha)*sin(theta/2))
    i=i+1
D=np.array(D[D!=0])
D_end_right=D[-1]+(dd-D[-1]*tan(alpha))*cos(alpha)*sin(theta/2)/cos(theta/2-alpha)

yy=[[-1]*len(D),[1]*len(D)]
xx=[D,D]
print(pd.DataFrame(D))

#验证是否还有缺漏
if D_end_right<2:
    print("有缺漏,补线:")
    length=2*len(D)+2
else:
    print("无缺漏,不补线")
    length=2*len(D)
print(length) 

#画图
fig=plt.figure(figsize=(10,8),dpi=80)
plt.plot(xx,yy)
plt.xlabel("由西到东/海里")
plt.ylabel("由南到北/海里")
plt.title("测线分布图")
plt.savefig("./测线分布图.png")
plt.show()

#证明条件1
beta=np.arange(0.1,90,0.1)*pi/180
gama=arctan(tan(alpha)*sin(beta))
n=2/(dd*cos(beta)*(cos(gama)*sin(theta/2)/cos(theta/2-gama)+cos(gama)*sin(theta/2)/cos(theta/2+gama)))
for i in range(len(n)-1):
    if n[i]<=n[i+1]:
        count=1
    else:
        count=0
        break
if count==1:
    print("单调递增")
else:
    print("非单调递增")

#证明条件2
w_depth=(dd+2*tan(alpha))*tan(theta/2)*2
n1=2/(w_depth*0.8)
w_undepth=(dd-2*tan(alpha))*tan(theta/2)*2
n2=2/(w_undepth*0.8)
if n1<=n2:
    print("验证成功")
else:
    print("验证失败")

#验证扇形区域
a=np.zeros((1,100))
b=np.zeros((1,100))
a[0,0]=5.1938*pi/180
b[0,0]=0
h=0
i=0
while b[0,i]<=75*pi/180:
    b[0,i+1]=b[0,i]+0.9*a[0,i]
    a[0,i+1]=13.007*sin(pi/3)*cos(arctan(tan(alpha)*sin(b[0,i])))*(1/cos(theta/2+arctan(tan(alpha)*sin(b[0,i])))+1/cos(theta/2-arctan(tan(alpha)*sin(b[0,i])))) / (496.74/cos(b[0,i]))
    if b[0,i] <13.95*pi/180:
        h=h+7480/cos(b[0,i])
    else:
        h=h+1852/sin(b[0,i])-496.03/cos(b[0,i])
    i=i+1
print(h*2/nm-4)

'''---------------------------------------第四问----------------------------------------'''
print('\n--------------------------第四问----------------------------\n')
fujian=pd.read_excel('附件.xlsx')
fujian=pd.DataFrame(fujian)

deepth=np.array([fujian.iloc[1:,2:]])
deepth=deepth.reshape((251,201))
x=np.arange(0,4.02,0.02) #200
y=np.arange(0,5.02,0.02) #250
X,Y=np.meshgrid(x,y)

#创建图窗
fig=plt.figure(figsize=(10,8),dpi=80)
axs=fig.add_subplot(projection='3d')
#绘制海底地形图
surf=axs.plot_surface(X,Y,-deepth,cmap='viridis')
plt.colorbar(surf)
axs.plot_wireframe(X, Y,-deepth, rstride=30, cstride=30, linewidth=0.5, color='orangered')
axs.get_proj = lambda: np.dot(Axes3D.get_proj(axs), np.diag([1, 1, 1.2, 1]))
axs.set_xlabel('由西向东/海里')
axs.set_ylabel('由南向北/海里')
axs.set_title('海底地形图')
plt.savefig("./海底地形图.png")
plt.show()

plt.figure(figsize=(10,8),dpi=80)
#填充颜色
c=plt.contourf(X,Y,deepth)
plt.colorbar(c)
#画等高线
contour=plt.contour(X,Y,deepth,colors='k')
plt.clabel(contour,fontsize=16,colors='k')
plt.xlabel("由西向东/海里")
plt.ylabel("由南向北/海里")
plt.title("海底等深线图")
plt.savefig("./海底等深线图.png")
plt.show()

#计算梯度
step=np.gradient(-deepth)
col=step[0]
row=step[1]

X=np.array(X)
Y=np.array(Y)
deepth=deepth/nm

#计算下半部分的坡面
col_1=np.sum(col[0:101,-1])
row_1=np.sum(row[0,100:202])
beta4_1=np.abs(arctan(col_1/row_1))
h1=(deepth[0,200]-deepth[0,0]);l1=np.sqrt(16+h1**2)
h2=(deepth[0,200]-deepth[250,200]);l2=np.sqrt(25+h2**2)
h3=(deepth[250,200]-deepth[0,0]);l3=np.sqrt(16+25+h3**2)
S=0.5*l1*l2*sin(np.arccos((l1**2+l2**2-l3**2)/(2*l1*l2)))
alpha1=np.arccos(10/S)
dd=deepth[0,0]+4*cos(beta4_1)*tan(alpha1) 
Length_1=0
omiga=arctan(5/4)
D=np.zeros((1,100))
D[0,0]=(dd*cos(alpha1)*sin(theta/2))/(cos(alpha1+theta/2)+sin(alpha1)*sin(theta/2))
i=0
while D[0,i] <= (4*cos(beta4_1)*cos(alpha1-theta/2)-dd*cos(alpha1)*sin(theta/2))/(cos(alpha1-theta/2)-sin(alpha1)*sin(theta/2)):
    D[0,i+1]=(0.9*dd*(cos(alpha1+theta/2)+cos(alpha1-theta/2))*cos(alpha1)*sin(theta/2)+(cos(alpha1+theta/2)*cos(alpha1-theta/2)-0.9*cos(alpha1+theta/2)*sin(theta/2)*sin(alpha1))*D[0,i])/(cos(alpha1+theta/2)*cos(alpha1-theta/2)+0.9*cos(alpha1-theta/2)*sin(theta/2)*sin(alpha1))
    if D[0,i]/sin(beta4_1)<=5:
        Length_1=Length_1+D[0,i]/(cos(beta4_1)*sin(beta4_1))
    else:
        Length_1=((4*cos(beta4_1)-D[0,i])*sin(omiga))/(cos(beta4_1)*sin(pi/2-beta4_1-omiga))+Length_1
    i=i+1
D=D[D!=0]
print(D[D!=0])
print(Length_1)

D_=pd.DataFrame(D)
r1=pd.ExcelWriter('第四问_深处.xlsx')
D_.to_excel(r1,'Sheet1',index=False)
r1.save()

k1=tan(pi/2-beta4_1)
x=np.arange(0,4.1,0.1)
y1=k1*(x-4)
for i in range(len(D)-1):
    plt.plot(x,y1+D[i]/sin(beta4_1))
plt.savefig("./图1.png")
plt.show()

#计算上半部分的坡面
col_2=np.sum(col[150:252,0])
row_2=np.sum(row[-1,0:101])
beta4_2=np.abs(arctan(col_2/row_2))
h4=(deepth[250,0]-deepth[0,0]);l4=np.sqrt(25+h4**2)
h5=(deepth[250,0]-deepth[250,200]);l5=np.sqrt(16+h5**2)
S=0.5*l4*l5*sin(np.arccos((l4**2+l5**2-l3**2)/(2*l4*l5)))
alpha2=np.arccos(10/S)

dd1=deepth[0,0]+5*cos(pi/2-beta4_2)*tan(alpha2) # 原点水深
u=arctan(4/5)
Length_2=0
D1=np.zeros((1,200))
D1[0,0]=(dd1*cos(alpha2)*sin(theta/2))/(cos(alpha2+theta/2)+sin(alpha2)*sin(theta/2))
i=0
while D1[0,i] <= (5*cos(pi/2-beta4_2)*cos(alpha2-theta/2)-dd1*cos(alpha2)*sin(theta/2))/(cos(alpha2-theta/2)-sin(alpha2)*sin(theta/2)):
    D1[0,i+1]=(0.9*dd1*(cos(alpha2+theta/2)+cos(alpha2-theta/2))*cos(alpha2)*sin(theta/2)+(cos(alpha2+theta/2)*cos(alpha2-theta/2)-0.9*cos(alpha2+theta/2)*sin(theta/2)*sin(alpha2))*D1[0,i])/(cos(alpha2+theta/2)*cos(alpha2-theta/2)+0.9*cos(alpha2-theta/2)*sin(theta/2)*sin(alpha2))
    if D1[0,i]/sin(pi/2-beta4_2)<=4:
        Length_2=Length_2+D1[0,i]/(cos(pi/2-beta4_2)*sin(pi/2-beta4_2))
    else:
        Length_2=((5*cos(pi/2-beta4_2)-D1[0,i])*sin(u))/(cos(pi/2-beta4_2)*sin(beta4_2-u))+Length_2
    i=i+1

D1=D1[D1!=0]
print(D1)
print(Length_2)

D1_=pd.DataFrame(D1)
r1=pd.ExcelWriter('第四问_浅处.xlsx')
D1_.to_excel(r1,'Sheet1',index=False)
r1.save()

k2=tan(pi/2-beta4_2)
x=np.arange(0,4.1,0.1)
y2=k2*x+5
for i in range(len(D1)-1):
    plt.plot(x,y2-D1[i]/sin(beta4_2))
plt.savefig("./图2.png")
plt.show()

x__=(D1[0]/sin(beta4_2)-5)/(k2-5/4)
x___=(1+5/4*k2)*x__+5*k2
d__=np.abs(5*x___-20)/np.sqrt(25+16)
s__=d__*np.sqrt(25+16)

yita=s__/20
l=yita*(Length_1+Length_2)
print(l)






































