import numpy as np
from numpy import sin,cos,tan
import pandas as pd

theta=120*np.pi/180
alpha=1.5*np.pi/180

#测线距中心点距离
distance_to_center=np.array([-800,-600,-400,-200,0,200,400,600,800]) # 离中心点的距离

#测线间隔距离
dd=200

#海水深度
D=70-distance_to_center*tan(alpha)

d_L1=(cos(alpha)*sin(theta/2)/cos(theta/2+alpha))*D
d_L2=(cos(alpha)*sin(theta/2)/cos(theta/2-alpha))*D # 两部分的宽度

#覆盖宽度
w=d_L1+d_L2

#重合率
yita=1-dd/(d_L2[:-1]+d_L1[1:])

print('海水深度为:',D)
print('覆盖宽度为:',w)
print('重叠率为:',yita*100)

#保存到excel文件中
f=pd.read_excel('result1.xlsx')
f.iloc[0,1:]=D
f.iloc[1,1:]=w
f.iloc[2,2:]=yita*100
r=pd.ExcelWriter('result1.xlsx')
f.to_excel(r,'Sheet1',index=False)
r.save()







