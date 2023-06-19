import numpy as np
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman',size=16)
x=[1,2,3]
isbar=True
name="pics/part_line"
if (isbar):
    name="pics/part"


def plot_sub(row,col,index,x,bsim,metis,mgmetis,lim,title,bar_width):
    # plt.subplot(row,col,index)
    plt.figure(figsize=(4.8,2.8))
    plt.grid(True, linestyle="--", alpha=0.5)
    if(isbar):
        xl=[i-bar_width for i in x]
        xr=[i+bar_width for i in x]
        plt.bar(xl,bsim,label="BSim-partition",width=bar_width,color="tab:blue")
        plt.bar(x,metis,label="Metis-based",width=bar_width,color="tab:green")
        plt.bar(xr,mgmetis,label="Ours",width=bar_width,color="tab:red")
    else:
        plt.plot(x,bsim,label="Bsim")
        plt.plot(x,metis,label="Metis")
        plt.plot(x,mgmetis,label="Ours")
        
    plt.xlabel("GPU Count")
    plt.ylabel("Simulation Time (s)")
    plt.ylim(lim)
    # plt.title(title,y=0,pad=-70)
    plt.legend(loc=2,prop={"family": "Times New Roman", "size": 14})
    plt.xticks(x,[2,4,8])
    plt.savefig(name+str(index)+".pdf", bbox_inches='tight', pad_inches=0.02)



bsim=[2.35938,1.92659 ,3.35832 ]
metis=[2.38392 ,1.95817 ,4.36687]
mgmetis=[1.98093,1.70026 ,2.776 ]
plot_sub(1,3,1,x,bsim,metis,mgmetis,[1,5],"(a) Synapse count: 0.2 B",0.2)#Synapse Count: 200m



bsim=[4.34812 , 3.02702 , 4.24365]
metis=[4.44759 , 3.22605 , 5.05481]
mgmetis=[3.32015 , 2.63636 , 3.60911]
plot_sub(1,3,2,x,bsim,metis,mgmetis,[2,7],"(b) Synapse count: 0.4 B",0.2)#Synapse Count: 400m



bsim=[6.55828 , 4.42186 , 5.17615 ]
metis=[6.68437 , 4.88608 , 5.38814]
mgmetis=[5.45235 , 3.76284 , 4.80339 ]
plot_sub(1,3,3,x,bsim,metis,mgmetis,[3,10],"(c) Synapse count: 0.6 B",0.2)#Synapse Count: 600m

