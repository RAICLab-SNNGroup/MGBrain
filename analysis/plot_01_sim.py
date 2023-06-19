import numpy as np
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman',size=16)
bar_width=50000000
x=[250000000,500000000,750000000,1000000000]
isbar=True

name="pics/sim_line"
if (isbar):
    name="pics/sim"


def plot_sub(row,col,index,x,bsim,spice,mgbrain,lim,title,bar_width):
    # plt.subplot(row,col,index)
    plt.figure(figsize=(4,3.6))
    plt.grid(True, linestyle="--", alpha=0.5)
    if(isbar):
        xl=[i-bar_width for i in x]
        xr=[i+bar_width for i in x]
        plt.bar(xl,bsim,label="BSim",width=bar_width,color="tab:blue")
        plt.bar(x,spice,label="Spice",width=bar_width,color="tab:green")
        plt.bar(xr,mgbrain,label="Ours",width=bar_width,color="tab:red")
    else:
        plt.plot(x,bsim,label="BSim")
        plt.plot(x,spice,label="Spice")
        plt.plot(x,mgbrain,label="Ours")
        
    plt.xlabel("Synapse Count")
    if(index==4):
        plt.ylabel("Simulation Time (s)",labelpad=-4)
    else:
        plt.ylabel("Simulation Time (s)")
    plt.ylim(lim)
    # plt.title(title,y=0,pad=-70)
    plt.legend(loc=2,prop={"family": "Times New Roman", "size": 14})
    plt.xticks(x)
    plt.savefig(name+str(index)+".pdf", bbox_inches='tight', pad_inches=0.02)

# plt.figure(figsize=(12,8))

bsim=[7.832509,9.976230,12.164757,14.787769 ]
spice=[1.7407 ,2.61139 ,3.16246,3.44184]
mgbrain=[3.72308,5.44128 ,7.19702,9.40656 ]
plot_sub(2,2,1,x,bsim,spice,mgbrain,[1,20],"(a) FR: 150 Hz",bar_width)#Firing Rate: 150hz


bsim=[16.259898,23.500075 ,28.219058,37.198510 ]
spice=[3.25265 ,5.72406 ,6.7539,7.82384]
mgbrain=[4.17827,6.07658 ,6.9291,9.85139 ]
plot_sub(2,2,2,x,bsim,spice,mgbrain,[1,45],"(b) FR: 450 Hz",bar_width)#Firing Rate: 450hz



bsim=[26.237032,40.100013,46.439995,54.066623 ]
spice=[4.92061 ,8.62344 ,10.5031,12.1544]
mgbrain=[4.02548,5.77166 ,7.15584,8.66841 ]
plot_sub(2,2,3,x,bsim,spice,mgbrain,[1,75],"(c) FR: 700 Hz",bar_width)#Firing Rate: 700hz


bsim=[38.747017,65.889814 ,72.371618,81.845419 ]
spice=[7.61888 ,14.2096 ,17.2903,20.2191]
mgbrain=[4.7942,7.61369 ,8.62243,13.3799 ]
plot_sub(2,2,4,x,bsim,spice,mgbrain,[1,120],"(d) FR: 1200 Hz",bar_width)#Firing Rate: 1200hz

# plt.tight_layout()
