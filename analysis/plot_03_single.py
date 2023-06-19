import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import matplotlib.ticker as mtick
plt.rc('font',family='Times New Roman',size=16)
def read_data(filename):
    file = open(filename, mode = 'r')
    data=[]
    for line in file.read().split("\n"):
        if line=="":continue
        arrs=[]
        for d in line.split(","):
            arrs.append(float(d))
        data.append(arrs)
    data=np.array(data).T.tolist()
    return data

# data1=read_data("./benchdata/single.txt")
data2=read_data("./benchdata/single3.txt")
# print(data2)
isbar=False
name="pics/single"
if (isbar):
    name="pics/single_bar"

# def plot_sub1(row,col,index,x,bsims,bsimn,mgbrains,mgbrainn,lim,title,bar_width):
#     plt.subplot(row,col,index)
#     if(isbar):
#         xll=[i-bar_width*3/2 for i in x]
#         xl=[i-bar_width/2 for i in x]
#         xr=[i+bar_width/2 for i in x]
#         xrr=[i+bar_width*3/2 for i in x]
#         plt.bar(xll,bsims,label="bsim-synapse",width=bar_width,color="tab:blue")
#         plt.bar(xr,mgbrains,label="ours-synapse",width=bar_width,color="tab:green")
#         plt.bar(xl,bsimn,label="bsim-neuron",width=bar_width,color="tab:cyan")
#         plt.bar(xrr,mgbrainn,label="ours-neuron",width=bar_width,color="tab:olive")
#         plt.grid(True, linestyle="--", alpha=0.5)
#     else:
#         plt.plot(x,bsims,label="bsim-synapse")
#         plt.plot(x,mgbrains,label="ours-synapse")
#         plt.plot(x,bsimn,label="bsim-neuron")
#         plt.plot(x,mgbrainn,label="ours-neuron")
#         plt.grid(True, linestyle="--", alpha=0.5)
#         plt.grid(True, linestyle="--", alpha=0.5)
#     plt.xlabel("Synapse Count")
#     plt.ylabel("Synapse Simulation Time (s)")
#     plt.ylim(lim)
#     plt.title(title)
#     plt.legend(prop={"family": "Times New Roman", "size": 10})
#     plt.xticks(x,["","20m","","40m","","60m","","80m","","100m"])

def plot_sub2(row,col,index,x,bsims,bsimn,mgbrains,mgbrainn,lim,title,bar_width):
    # plt.subplot(row,col,index)
    plt.figure(figsize=(4.8,3.2))
    if(isbar):
        xll=[i-bar_width*3/2 for i in x]
        xl=[i-bar_width/2 for i in x]
        xr=[i+bar_width/2 for i in x]
        xrr=[i+bar_width*3/2 for i in x]
        plt.bar(xl,bsimn,label="BSim-neuron",width=bar_width,color="tab:cyan")
        plt.bar(xll,bsims,label="BSim-propagation",width=bar_width,color="tab:blue")
        plt.bar(xrr,mgbrainn,label="Ours-neuron",width=bar_width,color="tab:orange")
        plt.bar(xr,mgbrains,label="Ours-propagation",width=bar_width,color="tab:red")
        plt.grid(True, linestyle="--", alpha=0.5)
    else:
        plt.plot(x,bsimn,label="BSim-neuron",c="tab:cyan")
        plt.plot(x,bsims,label="BSim-propagation",c="tab:blue")
        plt.plot(x,mgbrainn,label="Ours-neuron",c="tab:orange")
        plt.plot(x,mgbrains,label="Ours-propagation",c="tab:red")
        
        plt.grid(True, linestyle="--", alpha=0.5)
    plt.ticklabel_format(axis="x",style="sci",scilimits=(0,0))
    plt.xlabel("Synapse Count")
    plt.ylabel("Simulation Time (s)",labelpad=4.0)
    plt.ylim(lim)
    # plt.title(title,y=0,pad=-70)
    plt.legend(prop={"family": "Times New Roman", "size": 14},loc=2)
    plt.xticks(x)
    plt.gca().xaxis.set_major_locator(MultipleLocator(40000000))
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    plt.savefig(name+str(index)+".pdf", bbox_inches='tight', pad_inches=0.02)


x=data2[0]
bsims=data2[1]
bsimn=data2[2]
mgbrains=data2[3]
mgbrainn=data2[4]
plot_sub2(1,3,1,x,bsims,bsimn,mgbrains,mgbrainn,[-0.3,10],"(a) FR: 150 Hz",4)#Firing Rate: 150hz

bsims=data2[5]
bsimn=data2[6]
mgbrains=data2[7]
mgbrainn=data2[8]
plot_sub2(1,3,2,x,bsims,bsimn,mgbrains,mgbrainn,[-0.7,20],"(b) FR: 450 Hz",4)#Firing Rate: 450hz

bsims=data2[9]
bsimn=data2[10]
mgbrains=data2[11]
mgbrainn=data2[12]
plot_sub2(1,3,3,x,bsims,bsimn,mgbrains,mgbrainn,[-1,28],"(c) FR: 700 Hz",4)#Firing Rate: 700hz

