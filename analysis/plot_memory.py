import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman',size=16)
x=[10000,20000,30000,40000,50000,60000,70000,80000,90000]
# xlabels=["","20k","","40k","","60k","","80k",""]

def read_data(filename):
    file=open(filename,'r')
    data=file.read()
    lines=data.split("\n")
    arr=[]
    for line in lines:
        if line=="":continue
        lined=[]
        for d in line.split(","):
            lined.append(int(d))
        arr.append(lined)
    return arr


def plot_sub(index,x,const,variable,title):
    plt.subplot(130+index+1)
    plt.plot(x,const,label="const")
    plt.plot(x,variable,label="variable")
    plt.ylabel("Simulation Time (ms)")
    plt.xlabel("Simulation Step Count")
    plt.title(title,pad=1.0,y=-0.36)
    plt.xticks(x)
    plt.ticklabel_format(axis="x",style="sci",scilimits=(0,0))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.legend()



data=read_data("benchdata/memory.txt")
plt.figure(figsize=(12,4))
plot_sub(0,x,data[0],data[1],"(a) Neuron count:300")#Neuron Count: 300
plot_sub(1,x,data[2],data[3],"(b) Neuron count:600")#Neuron Count: 600
# plot_sub(2,x,data[4],data[5],"(c) Neuron count:900")#Neuron Count: 900
plot_sub(2,x,data[6],data[7],"(c) Neuron count:1500")#Neuron Count: 1500
plt.savefig("picture/memory.pdf")