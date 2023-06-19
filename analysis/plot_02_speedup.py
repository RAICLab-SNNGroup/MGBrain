import numpy as np
import matplotlib.pyplot as plt
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
    return data

def trans(data):
    mgbrain_speedup2=np.divide( np.array(data[0]),np.array(data[1]))
    mgbrain_speedup4=np.divide( np.array(data[0]),np.array(data[2]))
    mgbrain_speedup8=np.divide( np.array(data[0]),np.array(data[3]))
    bsim_speedup2=np.divide( np.array(data[4]),np.array(data[5]))
    bsim_speedup4=np.divide( np.array(data[4]),np.array(data[6]))
    bsim_speedup8=np.divide( np.array(data[4]),np.array(data[7]))
    spice_speedup2=np.divide( np.array(data[8]),np.array(data[9]))
    spice_speedup4=np.divide( np.array(data[8]),np.array(data[10]))
    spice_speedup8=np.divide( np.array(data[8]),np.array(data[11]))
    mgbrain=[]
    bsim=[]
    spice=[]
    
    for i in range(0,4):
        mgbrain.append([mgbrain_speedup2[i],mgbrain_speedup4[i],mgbrain_speedup8[i]])
        bsim.append([bsim_speedup2[i],bsim_speedup4[i],bsim_speedup8[i]])
        spice.append([spice_speedup2[i],spice_speedup4[i],spice_speedup8[i]])
    data={'mgbrain':mgbrain,'bsim':bsim,'spice':spice}
    return data

x=[1,2,3]
isbar=True
name="pics/speedup_line"
if (isbar):
    name="pics/speedup"
data=trans(read_data("benchdata/speedup.txt"))
print(data['mgbrain'])


# print(data)
def plot_sub(row,col,index,x,mgbrain,bsim,spice,i,lim,title,bar_width):
    # plt.subplot(row,col,index)
    plt.figure(figsize=(4,3.6))
    plt.grid(True, linestyle="--", alpha=0.5)
    if isbar:
        xl=[i-bar_width for i in x]
        xr=[i+bar_width for i in x]
        plt.bar(xl,bsim[i],label="BSim",width=bar_width,color="tab:blue")
        plt.bar(x,spice[i],label="Spice",width=bar_width,color="tab:green")
        plt.bar(xr,mgbrain[i],label="Ours",width=bar_width,color="tab:red")
    else:
        plt.plot(x,bsim[i],label="BSim")
        plt.plot(x,spice[i],label="Spice")
        plt.plot(x,mgbrain[i],label="Ours")
    plt.legend(loc=2,prop={"family": "Times New Roman", "size": 14})
    plt.xlabel("GPU Count")
    plt.xticks(x,["2","4","8"])
    plt.ylabel("Speedup (x)",labelpad=4.0)
    plt.ylim(lim)
    # plt.title(title,y=0,pad=-70)
    plt.savefig(name+str(index)+".pdf", bbox_inches='tight', pad_inches=0.02)
bar_width=0.16
plot_sub(2,2,1,x,data['mgbrain'],data['bsim'],data['spice'],0,[0,4],"(a) FR: 150 Hz",bar_width)#Firing Rate : 150hz
plot_sub(2,2,2,x,data['mgbrain'],data['bsim'],data['spice'],1,[0,4],"(b) FR: 450 Hz",bar_width)#Firing Rate : 450hz
plot_sub(2,2,3,x,data['mgbrain'],data['bsim'],data['spice'],2,[0,8],"(c) FR: 700 Hz",bar_width)#Firing Rate : 700hz
plot_sub(2,2,4,x,data['mgbrain'],data['bsim'],data['spice'],3,[0,8],"(d) FR: 1200 Hz",bar_width)#Firing Rate : 1200hz



# mgbrain
# [1.5180835549870884, 2.0967879369186497, 1.306111418337034]
# [1.9267916821388786, 3.156532398127561, 2.9953962275042936]
# [1.680792491959807, 3.867386720505184, 4.462397146945712]
# [1.8499025775559064, 3.7385600308840563, 6.738784774129799]
