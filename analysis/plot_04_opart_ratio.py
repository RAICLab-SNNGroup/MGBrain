import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
plt.rc('font',family='Times New Roman',size=16)
x=[10000000, 20000000, 30000000, 40000000, 50000000, 60000000, 70000000, 80000000, 90000000]
data_name="benchdata/opart/opart_brunel.txt"

def read_data_cut(filename,npart):
    file = open(filename, mode = 'r')
    data={'bsim':[],'metis':[],'model':[]}
    for line in file.read().split("\n"):
        if line=="":continue
        arrs=line.split(",")
        if int(arrs[1])==npart:
            data[arrs[0]].append(float(arrs[4]))
    return data



def plot_sub_cut(index,x,data,title,ylabel,lim):
    # plt.subplot(130+index+1)
    plt.figure(figsize=(4.8,3.2))
    plt.plot(x,data['bsim'],label="BSim-partition",c="tab:blue")
    plt.plot(x,data['metis'],label="Metis-based",c="tab:green")
    plt.plot(x,data['model'],label="Ours",c="tab:red")
    plt.ylabel(ylabel,labelpad=4.0)
    plt.ylim(lim)
    plt.xlim([5000000, 95000000])
    plt.xlabel("Synapse Count")
    plt.xticks(x)
    # plt.text(s="1e6",x=100000000,y=0)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(prop={"family": "Times New Roman", "size": 14},loc=2)
    # plt.title(title,y=0,pad=-70)
    if(index==1):
        plt.gca().yaxis.set_major_locator(MultipleLocator(0.04))
    elif (index==0):
        plt.gca().yaxis.set_major_locator(MultipleLocator(0.04))
    plt.gca().xaxis.set_major_locator(MultipleLocator(20000000))
    plt.savefig("pics/opart-ratio"+str(index+1)+".pdf", bbox_inches='tight', pad_inches=0.02)
    # plt.ticklabel_format(axis="y",style="sci",scilimits=(0,1))


data=read_data_cut(data_name,2)
plot_sub_cut(0,x,data,"(a) GPU count: 2","Ratio",[0.40,0.58])

data=read_data_cut(data_name,4)
plot_sub_cut(1,x,data,"(b) GPU count: 4","Ratio",[0.68,0.82])

data=read_data_cut(data_name,8)
plot_sub_cut(2,x,data,"(c) GPU count: 8","Ratio",[0.83,0.92])



# def read_data_synsd(filename,npart):
#     file = open(filename, mode = 'r')
#     data={'bsim':[],'metis':[],'model':[]}
#     for line in file.read().split("\n"):
#         if line=="":continue
#         arrs=line.split(",")
#         if int(arrs[1])==npart:
#             data[arrs[0]].append(float(arrs[6]))
#     return data

# def read_data_wgtsd(filename,npart):
#     file = open(filename, mode = 'r')
#     data={'bsim':[],'metis':[],'model':[]}
#     for line in file.read().split("\n"):
#         if line=="":continue
#         arrs=line.split(",")
#         if int(arrs[1])==npart:
#             data[arrs[0]].append(float(arrs[9]))
#     return data
# def plot_sub_syn(index,x,data,title,ylabel,pad,lim):
#     plt.subplot(130+index+1)
#     plt.plot(x,data['bsim'],label="Bsim-Par",c="tab:blue")
#     plt.plot(x,data['metis'],label="Metis-based",c="tab:green")
#     plt.plot(x,data['model'],label="Ours",c="tab:red")
#     plt.ylabel(ylabel,labelpad=pad)
#     plt.ylim(lim)
#     plt.xlim([5000000, 95000000])
#     plt.xlabel("Synapse Count")
#     plt.xticks(x)
#     # plt.text(s="1e6",x=100000000,y=lim[1]*-0.05)
#     plt.grid(True, linestyle="--", alpha=0.5)
#     plt.legend(prop={"family": "Times New Roman", "size": 11},loc=2)
#     plt.title(title,y=0,pad=-70)
#     plt.gca().xaxis.set_major_locator(MultipleLocator(20000000))
#     if(index==0):
#         plt.gca().yaxis.set_major_locator(MultipleLocator(3000000))
#     plt.ticklabel_format(axis="y",style="sci",scilimits=(0,1))

# def plot_sub_wgt(index,x,data,title,ylabel,pad,lim):
#     # plt.subplot(130+index+1)
#     plt.figure(figsize=(4.8,3.6))
#     plt.plot(x,data['bsim'],label="Bsim-partition",c="tab:blue")
#     plt.plot(x,data['metis'],label="Metis-based",c="tab:green")
#     plt.plot(x,data['model'],label="Ours",c="tab:red")
#     plt.ylabel(ylabel,labelpad=4.0)
#     plt.ylim(lim)
#     # plt.xlim([5000000, 95000000])
#     plt.xlabel("Synapse Count")
#     plt.xticks(x)
#     # plt.text(s="1e6",x=100000000,y=lim[1]*-0.05)
#     plt.grid(True, linestyle="--", alpha=0.5)
#     plt.legend(prop={"family": "Times New Roman", "size": 14},loc=2)
#     # plt.title(title,y=0,pad=-70)
#     plt.gca().xaxis.set_major_locator(MultipleLocator(20000000))
#     if(index==0):
#         plt.gca().yaxis.set_major_locator(MultipleLocator(50000))
#     elif (index==1):
#         plt.gca().yaxis.set_major_locator(MultipleLocator(30000))
#     elif (index==2):
#         plt.gca().yaxis.set_major_locator(MultipleLocator(20000))
#     plt.ticklabel_format(axis="y",style="sci",scilimits=(0,1))
#     plt.savefig("pics/opart-wgtsd"+str(index+1)+".pdf", bbox_inches='tight', pad_inches=0.02)
    

# plt.figure(figsize=(12,4))



# plt.tight_layout()
# plt.savefig("picture/opart-ratio.pdf", bbox_inches='tight', pad_inches=0.02)


# plt.figure(figsize=(12,4))
# plt.figure(figsize=(12,4))

# data=read_data_synsd(data_name,2)
# plot_sub_syn(0,x,data,"(a) GPU count: 2","SD",4,[-9.8e5,9.8e6])#GPU:2

# data=read_data_synsd(data_name,4)
# plot_sub_syn(1,x,data,"(b) GPU count: 4","SD",4,[-6e5,6e6])#GPU:4

# data=read_data_synsd(data_name,8)
# plot_sub_syn(2,x,data,"(c) GPU count: 8","SD",4,[-3.5e5,3.5e6])#GPU:8

# plt.tight_layout()
# plt.savefig("picture/opart-synsd.pdf", bbox_inches='tight', pad_inches=0.02)
# plt.tight_layout()
# plt.savefig("picture/opart-synsd.pdf", bbox_inches='tight', pad_inches=0.02)

# plt.figure(figsize=(12,4))

# data=read_data_wgtsd(data_name,2)
# plot_sub_wgt(0,x,data,"(a) GPU count: 2","SD",[-2.6e4,2.6e5])#GPU:2

# data=read_data_wgtsd(data_name,4)
# plot_sub_wgt(1,x,data,"(b) GPU count: 4","SD",[-1.6e4,1.6e5])#GPU:4

# data=read_data_wgtsd(data_name,8)
# plot_sub_wgt(2,x,data,"(c) GPU count: 8","SD",[-1.1e4,1.1e5])#GPU:8

# plt.tight_layout()
# plt.savefig("picture/opart-wgtsd.pdf", bbox_inches='tight', pad_inches=0.02)


# index="wgtsd"
# for npart in (2,4,8):
#     pic_name="analysis/opart/"+model_name+"_"+index+"_"+str(npart)+".png"
#     if index=="cut":
#         data=read_data_cut(data_name,npart)
#     elif index=="synsd":
#         data=read_data_synsd(data_name,npart)
#     elif index=="wgtsd":
#         data=read_data_wgtsd(data_name,npart)
    
#     bsim=data['bsim']
#     metis=data['metis']
#     model=data['model']
#     plt.subplot(330+index+1)
#     plt.plot(x,bsim,label="bsim")
#     plt.plot(x,metis,label="metis")
#     plt.plot(x,model,label="model")
#     plt.legend()
#     plt.savefig(pic_name)
