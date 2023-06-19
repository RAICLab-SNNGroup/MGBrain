#include "uvamgsim.h"

using std::get;
using std::pair;
using std::tuple;
using std::unordered_map;
using std::unordered_set;
using std::vector;
namespace MGBrain
{
    void MGBrain::MGSimulatorUVA::query_device()
    {
        //查询设备
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        NVMLCHECK(nvmlInit());
        std::vector<std::tuple<int,int,int>> devices(deviceCount);
        for (int d = 0; d < deviceCount; d++)
        {
            nvmlDevice_t device;
            NVMLCHECK(nvmlDeviceGetHandleByIndex(d, &device));
            nvmlUtilization_t utilization;
            NVMLCHECK(nvmlDeviceGetUtilizationRates(device, &utilization));
            devices[d]={d,utilization.gpu,utilization.memory};
        }
        std::sort(devices.begin(),devices.end(),[](std::tuple<int,int,int>& a,std::tuple<int,int,int>&b)->bool{
            if(std::get<1>(a)<std::get<1>(b))return true;
            else if(std::get<1>(a)==std::get<1>(b))return std::get<2>(a)<std::get<2>(b);
            return false;
        });
        for(int i=0;i<mapper.size();i++){
            mapper[i]=std::get<0>(devices[i]);
        }
        //开启P2P
        for(int i=0;i<mapper.size();i++){
            CUDACHECK(cudaSetDevice(mapper[i]));
            for(int j=0;j<mapper.size();j++){
                if(i==j)continue;
                CUDACHECK(cudaDeviceEnablePeerAccess(mapper[j],0));
            }
        }
        std::cout<<"Enable Peer Access"<<std::endl;
    }
    void MGBrain::MGSimulatorUVA::build_multinet_uva(Network &net, std::vector<int> part, int npart)
    {
        typedef tuple<int, int, real, int> syn_t;
        const int SRC = 0, TAR = 1, WEIGHT = 2, DELAY = 3;
        // vector<GSubNetUVA *> nets(npart, nullptr);
        multinet.cnets.resize(npart, nullptr);
        // 神经元的绝对位置与分部中相对位置映射
        std::vector<int> mapper(net.neurons.size());
        vector<vector<int>> neugrid(npart, vector<int>());
        // vector<int> neunums(npart, 0);
        // 记录各个分部中的内部突触
        vector<vector<syn_t>> syns(npart, vector<syn_t>());
        // 记录各个分部中的外部突触
        vector<vector<vector<syn_t>>> outs(npart, vector<vector<syn_t>>(npart, vector<syn_t>()));
        // 把神经元在整个网络中的绝对位置映射到子网络的相对位置上
        for (int i = 0; i < net.neurons.size(); i++)
        {
            net.neurons[i].id=i;
            mapper[i] = neugrid[part[i]].size(); // neunums[part[i]]++;
            neugrid[part[i]].push_back(i);
        }
        int max_delay = 0;
        for (int i = 0; i < net.synapses.size(); i++)
        {
            int src = net.synapses[i].src;
            int tar = net.synapses[i].tar;
            real weight = net.synapses[i].weight;
            int delay = net.synapses[i].delay / Config::STEP;
            max_delay = std::max(max_delay, delay);
            if (part[src] == part[tar])
            { // 内部突触
                syns[part[src]].emplace_back(mapper[src], mapper[tar], weight, delay);
            }
            else
            { // 外部突触
                outs[part[src]][part[tar]].emplace_back(mapper[src], mapper[tar], weight, delay);
            }
        }
        multinet.max_delay=max_delay;
        for (int k = 0; k < npart; k++)
        {
            multinet.cnets[k] = new GSubNetUVA();
            multinet.cnets[k]->id = k;
            // multinet.cnets[k]->max_delay = max_delay;
            multinet.cnets[k]->neus_size = neugrid[k].size();
            
            // 初始化神经元
            initGSubNetUVANeus(multinet.cnets[k],max_delay);
            // 初始化泊松神经元
            for (int i = 0; i < multinet.cnets[k]->neus_size; i++)
            {
                multinet.cnets[k]->neus.ids[i] = net.neurons[neugrid[k][i]].id;
                multinet.cnets[k]->neus.poisson[i] = net.neurons[neugrid[k][i]].source;
                multinet.cnets[k]->neus.rate[i] = net.neurons[neugrid[k][i]].rate;
            }
            multinet.cnets[k]->syns_size = syns[k].size();
            // 初始化突触
            initGSubNetUVASyns(multinet.cnets[k]);
            for (int i = 0; i < syns[k].size(); i++)
            {
                multinet.cnets[k]->syns.src[i] = get<SRC>(syns[k][i]);
                multinet.cnets[k]->syns.tar[i] = get<TAR>(syns[k][i]);
                multinet.cnets[k]->syns.weight[i] = get<WEIGHT>(syns[k][i]);
                multinet.cnets[k]->syns.delay[i] = get<DELAY>(syns[k][i]);
            }
            int devicenum = 0;
            for (int i = 0; i < outs[k].size(); i++)
            {
                if (!outs[k][i].empty())
                    devicenum++; // 当前分部k到分部i之间没有突触
            }
            multinet.cnets[k]->outs_size = devicenum;
            // 初始化外部突触
            multinet.cnets[k]->outs = new OUTSYNBlock[devicenum];

            // 交错排布
            for (int n = 0, d = 0, i = k; n < outs[k].size(); n++, i++)
            {
                i = i % outs[k].size();
                if (outs[k][i].empty())
                    continue;
                multinet.cnets[k]->outs[d].tar_id = i;
                multinet.cnets[k]->outs[d].syn_size = outs[k][i].size();
                // 初始化外部突触
                initGSubNetUVAOutSyns(&(multinet.cnets[k]->outs[d].block), multinet.cnets[k]->outs[d].syn_size);
                for (int j = 0; j < outs[k][i].size(); j++)
                {
                    multinet.cnets[k]->outs[d].block.src[j] = get<SRC>(outs[k][i][j]);
                    multinet.cnets[k]->outs[d].block.tar[j] = get<TAR>(outs[k][i][j]);
                    multinet.cnets[k]->outs[d].block.weight[j] = get<WEIGHT>(outs[k][i][j]);
                    multinet.cnets[k]->outs[d].block.delay[j] = get<DELAY>(outs[k][i][j]);
                }
                d++;
            }
            // 正常排布
            // for (int i = 0, d = 0; i < outs[k].size(); i++)
            // {
            //     if (outs[k][i].empty())
            //         continue;
            //     multinet.cnets[k]->outs[d].tar_id = i;
            //     multinet.cnets[k]->outs[d].syn_size = outs[k][i].size();
            //     // 初始化外部突触
            //     initGSubNetUVAOutSyns(&(multinet.cnets[k]->outs[d].block), multinet.cnets[k]->outs[d].syn_size);
            //     for (int j = 0; j < outs[k][i].size(); j++)
            //     {
            //         multinet.cnets[k]->outs[d].block.src[j] = get<SRC>(outs[k][i][j]);
            //         multinet.cnets[k]->outs[d].block.tar[j] = get<TAR>(outs[k][i][j]);
            //         multinet.cnets[k]->outs[d].block.weight[j] = get<WEIGHT>(outs[k][i][j]);
            //         multinet.cnets[k]->outs[d].block.delay[j] = get<DELAY>(outs[k][i][j]);
            //     }
            //     d++;
            // }
        }
    }
    void MGBrain::MGSimulatorUVA::copy_gnetuva_to_gpu()
    {
        int net_num = multinet.cnets.size();
        multinet.gnets.resize(net_num, nullptr);
        multinet.addrs.resize(net_num, nullptr);
        multinet.gaddrs.resize(net_num, vector<GSubNetAddrsUVA *>());
        multinet.gstreams.resize(net_num,vector<cudaStream_t>());
        for (int i = 0; i < net_num; i++)
        {
            multinet.addrs[i] = new GSubNetAddrsUVA;
            CUDACHECK(cudaSetDevice(mapper[i]));
            multinet.gnets[i] = copy_subnetuva_gpu(multinet.cnets[i], multinet.addrs[i],multinet.max_delay);
        }

        for (int i = 0; i < net_num; i++)
        {
            int outnum = multinet.cnets[i]->outs_size;
            multinet.gaddrs[i].resize(outnum, nullptr);
            multinet.gstreams[i].resize(outnum);
            cudaSetDevice(mapper[i]);
            for (int j = 0; j < outnum; j++)
            {
                cudaStreamCreate(&(multinet.gstreams[i][j]));
                multinet.gaddrs[i][j] = multinet.addrs[multinet.cnets[i]->outs[j].tar_id];
            }
        }
    }
    MGBrain::MGSimulatorUVA::MGSimulatorUVA(Network &net, std::vector<int> part, int nparts)
    {
        multinet.blocksize=1024;
        mapper.resize(nparts, 0);
        if(Config::SINGLE_GPU){
            int device=0;
            for (int i = 0; i < nparts; i++)
            {
                mapper[i] = device;
            }
            std::cout<<"run on single GPU"<<std::endl;
        }else{
            query_device();
            if(mapper.size()==1)
                std::cout<<"run on single GPU"<<std::endl;
            else
                std::cout<<"run on multi GPU"<<std::endl;
        }
        build_multinet_uva(net, part, nparts);
        copy_gnetuva_to_gpu();
        // copy_consts_to_gpu();
        copy_constsuva_gpu(multinet.max_delay);
    }
    MGBrain::MGSimulatorUVA::~MGSimulatorUVA()
    {
        //清除内存中的网络数据
        for (int i = 0; i < multinet.cnets.size(); i++)
        {
            freeGSubNetUVA(multinet.cnets[i]);
        }
        //清除GPU显存中的网络数据
        for (int i = 0; i < multinet.gnets.size(); i++)
        {
            gpuFreeGSubNetUVA(multinet.gnets[i]);
        }
        //清除CUDA流对象
        for (int i=0;i<multinet.gstreams.size();i++){
            for(int j=0;j<multinet.gstreams[i].size();j++){
                cudaStreamDestroy(multinet.gstreams[i][j]);
            }
        }
        //关闭PeerAccess
        for(int i=0;i<mapper.size();i++){
            CUDACHECK(cudaSetDevice(mapper[i]));
            for(int j=0;j<mapper.size();j++){
                if(i==j)continue;
                CUDACHECK(cudaDeviceDisablePeerAccess(mapper[j]));
            }
        }
    }
    void MGBrain::MGSimulatorUVA::simulate(real time)
    {
        int net_num = multinet.cnets.size();
        
        for (int t = 0; t < time / Config::STEP; t++)
        {
            for (int d = 0; d < net_num; d++)
            {
                CUDACHECK(cudaSetDevice(mapper[d]));
                // std::cout<<"device:"<<mapper[d]<<std::endl;
                mgsimStepUVA(multinet.gnets[d], multinet.cnets[d], t, multinet.blocksize, multinet.gaddrs[d],multinet.gstreams[d]);
                CUDACHECK(cudaDeviceSynchronize());
            }
        }
    }
};