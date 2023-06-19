#include "esamgsim.h"
using std::get;
using std::pair;
using std::tuple;
using std::unordered_map;
using std::unordered_set;
using std::vector;
void MGBrain::MGSimulator::build_multinet(Network &net, std::vector<int> part, int npart)
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
    vector<vector<int>> ins(npart, vector<int>(npart, 0));


    // 把神经元在整个网络中的绝对位置映射到子网络的相对位置上
    for (int i = 0; i < net.neurons.size(); i++)
    {
        net.neurons[i].id = i;
        mapper[i] = neugrid[part[i]].size();
        neugrid[part[i]].push_back(i);
    }
    int max_delay = 0;
    int min_delay = INT_MAX;
    for (size_t i = 0; i < net.synapses.size(); i++)
    {
        int src = net.synapses[i].src;
        int tar = net.synapses[i].tar;
        real weight = net.synapses[i].weight;
        int delay = net.synapses[i].delay / Config::STEP;
        max_delay = std::max(max_delay, delay);
        min_delay = std::min(min_delay, delay);
        if (part[src] == part[tar])
        { // 内部突触
            syns[part[src]].emplace_back(mapper[src], mapper[tar], weight, delay);
        }
        else
        { // 外部突触
            outs[part[src]][part[tar]].emplace_back(mapper[src], mapper[tar], weight, delay);
        }
    }
    /// 计算每个设备中从其他设备连入的数量
    vector<int> ins_num(npart, 0);
    for (int i = 0; i < npart; i++)
        for (int j = 0; j < npart; j++)
            if (ins[i][j] != 0)
                ins_num[i]++;

    multinet.max_delay = max_delay;
    multinet.min_delay = min_delay;
    for (int k = 0; k < npart; k++)
    {
        multinet.cnets[k] = new GSubNet();
        multinet.cnets[k]->id = k;
        multinet.cnets[k]->in_net_size = ins_num[k];
        multinet.cnets[k]->neus_size = neugrid[k].size();

        // 初始化神经元
        init_gsubnet_neus(multinet.cnets[k], max_delay);
        // 初始化神经元信息
        for (int i = 0; i < multinet.cnets[k]->neus_size; i++)
        {
            multinet.cnets[k]->neus.ids[i] = net.neurons[neugrid[k][i]].id;
            multinet.cnets[k]->neus.poisson[i] = net.neurons[neugrid[k][i]].source;
            multinet.cnets[k]->neus.type[i] = net.neurons[neugrid[k][i]].type;
            multinet.cnets[k]->neus.rate[i] = net.neurons[neugrid[k][i]].rate;
        }
        size_t sid = 0;
        multinet.cnets[k]->syns_size = syns[k].size();
        // 初始化突触
        init_gsubnet_syns(multinet.cnets[k]);
        for (int i = 0; i < syns[k].size(); i++)
        {
            size_t ref = gen_syn_ref(255, i);
            int src = get<SRC>(syns[k][i]);

            multinet.cnets[k]->syns.src[i] = src;
            multinet.cnets[k]->syns.tar[i] = get<TAR>(syns[k][i]);
            multinet.cnets[k]->syns.weight[i] = get<WEIGHT>(syns[k][i]);
            multinet.cnets[k]->syns.delay[i] = get<DELAY>(syns[k][i]);
        }
        int outsnum = 0;
        for (int i = 0; i < outs[k].size(); i++)
        {
            if (!outs[k][i].empty())
                outsnum++; // 当前分部k到分部i之间没有突触
        }
        multinet.cnets[k]->out_net_size = outsnum;
        // 初始化外部突触
        multinet.cnets[k]->outs = new OUTSYNBlock[outsnum];
        // 交错排布
        for (int n = 0, d = 0, i = k; n < outs[k].size(); n++, i++)
        {
            i = i % outs[k].size();
            if (outs[k][i].empty())
                continue;
            multinet.cnets[k]->outs[d].tar_id = i;
            multinet.cnets[k]->outs[d].syn_size = outs[k][i].size();
            // 初始化外部突触
            initGSubNetOutSyns(&(multinet.cnets[k]->outs[d].block), multinet.cnets[k]->outs[d].syn_size);
            for (int j = 0; j < outs[k][i].size(); j++)
            {

                multinet.cnets[k]->outs[d].block.src[j] = get<SRC>(outs[k][i][j]);
                multinet.cnets[k]->outs[d].block.tar[j] = get<TAR>(outs[k][i][j]);
                multinet.cnets[k]->outs[d].block.weight[j] = get<WEIGHT>(outs[k][i][j]);
                multinet.cnets[k]->outs[d].block.delay[j] = get<DELAY>(outs[k][i][j]);
            }
            d++;
        }
    }
}
void MGBrain::MGSimulator::copy_gnets_to_gpu()
{
    int net_num = multinet.cnets.size();
    multinet.gnets.resize(net_num, nullptr);
    for (int i = 0; i < net_num; i++)
    {
        CUDACHECK(cudaSetDevice(mapper[i]));
        multinet.gnets[i] = copy_subnet_gpu(multinet.cnets[i], multinet.max_delay);
    }
    multinet.manager.initBufferManager(net_num);

    multinet.manager.trans_streams.resize(net_num, vector<cudaStream_t>());
    multinet.manager.recv_streams.resize(net_num, vector<cudaStream_t>());
    for (int i = 0; i < net_num; i++)
    {
        int outnum = multinet.cnets[i]->out_net_size;
        int innum = multinet.cnets[i]->in_net_size;
        // std::cout<<outnum<<","<<innum<<std::endl;
        multinet.manager.trans_streams[i].resize(outnum);
        multinet.manager.recv_streams[i].resize(innum);
    }
    /// 生成记录缓冲池大小的列表
    for (int i = 0; i < net_num; i++)
    {
        int outnum = multinet.cnets[i]->out_net_size;
        int innum = multinet.cnets[i]->in_net_size;
        cudaSetDevice(mapper[i]);
        int *size_list0 = new int[outnum]();
        int *size_list1 = new int[outnum]();
        multinet.manager.netbuffers0[i].cbuffer_size_list = size_list0;
        multinet.manager.netbuffers0[i].gbuffer_size_list = toGPU(size_list0, outnum);
        multinet.manager.netbuffers1[i].cbuffer_size_list = size_list1;
        multinet.manager.netbuffers1[i].gbuffer_size_list = toGPU(size_list1, outnum);

        for (int j = 0; j < outnum; j++)
        {
            int tar_id = multinet.cnets[i]->outs[j].tar_id;
            int syn_size = multinet.cnets[i]->outs[j].syn_size;
            int buffer_cap = syn_size * (multinet.min_delay / 2 + 1);
            SpikeBuffer *cbuffer = new SpikeBuffer;
            ;
            init_buffer(cbuffer, buffer_cap);
            int t = multinet.manager.netbuffers0[i].cout_buffers.size();
            SpikeBuffer *gbuffer0 = copy_buffer_gpu(cbuffer, buffer_cap);
            SpikeBuffer *gbuffer1 = copy_buffer_gpu(cbuffer, buffer_cap);
            multinet.manager.netbuffers0[i].cout_buffers.push_back(gbuffer0);
            multinet.manager.netbuffers1[i].cout_buffers.push_back(gbuffer1);

            CUDACHECK(cudaStreamCreate(&(multinet.manager.trans_streams[i][t])));

            int tar_net = multinet.cnets[i]->outs[t].tar_id;
            int tar_idx = multinet.manager.netbuffers0[tar_net].cin_buffers.size();
            multinet.manager.mapperouts[i][t] = {tar_net, tar_idx};
            multinet.manager.mapperins[tar_net][tar_idx] = {i, t};
            CUDACHECK(cudaSetDevice(mapper[tar_net]));
            multinet.manager.netbuffers0[tar_net].cin_buffers.push_back(copy_buffer_gpu(cbuffer, buffer_cap));
            multinet.manager.netbuffers1[tar_net].cin_buffers.push_back(copy_buffer_gpu(cbuffer, buffer_cap));
            CUDACHECK(cudaStreamCreate(&(multinet.manager.recv_streams[tar_net][tar_idx])));
            CUDACHECK(cudaSetDevice(mapper[i]));
            free_buffer(cbuffer);
        }
    }
    for (int i = 0; i < net_num; i++)
    {
        multinet.manager.netbuffers0[i].ggout_buffers = copy_buffers_gpu(multinet.manager.netbuffers0[i].cout_buffers);
        multinet.manager.netbuffers1[i].ggout_buffers = copy_buffers_gpu(multinet.manager.netbuffers1[i].cout_buffers);
        multinet.manager.netbuffers0[i].ggin_buffers = copy_buffers_gpu(multinet.manager.netbuffers0[i].cin_buffers);
        multinet.manager.netbuffers1[i].ggin_buffers = copy_buffers_gpu(multinet.manager.netbuffers1[i].cin_buffers);
    }

    /// 生成脉冲传输流和脉冲收集流
}
void MGBrain::MGSimulator::query_device()
{
    // 查询设备
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    NVMLCHECK(nvmlInit());
    std::vector<std::tuple<int, int, int>> devices(deviceCount);
    for (int d = 0; d < deviceCount; d++)
    {
        nvmlDevice_t device;
        NVMLCHECK(nvmlDeviceGetHandleByIndex(d, &device));
        nvmlUtilization_t utilization;
        NVMLCHECK(nvmlDeviceGetUtilizationRates(device, &utilization));
        devices[d] = {d, utilization.gpu, utilization.memory};
    }
    std::sort(devices.begin(), devices.end(), [](std::tuple<int, int, int> &a, std::tuple<int, int, int> &b) -> bool
              {
            if(std::get<1>(a)<std::get<1>(b))return true;
            else if(std::get<1>(a)==std::get<1>(b))return std::get<2>(a)<std::get<2>(b);
            return false; });
    for (int i = 0; i < mapper.size(); i++)
    {
        mapper[i] = std::get<0>(devices[i]);
    }
    // 开启P2P
    if (Config::PEER_ACCESS)
    {
        for (int i = 0; i < mapper.size(); i++)
        {
            CUDACHECK(cudaSetDevice(mapper[i]));
            for (int j = 0; j < mapper.size(); j++)
            {
                if (i == j)
                    continue;
                CUDACHECK(cudaDeviceEnablePeerAccess(mapper[j], 0));
            }
        }
        std::cout << "Enable Peer Access" << std::endl;
    }
}
MGBrain::MGSimulator::MGSimulator(Network &net, std::vector<int> part, int nparts)
{
    multinet.blocksize = 1024;
    mapper.resize(nparts, 0);
    if (Config::SINGLE_GPU)
    {
        int device = 0;
        for (int i = 0; i < nparts; i++)
        {
            mapper[i] = device;
        }
        if (mapper.size() == 1)
            std::cout << "run single-net on single-GPU" << std::endl;
        else
            std::cout << "run multi-net on single-GPU" << std::endl;
    }
    else
    {
        query_device();
        if (mapper.size() == 1)
            std::cout << "run single-net on single-GPU" << std::endl;
        else
            std::cout << "run multi-net on multi-GPU" << std::endl;
    }
    build_multinet(net, part, nparts);
    copy_gnets_to_gpu();
    copy_consts_gpu(multinet.max_delay);
}
MGBrain::MGSimulator::~MGSimulator()
{
    for (int i = 0; i < multinet.cnets.size(); i++)
    {
        free_gsubnet(multinet.cnets[i]);
    }
    for (int i = 0; i < multinet.gnets.size(); i++)
    {
        free_gsubnet_gpu(multinet.gnets[i]);
    }
    for (int i = 0; i < multinet.manager.netbuffers0.size(); i++)
    {
        gpuFree(multinet.manager.netbuffers0[i].gbuffer_size_list);
        gpuFree(multinet.manager.netbuffers1[i].gbuffer_size_list);
        for (int j = 0; j < multinet.manager.netbuffers0[i].cout_buffers.size(); j++)
        {
            free_buffer_gpu(multinet.manager.netbuffers0[i].cout_buffers[j]);
            free_buffer_gpu(multinet.manager.netbuffers1[i].cout_buffers[j]);
        }
        gpuFree(multinet.manager.netbuffers0[i].ggout_buffers);
        gpuFree(multinet.manager.netbuffers1[i].ggout_buffers);
        for (int j = 0; j < multinet.manager.netbuffers0[i].cin_buffers.size(); j++)
        {
            free_buffer_gpu(multinet.manager.netbuffers0[i].cin_buffers[j]);
            free_buffer_gpu(multinet.manager.netbuffers1[i].cin_buffers[j]);
        }
        gpuFree(multinet.manager.netbuffers0[i].ggin_buffers);
        gpuFree(multinet.manager.netbuffers1[i].ggin_buffers);
    }
    // 关闭PeerAccess
    if (Config::PEER_ACCESS)
    {
        for (int i = 0; i < mapper.size(); i++)
        {
            CUDACHECK(cudaSetDevice(mapper[i]));
            for (int j = 0; j < mapper.size(); j++)
            {
                if (i == j)
                    continue;
                CUDACHECK(cudaDeviceDisablePeerAccess(mapper[j]));
            }
        }
        std::cout << "Disable Peer Access" << std::endl;
    }
    // 销毁流
    auto trans_streams = multinet.manager.trans_streams;
    for (int i = 0; i < trans_streams.size(); i++)
    {
        for (int j = 0; j < trans_streams[i].size(); j++)
        {
            CUDACHECK(cudaStreamDestroy(trans_streams[i][j]));
        }
    }
    auto recv_streams = multinet.manager.recv_streams;
    for (int i = 0; i < recv_streams.size(); i++)
    {
        for (int j = 0; j < recv_streams[i].size(); j++)
        {
            CUDACHECK(cudaStreamDestroy(recv_streams[i][j]));
        }
    }
}
void MGBrain::MGSimulator::simulate(real time)
{
    int net_num = multinet.cnets.size();
    int half_group_size = multinet.min_delay / 2;
    int sim_time_steps = time / Config::STEP;
    int blocksize = multinet.blocksize;
    BufferManager &manager = multinet.manager;
    std::vector<int> time_records(net_num, 0);
#pragma omp parallel num_threads(net_num)
    {

        int net_id = omp_get_thread_num();
        // std::cout<<net_id<<std::endl;
        CUDACHECK(cudaSetDevice(mapper[net_id]));
        int turn = 0;
        GSubNet *gnet = multinet.gnets[net_id];
        GSubNet *cnet = multinet.cnets[net_id];
        int id = cnet->id;
        for (int t = 0; t < sim_time_steps; t += half_group_size, turn = !turn)
        {
            // 仿真半个时间片组
            // time_t step_s,step_e;
            // step_s=clock();
            for (int i = t; i < t + half_group_size; i++)
            {
                mgsim_step(gnet, cnet, i, blocksize, manager, turn);
            }

            /// 设备同步点,多线程同步点
            // CUDACHECK(cudaStreamSynchronize((cudaStream_t)cudaStreamDefault));
            CUDACHECK(cudaDeviceSynchronize());
            // step_e=clock();
            // #pragma omp critical
            // {
            //     std::cout<<net_id<<" step elapsed:"<<(float)(step_e-step_s)/1000<<" ms"<<std::endl;
            // }
#pragma omp barrier
            manager.syncBufferSizeList(id, turn);
            CUDACHECK(cudaDeviceSynchronize());
            /// 脉冲同步分为两步：脉冲数据传输和脉冲数据收集
            /// 脉冲数据传输
            // time_t trans_s,trans_e;
            // trans_s=clock();
            trans_spikes(cnet, manager, turn, manager.trans_streams[id], mapper);
            // trans_e=clock();
            // #pragma omp critical
            // {
            // std::cout<<net_id<<" trans elapsed:"<<(float)(trans_e-trans_s)/1000<<"ms"<<std::endl;
            // }
#pragma omp barrier
            /// 脉冲数据收集
            // time_t recv_s,recv_e;
            // recv_s=clock();
            recvs_spikes(gnet, cnet, blocksize, manager, turn, manager.recv_streams[id]);
            // recv_e=clock();
            // #pragma omp critical
            // {
            // std::cout<<net_id<<" recvs elapsed:"<<(float)(recv_e-recv_s)/1000<<"ms"<<std::endl;
            // }
#pragma omp barrier
            manager.clearGBuffer(id, turn);
        }
    }
}