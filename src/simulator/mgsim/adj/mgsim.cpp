#include "mgsim.h"
using std::get;
using std::pair;
using std::tuple;
using std::unordered_map;
using std::unordered_set;
using std::vector;
void MGBrain::MGSimulator::build_multinet(Network &net, std::vector<int> &part, int npart)
{
    typedef tuple<int, int, real, int> syn_t;
    const int SRC = 0, TAR = 1, WEIGHT = 2, DELAY = 3;
    multinet.cnets.resize(npart, nullptr);
    // 神经元的绝对位置与分部中相对位置映射
    std::vector<int> nmapper(net.neurons.size());
    vector<vector<int>> neugrid(npart, vector<int>());
    // 记录各个分部中的突触
    vector<vector<syn_t>> syns(npart, vector<syn_t>());

    vector<vector<vector<int>>> outs(npart, vector<vector<int>>(npart));
    vector<size_t> net_axon_num(npart, 0);
    vector<size_t> net_dend_num(npart, 0);

    // 把神经元在整个网络中的绝对位置映射到子网络的相对位置上
    for (int i = 0; i < net.neurons.size(); i++)
    {
        net.neurons[i].id = i;
        nmapper[i] = neugrid[part[i]].size();
        neugrid[part[i]].push_back(i);
    }
    vector<vector<vector<size_t>>> axons(npart);
    vector<vector<vector<size_t>>> dends(npart);
    for (int i = 0; i < npart; i++)
    {
        axons[i].resize(neugrid[i].size());
        dends[i].resize(neugrid[i].size());
    }
    for (int i = 0; i < net.neurons.size(); i++)
    {
        axons[part[i]][nmapper[i]].reserve(net.neurons[i].nxt.size());
        dends[part[i]][nmapper[i]].reserve(net.neurons[i].pre.size());
    }
    int max_delay = 0;
    int min_delay = INT_MAX;
    for (size_t i = 0; i < net.synapses.size(); i++)
    {
        int src = net.synapses[i].src;
        int tar = net.synapses[i].tar;
        real weight = net.synapses[i].weight;
        int delay = std::round(net.synapses[i].delay / Config::STEP);
        max_delay = std::max(max_delay, delay);
        min_delay = std::min(min_delay, delay);
        size_t offset = syns[part[src]].size();
        if (part[src] == part[tar])
        { // 内部突触
            size_t ref = gen_syn_ref(part[src], offset);
            axons[part[src]][nmapper[src]].push_back(ref);
            dends[part[tar]][nmapper[tar]].push_back(ref);
        }
        else
        { // 外部突触
            size_t axref = gen_syn_ref(part[tar], offset);
            size_t deref = gen_syn_ref(part[src], offset);
            axons[part[src]][nmapper[src]].push_back(axref);
            dends[part[tar]][nmapper[tar]].push_back(deref);
            outs[part[src]][part[tar]].push_back(nmapper[tar]);
        }
        syns[part[src]].emplace_back(nmapper[src], nmapper[tar], weight, delay);
        net_axon_num[part[src]]++;
        net_dend_num[part[tar]]++;
    }
    max_delay++;
    multinet.max_delay = max_delay;
    multinet.min_delay = min_delay;
    // 初始化神经网络
    for (int k = 0; k < npart; k++)
    {
        multinet.cnets[k] = new GSubNet();
        multinet.cnets[k]->id = k;
        multinet.cnets[k]->npart = npart;
        
        multinet.cnets[k]->out_net_id_list = new int[npart];
        multinet.cnets[k]->out_syn_size_list = new size_t[npart];
        // 交错排布
        for (int n = 0,  i = k; n < npart;n++, i++)
        {
            i = i % npart;
            multinet.cnets[k]->out_net_id_list[n] = i;
            multinet.cnets[k]->out_syn_size_list[n] = outs[k][i].size();
        }

        // 初始化神经元
        size_t neus_size = neugrid[k].size();
        init_gsubnet_neus(multinet.cnets[k],neus_size, max_delay);
        for (int i = 0; i < neus_size; i++)
        {
            multinet.cnets[k]->neus.ids[i] = net.neurons[neugrid[k][i]].id;
            multinet.cnets[k]->neus.V_m[i] = net.lifconst[1];
            multinet.cnets[k]->neus.source[i] = net.neurons[neugrid[k][i]].source;
            multinet.cnets[k]->neus.type[i] = net.neurons[neugrid[k][i]].type;
            multinet.cnets[k]->neus.rate[i] = net.neurons[neugrid[k][i]].rate;
        }

        // 初始化突触
        size_t syns_size = syns[k].size();
        init_gsubnet_syns(multinet.cnets[k],syns_size);
        for (int i = 0; i < syns_size; i++)
        {
            multinet.cnets[k]->syns.src[i] = get<SRC>(syns[k][i]);
            multinet.cnets[k]->syns.tar[i] = get<TAR>(syns[k][i]);
            multinet.cnets[k]->syns.weight[i] = get<WEIGHT>(syns[k][i]);
            multinet.cnets[k]->syns.delay[i] = get<DELAY>(syns[k][i]);
        }

        // 初始化邻接信息
        init_gsubnet_adjs(multinet.cnets[k], net_axon_num[k], net_dend_num[k]);
        for (int i = 0, a = 0, d = 0; i < neus_size; i++)
        {
            size_t axon_size = axons[k][i].size();
            size_t dend_size = dends[k][i].size();

            multinet.cnets[k]->adjs.axon_offs[i + 1] = multinet.cnets[k]->adjs.axon_offs[i] + axon_size;
            multinet.cnets[k]->adjs.dend_offs[i + 1] = multinet.cnets[k]->adjs.dend_offs[i] + dend_size;
            for (int j = 0; j < axon_size; j++, a++)
            {
                multinet.cnets[k]->adjs.axon_refs[a] = axons[k][i][j];
            }
            for (int j = 0; j < dend_size; j++, d++)
            {
                multinet.cnets[k]->adjs.dend_refs[d] = dends[k][i][j];
            }
        }
    }
    // 神经网络拷贝到GPU

    // 构建缓冲池
    if (Config::DENSE_SPIKE)
    {
        gen_dense_buffer(outs);
    }
    else
    {
        gen_normal_buffer();
    }
}
void MGBrain::MGSimulator::copy_gnets_to_gpu(Network &net)
{
    int net_num = multinet.cnets.size();
    multinet.gnets.resize(net_num, nullptr);
    multinet.caddrs.clast_fired_addrs.resize(net_num);
    multinet.caddrs.csyn_src_addrs.resize(net_num);
    multinet.caddrs.csyn_weight_addrs.resize(net_num);
    multinet.gaddrs.resize(net_num);
    // 构建GPU仿真网络并收集全局地址
    for (int i = 0; i < net_num; i++)
    {
        CUDACHECK(cudaSetDevice(mapper[i]));

        multinet.gnets[i] = copy_subnet_gpu(multinet.cnets[i], multinet.max_delay, multinet.caddrs);
    }
    /// 将全局地址复制到所有设备上
    for (int i = 0; i < net_num; i++)
    {
        CUDACHECK(cudaSetDevice(mapper[i]));
        copy_netaddrs_gpu(multinet.gaddrs[i], multinet.caddrs);
        if (net.nlifconst || net.nstdpconst)
        {
            copy_consts_gpu(multinet.max_delay, Config::DT, net.nlifconst, net.lifconst, net.nstdpconst, net.stdpconst);
        }
        else
        {
            copy_consts_gpu(multinet.max_delay, Config::DT);
        }
    }
}
void MGBrain::MGSimulator::gen_normal_buffer()
{
    int net_num = multinet.cnets.size();
    /// 生成脉冲缓冲池
    multinet.manager.initBufferManager(net_num);
    for (int i = 0; i < net_num; i++)
    {
        int net_id = i;
        cudaSetDevice(mapper[net_id]);
        multinet.manager.netbuffers0[net_id].gbuffer_size_list = init_buffer_size_list_gpu(net_num);
        multinet.manager.netbuffers1[net_id].gbuffer_size_list = init_buffer_size_list_gpu(net_num);
        CUDACHECK(cudaStreamCreate(&(multinet.manager.trans_streams[net_id])));
        CUDACHECK(cudaStreamCreate(&(multinet.manager.sim_streams[net_id])));
        for (int j = 0; j < net_num; j++)
        {
            int src_net = net_id;
            int tar_net = multinet.cnets[i]->out_net_id_list[j];
            int src_idx = tar_net;
            int tar_idx = src_net;
            int syn_size = multinet.cnets[i]->out_syn_size_list[j];
            multinet.manager.mapperouts[src_net][src_idx] = {tar_net, tar_idx};
            multinet.manager.mapperins[tar_net][tar_idx] = {src_net, src_idx};
            if (syn_size == 0 || src_net == tar_net)
                continue;

            // 计算缓冲区容量
            // int buffer_cap = syn_size * (multinet.min_delay / 2 + 1);//神经元激活率为100%时无访存出错的容量大小
            int buffer_cap = syn_size * 2;
            CUDACHECK(cudaSetDevice(mapper[src_net]));

            multinet.manager.netbuffers0[src_net].gout_buffers[src_idx] = init_buffer_gpu(buffer_cap, multinet.manager.netbuffers0[src_net].cout_buffers[src_idx]);
            multinet.manager.netbuffers1[src_net].gout_buffers[src_idx] = init_buffer_gpu(buffer_cap, multinet.manager.netbuffers1[src_net].cout_buffers[src_idx]);
            multinet.manager.out_valid[src_net][src_idx] = true;


            CUDACHECK(cudaSetDevice(mapper[tar_net]));
            multinet.manager.netbuffers0[tar_net].gin_buffers[tar_idx] = init_buffer_gpu(buffer_cap, multinet.manager.netbuffers0[tar_net].cin_buffers[tar_idx]);
            multinet.manager.netbuffers1[tar_net].gin_buffers[tar_idx] = init_buffer_gpu(buffer_cap, multinet.manager.netbuffers1[tar_net].cin_buffers[tar_idx]);
            CUDACHECK(cudaStreamCreate(&(multinet.manager.recv_streams[tar_net][tar_idx])));
            multinet.manager.in_valid[tar_net][tar_idx] = true;
            CUDACHECK(cudaSetDevice(mapper[net_id]));
        }
    }
    /// 拷贝所有缓冲池地址到GPU中
    for (int i = 0; i < net_num; i++)
    {
        int net_id = multinet.cnets[i]->id;
        cudaSetDevice(mapper[net_id]);
        multinet.manager.netbuffers0[i].ggout_buffers = copy_buffers_gpu(multinet.manager.netbuffers0[i].gout_buffers);
        multinet.manager.netbuffers1[i].ggout_buffers = copy_buffers_gpu(multinet.manager.netbuffers1[i].gout_buffers);
        multinet.manager.netbuffers0[i].ggin_buffers = copy_buffers_gpu(multinet.manager.netbuffers0[i].gin_buffers);
        multinet.manager.netbuffers1[i].ggin_buffers = copy_buffers_gpu(multinet.manager.netbuffers1[i].gin_buffers);
    }
}
void MGBrain::MGSimulator::gen_dense_buffer(std::vector<std::vector<std::vector<int>>> &out)
{
    int net_num = multinet.cnets.size();
    /// 生成脉冲缓冲池
    multinet.managerd.initBufferManager(net_num, multinet.max_delay);
    for (int i = 0; i < net_num; i++)
    {
        int net_id = i;
        cudaSetDevice(mapper[net_id]);
        CUDACHECK(cudaStreamCreate(&(multinet.managerd.trans_streams[net_id])));
        CUDACHECK(cudaStreamCreate(&(multinet.managerd.sim_streams[net_id])));
        for (int j = 0; j < net_num; j++)
        {
            int src_net = net_id;
            int tar_net = multinet.cnets[i]->out_net_id_list[j];
            int src_idx = tar_net;
            int tar_idx = src_net;
            int syn_size = multinet.cnets[i]->out_syn_size_list[j];
            multinet.managerd.mapperouts[src_net][src_idx] = {tar_net, tar_idx};
            multinet.managerd.mapperins[tar_net][tar_idx] = {src_net, src_idx};
            if (syn_size == 0 || src_net == tar_net||out[src_net][tar_net].size()==0)
                continue;
            std::unordered_set<int> nset;
            for (auto o:(out[src_net][tar_net]))
            {
                nset.insert(o);
            }
            std::vector<int> targets;
            targets.reserve(nset.size());
            for (auto n : nset)
            {
                targets.push_back(n);
            }
            int buffer_cap = targets.size();
            int max_delay = multinet.max_delay;
            if (buffer_cap == 0)continue;
            CUDACHECK(cudaSetDevice(mapper[src_net]));
            // std::cout<<"src_net:"<<src_net<<"["<<src_idx<<"]"<<"->"<<"tar_net:"<<tar_net<<"["<<tar_idx<<"]"<<"="<<buffer_cap<<std::endl;
            multinet.managerd.netbuffers0[src_net].cout_buffers[src_idx]=new SpikeDenseBuffer;
            multinet.managerd.netbuffers1[src_net].cout_buffers[src_idx]=new SpikeDenseBuffer;
            multinet.managerd.netbuffers0[src_net].gout_buffers[src_idx] = init_dense_buffer_gpu(buffer_cap, max_delay, multinet.managerd.netbuffers0[src_net].cout_buffers[src_idx], targets);
            multinet.managerd.netbuffers1[src_net].gout_buffers[src_idx] = init_dense_buffer_gpu(buffer_cap, max_delay, multinet.managerd.netbuffers1[src_net].cout_buffers[src_idx], targets);
            multinet.managerd.out_valid[src_net][src_idx] = true;
            multinet.managerd.out_neu_sizes[src_net][src_idx] = buffer_cap;
            CUDACHECK(cudaSetDevice(mapper[tar_net]));
            multinet.managerd.netbuffers0[tar_net].cin_buffers[tar_idx]=new SpikeDenseBuffer;
            multinet.managerd.netbuffers1[tar_net].cin_buffers[tar_idx]=new SpikeDenseBuffer;
            multinet.managerd.netbuffers0[tar_net].gin_buffers[tar_idx] = init_dense_buffer_gpu(buffer_cap, max_delay, multinet.managerd.netbuffers0[tar_net].cin_buffers[tar_idx], targets);
            multinet.managerd.netbuffers1[tar_net].gin_buffers[tar_idx] = init_dense_buffer_gpu(buffer_cap, max_delay, multinet.managerd.netbuffers1[tar_net].cin_buffers[tar_idx], targets);
            CUDACHECK(cudaStreamCreate(&(multinet.managerd.recv_streams[tar_net][tar_idx])));
            multinet.managerd.in_valid[tar_net][tar_idx] = true;
        }
    }
    /// 拷贝所有缓冲池地址到GPU中
    for (int i = 0; i < net_num; i++)
    {
        CUDACHECK(cudaSetDevice(mapper[i]));
        multinet.managerd.netbuffers0[i].ggout_buffers = copy_dense_buffers_gpu(multinet.managerd.netbuffers0[i].gout_buffers);
        multinet.managerd.netbuffers1[i].ggout_buffers = copy_dense_buffers_gpu(multinet.managerd.netbuffers1[i].gout_buffers);
        multinet.managerd.netbuffers0[i].ggin_buffers = copy_dense_buffers_gpu(multinet.managerd.netbuffers0[i].gin_buffers);
        multinet.managerd.netbuffers1[i].ggin_buffers = copy_dense_buffers_gpu(multinet.managerd.netbuffers1[i].gin_buffers);
    }
}
void MGBrain::MGSimulator::query_device()
{
    // 查询设备
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    NVMLCHECK(nvmlInit());
    std::vector<std::tuple<int, int, float>> devices(deviceCount);
    for (int d = 0; d < deviceCount; d++)
    {
        nvmlDevice_t device;
        NVMLCHECK(nvmlDeviceGetHandleByIndex(d, &device));
        nvmlUtilization_t utilization;
        nvmlMemory_t memory;
        NVMLCHECK(nvmlDeviceGetUtilizationRates(device, &utilization));
        NVMLCHECK(nvmlDeviceGetMemoryInfo(device, &memory));
        devices[d] = {d, utilization.gpu, (float)memory.free / 1024 / 1024 / 1024};
    }
    std::sort(devices.begin(), devices.end(), [](std::tuple<int, int, float> &a, std::tuple<int, int, float> &b) -> bool
              {
            if(std::get<1>(a)<std::get<1>(b))return true;
            else if(std::get<1>(a)==std::get<1>(b))return std::get<2>(a)<std::get<2>(b);
            return false; });
    for (int i = 0, d = 0; i < mapper.size() && d < deviceCount; d++)
    {
        if (std::get<2>(devices[d]) < Config::MIN_GPUMEM)
            continue;
        mapper[i++] = std::get<0>(devices[d]);
        // std::cout << "device(" << std::get<0>(devices[d]) << "):" << std::get<1>(devices[d]) << "%," << std::get<2>(devices[d]) << "GB" << std::endl;
    }
    // 开启P2P
    if (Config::PEER_ACCESS)
    {
        for (int i = 0; i < mapper.size(); i++)
        {
            CUDACHECK(cudaSetDevice(mapper[i]));
            for (int j = 0; j < mapper.size(); j++)
            {
                if (mapper[i] == mapper[j])
                    continue;
                CUDACHECK(cudaDeviceEnablePeerAccess(mapper[j], 0));
            }
        }
        // std::cout << "Enable Peer Access" << std::endl;
    }
}
MGBrain::MGSimulator::MGSimulator(Network &net, std::vector<int> &part, int nparts)
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
    copy_gnets_to_gpu(net);
}
MGBrain::MGSimulator::~MGSimulator()
{
    // 释放全局地址
    for (int i = 0; i < multinet.cnets.size(); i++)
    {
        free_netaddrs_gpu(multinet.gaddrs[i]);
    }
    // 释放神经网络仿真数据
    for (int i = 0; i < multinet.cnets.size(); i++)
    {
        free_gsubnet(multinet.cnets[i]);
    }
    for (int i = 0; i < multinet.gnets.size(); i++)
    {
        free_gsubnet_gpu(multinet.gnets[i]);
    }
    // 释放缓冲池
    if (Config::DENSE_SPIKE)
    {
        free_dense_buffer();
    }
    else
    {
        free_normal_buffer();
    }

    // 关闭PeerAccess
    if (Config::PEER_ACCESS)
    {
        for (int i = 0; i < mapper.size(); i++)
        {
            CUDACHECK(cudaSetDevice(mapper[i]));
            for (int j = 0; j < mapper.size(); j++)
            {
                if (mapper[i] == mapper[j])
                    continue;
                CUDACHECK(cudaDeviceDisablePeerAccess(mapper[j]));
            }
        }
        // std::cout << "Disable Peer Access" << std::endl;
    }
    CUDACHECK(cudaDeviceReset());
}
void MGBrain::MGSimulator::free_normal_buffer()
{
    // 释放缓冲池
    for (int i = 0; i < multinet.manager.netbuffers0.size(); i++)
    {
        delete[] multinet.manager.netbuffers0[i].cbuffer_size_list;
        delete[] multinet.manager.netbuffers1[i].cbuffer_size_list;
        gpuFree(multinet.manager.netbuffers0[i].gbuffer_size_list);
        gpuFree(multinet.manager.netbuffers1[i].gbuffer_size_list);
        for (int j = 0; j < multinet.manager.netbuffers0[i].gout_buffers.size(); j++)
        {
            if (multinet.manager.netbuffers0[i].gout_buffers[j] == nullptr)
                continue;
            free_buffer_gpu(multinet.manager.netbuffers0[i].gout_buffers[j]);
            free_buffer_gpu(multinet.manager.netbuffers1[i].gout_buffers[j]);
        }
        gpuFree(multinet.manager.netbuffers0[i].ggout_buffers);
        gpuFree(multinet.manager.netbuffers1[i].ggout_buffers);
        for (int j = 0; j < multinet.manager.netbuffers0[i].gin_buffers.size(); j++)
        {
            if (multinet.manager.netbuffers0[i].gin_buffers[j] == nullptr)
                continue;
            free_buffer_gpu(multinet.manager.netbuffers0[i].gin_buffers[j]);
            free_buffer_gpu(multinet.manager.netbuffers1[i].gin_buffers[j]);
        }
        gpuFree(multinet.manager.netbuffers0[i].ggin_buffers);
        gpuFree(multinet.manager.netbuffers1[i].ggin_buffers);
    }
    // 销毁流
    auto trans_streams = multinet.manager.trans_streams;
    for (int i = 0; i < trans_streams.size(); i++)
    {
        CUDACHECK(cudaStreamDestroy(trans_streams[i]));

    }
    auto recv_streams = multinet.manager.recv_streams;
    for (int i = 0; i < recv_streams.size(); i++)
    {
        for (int j = 0; j < recv_streams[i].size(); j++)
        {
            if (multinet.manager.in_valid[i][j])
                CUDACHECK(cudaStreamDestroy(recv_streams[i][j]));
        }
    }
}
void MGBrain::MGSimulator::free_dense_buffer()
{
    // 释放缓冲池
    for (int i = 0; i < multinet.managerd.netbuffers0.size(); i++)
    {
        for (int j = 0; j < multinet.managerd.netbuffers0[i].gout_buffers.size(); j++)
        {
            if (multinet.managerd.netbuffers0[i].gout_buffers[j] == nullptr)
                continue;
            free_dense_buffer_gpu(multinet.managerd.netbuffers0[i].gout_buffers[j]);
            free_dense_buffer_gpu(multinet.managerd.netbuffers1[i].gout_buffers[j]);
        }
        gpuFree(multinet.managerd.netbuffers0[i].ggout_buffers);
        gpuFree(multinet.managerd.netbuffers1[i].ggout_buffers);
        for (int j = 0; j < multinet.managerd.netbuffers0[i].gin_buffers.size(); j++)
        {
            if (multinet.managerd.netbuffers0[i].gin_buffers[j] == nullptr)
                continue;
            free_dense_buffer_gpu(multinet.managerd.netbuffers0[i].gin_buffers[j]);
            free_dense_buffer_gpu(multinet.managerd.netbuffers1[i].gin_buffers[j]);
        }
        gpuFree(multinet.managerd.netbuffers0[i].ggin_buffers);
        gpuFree(multinet.managerd.netbuffers1[i].ggin_buffers);
    }
    // 销毁流
    auto trans_streams = multinet.managerd.trans_streams;
    for (int i = 0; i < trans_streams.size(); i++)
    {
        CUDACHECK(cudaStreamDestroy(trans_streams[i]));
    }
    auto sim_streams = multinet.managerd.sim_streams;
    for (int i = 0; i < trans_streams.size(); i++)
    {
        CUDACHECK(cudaStreamDestroy(sim_streams[i]));
    }
    auto recv_streams = multinet.managerd.recv_streams;
    for (int i = 0; i < recv_streams.size(); i++)
    {
        for (int j = 0; j < recv_streams[i].size(); j++)
        {
            if (multinet.managerd.in_valid[i][j])
                CUDACHECK(cudaStreamDestroy(recv_streams[i][j]));
        }
    }
}

void MGBrain::MGSimulator::simulate_seq_sparse(real time)
{
    sim_time = 0.0;
    int net_num = multinet.cnets.size();
    int min_delay = multinet.min_delay;
    int pre_half_group_size = multinet.min_delay / 2;
    int post_half_group_size = multinet.min_delay - pre_half_group_size;
    int sim_time_steps =static_cast<int>( std::round(time / Config::STEP));
    int cycles = sim_time_steps / multinet.min_delay + (sim_time_steps % multinet.min_delay == 0);
    int blocksize = multinet.blocksize;
    BufferManager &manager = multinet.manager;
    int threads = net_num;
    std::vector<float> times(7);
    timer sim;
#pragma omp parallel num_threads(threads)
    {
        int step = 0;
        int net_id = omp_get_thread_num();
        CUDACHECK(cudaSetDevice(mapper[net_id]));
        GSubNet *gnet = multinet.gnets[net_id];
        GSubNet *cnet = multinet.cnets[net_id];
        #pragma omp barrier
        for (int c = 0; c < cycles; ++c)
        {
            //清除缓存
            recvs_spikes_sparse(gnet, cnet, blocksize, manager, 0, manager.recv_streams[net_id]);
            manager.clearBuffer(net_id, 0,manager.sim_streams[net_id]);
            //仿真网络
            for (int i = step; i < step + min_delay && i < sim_time_steps; ++i)
            {
                mgsim_step_sparse(gnet, cnet, i, blocksize, manager, multinet.gaddrs[net_id], 0, manager.sim_streams[net_id]);
            }
            manager.syncBufferSizeList(net_id, 0,manager.sim_streams[net_id]);
            /// 脉冲数据传输
            CUDACHECK(cudaStreamSynchronize(manager.sim_streams[net_id]));
            #pragma omp barrier
            /// 脉冲数据传输
            trans_spikes_sparse(cnet, manager, mapper, 0, manager.trans_streams[net_id]);
            /// 脉冲数据收集
            CUDACHECK(cudaStreamSynchronize(manager.trans_streams[net_id]));
            step += min_delay;
            #pragma omp barrier
        }
        
    }
    sim_time = sim.stop();
    if (Config::FIRE_CHECK)
    {
        get_firing_rate(time);
    }
}
void MGBrain::MGSimulator::simulate_seq_dense(real time)
{
    sim_time = 0.0;
    int net_num = multinet.cnets.size();
    int min_delay = multinet.min_delay;
    int pre_half_group_size = multinet.min_delay / 2;
    int post_half_group_size = multinet.min_delay - pre_half_group_size;
    int sim_time_steps = static_cast<int>( std::round(time / Config::STEP));
    int cycles = sim_time_steps / multinet.min_delay + (sim_time_steps % multinet.min_delay == 0);
    int blocksize = multinet.blocksize;
    DenseBufferManager &manager = multinet.managerd;
    int threads = net_num;
    std::vector<float> times(7);
    timer sim;
#pragma omp parallel num_threads(threads)
    {
        int step = 0;
        int net_id = omp_get_thread_num();
        CUDACHECK(cudaSetDevice(mapper[net_id]));
        GSubNet *gnet = multinet.gnets[net_id];
        GSubNet *cnet = multinet.cnets[net_id];
        #pragma omp barrier
        for (int c = 0; c < cycles; ++c)
        {
            // 接收域外脉冲
            recvs_spikes_dense(gnet, cnet, step, blocksize, manager, 0, manager.recv_streams[net_id]);
            // 仿真时间片组
            for (int i = step; i < step + min_delay && i < sim_time_steps; ++i)
            {
                mgsim_step_dense(gnet, cnet, i, blocksize, manager, multinet.gaddrs[net_id], 0);
            }
            // 传输设备间脉冲
            CUDACHECK(cudaStreamSynchronize(manager.sim_streams[net_id]));
            #pragma omp barrier
            // 传输设备间脉冲
            trans_spikes_dense(cnet, manager, 0, manager.trans_streams[net_id], mapper, multinet.max_delay);
            CUDACHECK(cudaStreamSynchronize(manager.trans_streams[net_id]));
            step += min_delay;
            #pragma omp barrier
        }
    }
    sim_time = sim.stop();
    if (Config::FIRE_CHECK)
    {
        get_firing_rate(time);
    }
}
MGBrain::real MGBrain::MGSimulator::get_firing_rate(real time)
{
    size_t allneu = 0;
    size_t allsum = 0;
    for (int i = 0; i < multinet.cnets.size(); i++)
    {
        int neusize = multinet.cnets[i]->neus.size;
        // std::cout<<"neusize:"<<neusize<<std::endl;
        size_t num = get_subnet_firecnt(multinet.cnets[i], multinet.gnets[i]);
        std::cout << "zone(" << i << ")'fire cnt:" << num << std::endl;
        std::cout << "zone(" << i << ")'fire rate:" << (float)num / neusize / time << " hz" << std::endl;
        allneu += neusize;
        allsum += num;
    }

    // std::cout << "fire cnt:" << allsum << std::endl;
    float rate = ((float)allsum / allneu / time);
    std::cout << "fire rate:" << rate << "hz" << std::endl;
    return rate;
}
void MGBrain::MGSimulator::simulate_test_s(real time)
{
    sim_time = 0.0;
    int net_num = multinet.cnets.size();
    int pre_half_group_size = multinet.min_delay / 2;
    int post_half_group_size = multinet.min_delay - pre_half_group_size;
    int sim_time_steps = time / Config::STEP;

    int cycles = sim_time_steps / multinet.min_delay + (sim_time_steps % multinet.min_delay == 0);
    int blocksize = multinet.blocksize;
    BufferManager &manager = multinet.manager;
    int threads = net_num * 2;
    int step = 0;
    std::vector<float> times(7);
    timer sim;
    for (int c = 0; c < cycles; ++c)
    {
// 前半个时间片组
#pragma omp parallel num_threads(threads)
        {
            int th = omp_get_thread_num();
            if (th % 2 == 0)
            { // 仿真神经元网络
                int net_id = th / 2;
                CUDACHECK(cudaSetDevice(mapper[net_id]));
                GSubNet *gnet = multinet.gnets[net_id];
                GSubNet *cnet = multinet.cnets[net_id];
                manager.clearBuffer(net_id, 0,manager.sim_streams[net_id]);
#pragma omp barrier
                int i;
                for (i = step; i < step + pre_half_group_size && i < sim_time_steps; ++i)
                {
                    mgsim_step_sparse(gnet, cnet, i, blocksize, manager, multinet.gaddrs[net_id], 0,manager.sim_streams[net_id]);
                }
#pragma omp barrier
                manager.syncBufferSizeList(net_id, 0,manager.sim_streams[net_id]);
            }
            else
            { // 同步脉冲
                int net_id = th / 2;
                CUDACHECK(cudaSetDevice(mapper[net_id]));
                GSubNet *gnet = multinet.gnets[net_id];
                GSubNet *cnet = multinet.cnets[net_id];
#pragma omp barrier
                /// 脉冲同步分为两步：脉冲数据传输和脉冲数据收集
                /// 脉冲数据传输
                trans_spikes_sparse(cnet, manager, mapper, 1, manager.trans_streams[net_id]);
#pragma omp barrier
                /// 脉冲数据收集
                recvs_spikes_sparse(gnet, cnet, blocksize, manager, 1, manager.recv_streams[net_id]);
            }
        }
        step += pre_half_group_size;

// 后半个时间片组
#pragma omp parallel num_threads(threads)
        {
            int th = omp_get_thread_num();
            if (th % 2 == 0)
            { // 仿真神经元网络
                int net_id = th / 2;
                CUDACHECK(cudaSetDevice(mapper[net_id]));
                GSubNet *gnet = multinet.gnets[net_id];
                GSubNet *cnet = multinet.cnets[net_id];
                manager.clearBuffer(net_id, 1,manager.sim_streams[net_id]);
#pragma omp barrier

                int i;
                for (i = step; i < step + post_half_group_size && i < sim_time_steps; ++i)
                {
                    mgsim_step_sparse(gnet, cnet, i, blocksize, manager, multinet.gaddrs[net_id], 1,manager.sim_streams[net_id]);
                }
#pragma omp barrier
                manager.syncBufferSizeList(net_id, 1,manager.sim_streams[net_id]);
            }
            else
            { // 同步脉冲
                int net_id = th / 2;
                CUDACHECK(cudaSetDevice(mapper[net_id]));
                GSubNet *gnet = multinet.gnets[net_id];
                GSubNet *cnet = multinet.cnets[net_id];
#pragma omp barrier
                /// 脉冲同步分为两步：脉冲数据传输和脉冲数据收集
                /// 脉冲数据传输
                trans_spikes_sparse(cnet, manager, mapper, 0, manager.trans_streams[net_id]);

#pragma omp barrier
                /// 脉冲数据收集
                recvs_spikes_sparse(gnet, cnet, blocksize, manager, 0, manager.recv_streams[net_id]);
            }
        }
        step += post_half_group_size;
    }
    sim_time = sim.stop();
    if (Config::FIRE_CHECK)
    {
        get_firing_rate(time);
    }
}

void MGBrain::MGSimulator::simulate_test_d(real time)
{
    sim_time = 0.0;
    int net_num = multinet.cnets.size();
    int pre_half_group_size = multinet.min_delay / 2;
    int post_half_group_size = multinet.min_delay - pre_half_group_size;
    int sim_time_steps = time / Config::STEP;

    int cycles = sim_time_steps / multinet.min_delay + (sim_time_steps % multinet.min_delay == 0);
    int blocksize = multinet.blocksize;
    DenseBufferManager &manager = multinet.managerd;
    int threads = net_num * 2;
    int step = 0;
    std::vector<float> times(7);
    timer sim;
    for (int c = 0; c < cycles; ++c)
    {
// 前半个时间片组
#pragma omp parallel num_threads(threads)
        {
            int th = omp_get_thread_num();
            int net_id = th / 2;
            CUDACHECK(cudaSetDevice(mapper[net_id]));
            GSubNet *gnet = multinet.gnets[net_id];
            GSubNet *cnet = multinet.cnets[net_id];
#pragma omp barrier
            if (th % 2 == 0)
            { // 仿真神经元网络
                for (int i = step; i < step + pre_half_group_size && i < sim_time_steps; ++i)
                {
                    mgsim_step_dense(gnet, cnet, i, blocksize, manager, multinet.gaddrs[net_id], 0);
                }
#pragma omp barrier
                CUDACHECK(cudaStreamSynchronize((cudaStream_t)cudaStreamDefault));
            }
            else
            { // 同步脉冲
                trans_spikes_dense(cnet, manager, 1, manager.trans_streams[net_id], mapper, multinet.max_delay);
#pragma omp barrier
                recvs_spikes_dense(gnet, cnet, step, blocksize, manager, 1, manager.recv_streams[net_id]);
            }
        }
        step += pre_half_group_size;

// 后半个时间片组
#pragma omp parallel num_threads(threads)
        {
            int th = omp_get_thread_num();
            int net_id = th / 2;
            CUDACHECK(cudaSetDevice(mapper[net_id]));
            GSubNet *gnet = multinet.gnets[net_id];
            GSubNet *cnet = multinet.cnets[net_id];
#pragma omp barrier
            if (th % 2 == 0)
            { // 仿真神经元网络
                for (int i = step; i < step + post_half_group_size && i < sim_time_steps; ++i)
                {
                    mgsim_step_dense(gnet, cnet, i, blocksize, manager, multinet.gaddrs[net_id], 1);
                }
#pragma omp barrier
                CUDACHECK(cudaStreamSynchronize((cudaStream_t)cudaStreamDefault));
            }
            else
            { // 同步脉冲
                trans_spikes_dense(cnet, manager, 0, manager.trans_streams[net_id], mapper, multinet.max_delay);
#pragma omp barrier
                recvs_spikes_dense(gnet, cnet, step, blocksize, manager, 0, manager.recv_streams[net_id]);
            }
        }
        step += post_half_group_size;
    }
    sim_time = sim.stop();
    if (Config::FIRE_CHECK)
    {
        get_firing_rate(time);
    }
}

void MGBrain::MGSimulator::simulate_sparse(real time)
{
    sim_time = 0.0;
    int net_num = multinet.cnets.size();
    int min_delay = multinet.min_delay;
    int pre_half_group_size = multinet.min_delay / 2;
    int post_half_group_size = multinet.min_delay - pre_half_group_size;
    int sim_time_steps =static_cast<int>( std::round(time / Config::STEP));
    int cycles = sim_time_steps / multinet.min_delay + (sim_time_steps % multinet.min_delay == 0);
    int blocksize = multinet.blocksize;
    BufferManager &manager = multinet.manager;
    int threads = net_num;
    std::vector<float> times(7);
    timer sim;
#pragma omp parallel num_threads(threads)
    {
        int step = 0;
        int net_id = omp_get_thread_num();
        CUDACHECK(cudaSetDevice(mapper[net_id]));
        GSubNet *gnet = multinet.gnets[net_id];
        GSubNet *cnet = multinet.cnets[net_id];
        #pragma omp barrier
        for (int c = 0; c < cycles; ++c)
        {
            //清除缓存
            manager.clearBuffer(net_id, 0,manager.sim_streams[net_id]);
            //仿真网络
            for (int i = step; i < step + pre_half_group_size && i < sim_time_steps; ++i)
            {
                mgsim_step_sparse(gnet, cnet, i, blocksize, manager, multinet.gaddrs[net_id], 0, manager.sim_streams[net_id]);
            }
            manager.syncBufferSizeList(net_id, 0,manager.sim_streams[net_id]);
            /// 脉冲数据传输
            trans_spikes_sparse(cnet, manager, mapper, 1, manager.trans_streams[net_id]);
            /// 脉冲数据收集
            recvs_spikes_sparse(gnet, cnet, blocksize, manager, 1, manager.recv_streams[net_id]);
            CUDACHECK(cudaStreamSynchronize(manager.trans_streams[net_id]));
            CUDACHECK(cudaStreamSynchronize(manager.sim_streams[net_id]));
            #pragma omp barrier
            manager.clearBuffer(net_id, 1,manager.sim_streams[net_id]);
            
            for (int i = step+pre_half_group_size; i < step + min_delay && i < sim_time_steps; ++i)
            {
                mgsim_step_sparse(gnet, cnet, i, blocksize, manager, multinet.gaddrs[net_id], 1, manager.sim_streams[net_id]);
            }
            manager.syncBufferSizeList(net_id, 1,manager.sim_streams[net_id]);
            /// 脉冲数据传输
            trans_spikes_sparse(cnet, manager, mapper, 0, manager.trans_streams[net_id]);
            /// 脉冲数据收集
            recvs_spikes_sparse(gnet, cnet, blocksize, manager, 0, manager.recv_streams[net_id]);

            CUDACHECK(cudaStreamSynchronize(manager.trans_streams[net_id]));
            CUDACHECK(cudaStreamSynchronize(manager.sim_streams[net_id]));
            step += min_delay;
            #pragma omp barrier
        }
        
    }
    sim_time = sim.stop();
    if (Config::FIRE_CHECK)
    {
        get_firing_rate(time);
    }
}

void MGBrain::MGSimulator::simulate_dense(real time)
{
    sim_time = 0.0;
    int net_num = multinet.cnets.size();
    int min_delay = multinet.min_delay;
    int pre_half_group_size = multinet.min_delay / 2;
    int post_half_group_size = multinet.min_delay - pre_half_group_size;
    int sim_time_steps = static_cast<int>( std::round(time / Config::STEP));
    int cycles = sim_time_steps / multinet.min_delay + (sim_time_steps % multinet.min_delay == 0);
    int blocksize = multinet.blocksize;
    DenseBufferManager &manager = multinet.managerd;
    int threads = net_num;
    std::vector<float> times(7);
    timer sim;
#pragma omp parallel num_threads(threads)
    {
        int step = 0;
        int net_id = omp_get_thread_num();
        CUDACHECK(cudaSetDevice(mapper[net_id]));
        GSubNet *gnet = multinet.gnets[net_id];
        GSubNet *cnet = multinet.cnets[net_id];
        #pragma omp barrier
        for (int c = 0; c < cycles; ++c)
        {

            // 接收域外脉冲
            recvs_spikes_dense(gnet, cnet, step, blocksize, manager, 0, manager.recv_streams[net_id]);

            // 仿真时间片组
            for (int i = step; i < step + pre_half_group_size && i < sim_time_steps; ++i)
            {
                mgsim_step_dense(gnet, cnet, i, blocksize, manager, multinet.gaddrs[net_id], 0);
            }
            // 传输设备间脉冲
            trans_spikes_dense(cnet, manager, 1, manager.trans_streams[net_id], mapper, multinet.max_delay);
            CUDACHECK(cudaStreamSynchronize(manager.trans_streams[net_id]));
            CUDACHECK(cudaStreamSynchronize(manager.sim_streams[net_id]));
            #pragma omp barrier

            // 接收域外脉冲
            recvs_spikes_dense(gnet, cnet, step, blocksize, manager, 1, manager.recv_streams[net_id]);

            // 仿真时间片组
            for (int i = step + pre_half_group_size; i < step + min_delay && i < sim_time_steps; ++i)
            {
                mgsim_step_dense(gnet, cnet, i, blocksize, manager, multinet.gaddrs[net_id], 1);
            }
            // 传输设备间脉冲
            trans_spikes_dense(cnet, manager, 0, manager.trans_streams[net_id], mapper, multinet.max_delay);
            CUDACHECK(cudaStreamSynchronize(manager.trans_streams[net_id]));
            CUDACHECK(cudaStreamSynchronize(manager.sim_streams[net_id]));
            step += min_delay;
            #pragma omp barrier
        }
    }
    sim_time = sim.stop();
    if (Config::FIRE_CHECK)
    {
        get_firing_rate(time);
    }
}

void MGBrain::MGSimulator::simulate(real time)
{
    if (Config::DENSE_SPIKE)
    {
        if(Config::SEQUENCE){
            simulate_seq_dense(time);
        }else{
            simulate_dense(time);
        }
        
    }
    else
    {
        if(Config::SEQUENCE){
            simulate_seq_sparse(time);
        }else{
            simulate_sparse(time);
        }
        
    }
}
double MGBrain::MGSimulator::get_time()
{
    return sim_time;
}