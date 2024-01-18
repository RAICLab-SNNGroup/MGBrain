#include "mgnet.h"
/// 数据操作部分
namespace MGBrain
{
    void init_gsubnet_neus(GSubNet *cnet, size_t num, int max_delay)
    {
        cnet->neus.size = num;
        cnet->neus.ids = new int[num]();
        cnet->neus.V_m = new real[num]();
        cnet->neus.fired = new bool[num]();
        cnet->neus.fire_cnt = new int[num]();
        cnet->neus.I_buffer_exc = new real[num * max_delay]();
        cnet->neus.I_buffer_inh = new real[num * max_delay]();
        cnet->neus.I_exc = new real[num]();
        cnet->neus.I_inh = new real[num]();
        cnet->neus.last_fired = new int[num]();
        cnet->neus.Refrac_state = new int[num]();
        cnet->neus.source = new bool[num]();
        cnet->neus.type = new int[num]();
        cnet->neus.rate = new real[num]();
        // cnet->neus.state = new curandState[num]();
    }
    void init_gsubnet_syns(GSubNet *cnet, size_t num)
    {
        cnet->syns.size = num;
        cnet->syns.src = new int[num]();
        cnet->syns.tar = new int[num]();
        cnet->syns.delay = new int[num]();
        cnet->syns.weight = new real[num]();
    }
    void init_gsubnet_adjs(GSubNet *cnet, size_t net_axon_size, size_t net_dend_size)
    {
        int num = cnet->neus.size;

        cnet->adjs.axon_offs = new size_t[num + 1]();
        cnet->adjs.axon_refs = new size_t[net_axon_size]();

        cnet->adjs.dend_offs = new size_t[num + 1]();
        cnet->adjs.dend_refs = new size_t[net_dend_size]();
    }
    void free_gsubnet(GSubNet *cnet)
    {
        /// delete 神经元内存空间
        delete[] cnet->neus.ids;
        delete[] cnet->neus.V_m;
        delete[] cnet->neus.fired;
        delete[] cnet->neus.fire_cnt;
        delete[] cnet->neus.I_buffer_exc;
        delete[] cnet->neus.I_buffer_inh;
        delete[] cnet->neus.I_exc;
        delete[] cnet->neus.I_inh;
        delete[] cnet->neus.last_fired;
        delete[] cnet->neus.Refrac_state;
        delete[] cnet->neus.source;
        delete[] cnet->neus.type;
        delete[] cnet->neus.rate;
        // delete[] cnet->neus.state;
        /// delete 内突触内存空间
        delete[] cnet->syns.src;
        delete[] cnet->syns.tar;
        delete[] cnet->syns.delay;
        delete[] cnet->syns.weight;

        delete cnet;
    }
    void free_gsubnet_gpu(GSubNet *gnet)
    {
        GSubNet *tmp = toCPU(gnet, 1);
        /// 释放神经元显存空间
        gpuFree(tmp->neus.ids);
        gpuFree(tmp->neus.V_m);
        gpuFree(tmp->neus.fired);
        gpuFree(tmp->neus.last_fired);
        gpuFree(tmp->neus.fire_cnt);
        gpuFree(tmp->neus.I_buffer_exc);
        gpuFree(tmp->neus.I_buffer_inh);
        gpuFree(tmp->neus.I_exc);
        gpuFree(tmp->neus.I_inh);
        gpuFree(tmp->neus.Refrac_state);
        gpuFree(tmp->neus.source);
        gpuFree(tmp->neus.type);
        gpuFree(tmp->neus.rate);
        gpuFree(tmp->neus.state);
        /// 释放内部突触显存空间
        gpuFree(tmp->syns.src);
        gpuFree(tmp->syns.tar);
        gpuFree(tmp->syns.weight);
        gpuFree(tmp->syns.delay);
        delete[] tmp;
        gpuFree(gnet);
    }
    GSubNet *copy_subnet_gpu(GSubNet *cnet, int max_delay, CNetAddrs &caddrs)
    {
        GSubNet *tmp = new GSubNet;
        int netid = cnet->id;
        tmp->id = netid;
        // tmp->max_delay=cnet->max_delay;
        tmp->neus.size = cnet->neus.size;
        tmp->syns.size = cnet->syns.size;
        // 拷贝神经元数据
        int num = tmp->neus.size;
        tmp->neus.ids = toGPU(cnet->neus.ids, num);
        tmp->neus.V_m = toGPU(cnet->neus.V_m, num); // lif
        // for(int i=0;i<10;i++){
        //     printf("vm:%f\n",cnet->neus.V_m[i]);
        // }
        tmp->neus.I_exc = toGPU(cnet->neus.I_exc, num);
        tmp->neus.I_inh = toGPU(cnet->neus.I_inh, num);
        tmp->neus.I_buffer_exc = toGPU(cnet->neus.I_buffer_exc, num * max_delay);
        tmp->neus.I_buffer_inh = toGPU(cnet->neus.I_buffer_inh, num * max_delay);
        tmp->neus.fired = toGPU(cnet->neus.fired, num);
        tmp->neus.fire_cnt = toGPU(cnet->neus.fire_cnt, num);
        tmp->neus.last_fired = toGPU(cnet->neus.last_fired, num);
        tmp->neus.Refrac_state = toGPU(cnet->neus.Refrac_state, num); // lif
        tmp->neus.rate = toGPU(cnet->neus.rate, num);                 // poisson
        tmp->neus.source = toGPU(cnet->neus.source, num);
        tmp->neus.type = toGPU(cnet->neus.type, num);

        cudaMalloc(&tmp->neus.state, num * sizeof(curandState));
        init_state(num, tmp->neus.state);

        // 拷贝邻接信息
        tmp->adjs.axon_offs = toGPU(cnet->adjs.axon_offs, num + 1);
        tmp->adjs.dend_offs = toGPU(cnet->adjs.dend_offs, num + 1);
        size_t axon_size = cnet->adjs.axon_offs[num];
        size_t dend_size = cnet->adjs.dend_offs[num];
        tmp->adjs.axon_refs = toGPU(cnet->adjs.axon_refs, axon_size);
        tmp->adjs.dend_refs = toGPU(cnet->adjs.dend_refs, dend_size);

        // 拷贝突触
        num = cnet->syns.size;
        tmp->syns.size = num;
        tmp->syns.src = toGPU(cnet->syns.src, num);
        tmp->syns.tar = toGPU(cnet->syns.tar, num);
        tmp->syns.weight = toGPU(cnet->syns.weight, num);
        tmp->syns.delay = toGPU(cnet->syns.delay, num);

        // 记录地址信息
        caddrs.clast_fired_addrs[netid] = tmp->neus.last_fired;
        caddrs.csyn_src_addrs[netid] = tmp->syns.src;
        caddrs.csyn_weight_addrs[netid] = tmp->syns.weight;

        GSubNet *gnet = toGPU(tmp, 1);
        delete tmp;
        return gnet;
    }

    size_t get_subnet_firecnt(GSubNet *cnet, GSubNet *gnet)
    {
        GSubNet *tmp = toCPU(gnet, 1);
        toCPU(tmp->neus.fire_cnt, cnet->neus.fire_cnt, tmp->neus.size);
        size_t sum = 0;
        for (int i = 0; i < tmp->neus.size; i++)
        {
            sum += cnet->neus.fire_cnt[i];
        }
        return sum;
    }
    SpikeBuffer *init_buffer_gpu(int size, SpikeBuffer &cbuffer)
    {
        SpikeBuffer *tmp = new SpikeBuffer;
        CUDACHECK(cudaMalloc((void **)&(tmp->spikes), sizeof(real) * size));
        // printf("spikes:%p\n",tmp->spikes);
        cbuffer.spikes = tmp->spikes;
        CUDACHECK(cudaMemset(tmp->spikes, 0.0f, sizeof(real) * size));

        CUDACHECK(cudaMalloc((void **)&(tmp->targets), sizeof(int) * size));
        cbuffer.targets = tmp->targets;
        CUDACHECK(cudaMemset(tmp->targets, 0, sizeof(int) * size));

        CUDACHECK(cudaMalloc((void **)&(tmp->times), sizeof(int) * size));
        cbuffer.times = tmp->times;
        CUDACHECK(cudaMemset(tmp->times, 0, sizeof(int) * size));

        SpikeBuffer *gbuffer = toGPU(tmp, 1);
        delete tmp;
        return gbuffer;
    }
    void free_buffer_gpu(SpikeBuffer *gbuffer)
    {
        SpikeBuffer *tmp = toCPU(gbuffer, 1);
        gpuFree(tmp->spikes);
        gpuFree(tmp->targets);
        gpuFree(tmp->times);
        delete tmp;
        gpuFree(gbuffer);
    }
    int *init_buffer_size_list_gpu(int size)
    {
        int *tmp;
        CUDACHECK(cudaMalloc((void **)&(tmp), sizeof(int) * size));
        CUDACHECK(cudaMemset(tmp, 0, sizeof(int) * size));
        return tmp;
    }
    SpikeBuffer **copy_buffers_gpu(std::vector<SpikeBuffer *> &cbuffers)
    {
        SpikeBuffer **tmp = new SpikeBuffer *[cbuffers.size()];
        for (int i = 0; i < cbuffers.size(); i++)
        {
            tmp[i] = cbuffers[i];
        }
        SpikeBuffer **gbuffers = toGPU(tmp, cbuffers.size());
        delete[] tmp;
        return gbuffers;
    }

    SpikeDenseBuffer *init_dense_buffer_gpu(int size, int max_delay, SpikeDenseBuffer *cbuffer, std::vector<int> &targets)
    {
        int max_num = 0;
        int min_num = INT32_MAX / 2;
        if (targets.size() == 0)
            return nullptr;
        for (int n = 0; n < targets.size(); n++)
        {
            max_num = std::max(max_num, targets[n]);
            min_num = std::min(min_num, targets[n]);
        }
        int mapper_size = max_num - min_num + 1;
        SpikeDenseBuffer *tmp = new SpikeDenseBuffer;
        CUDACHECK(cudaMalloc((void **)&(tmp->buffer_exc), sizeof(real) * size * max_delay));
        // printf("spikes:%p\n",tmp->spikes);
        cbuffer->buffer_exc = tmp->buffer_exc;
        CUDACHECK(cudaMemset(tmp->buffer_exc, 0.0f, sizeof(real) * size * max_delay));

        CUDACHECK(cudaMalloc((void **)&(tmp->buffer_inh), sizeof(real) * size * max_delay));
        // printf("spikes:%p\n",tmp->spikes);
        cbuffer->buffer_inh = tmp->buffer_inh;
        CUDACHECK(cudaMemset(tmp->buffer_inh, 0.0f, sizeof(real) * size * max_delay));

        std::vector<int> nmapper(mapper_size);
        for (int n = 0; n < targets.size(); n++)
        {
            nmapper[targets[n] - min_num] = n;
        }
        tmp->start = min_num;
        tmp->end = max_num;
        tmp->neusize = size;
        tmp->mapper = toGPU(nmapper.data(), mapper_size);
        cbuffer->mapper = tmp->mapper;
        tmp->targets = toGPU(targets.data(), targets.size());
        cbuffer->targets = tmp->targets;
        auto gbuffer = toGPU(tmp, 1);
        delete tmp;
        return gbuffer;
    }
    void free_dense_buffer_gpu(SpikeDenseBuffer *gbuffer)
    {
        SpikeDenseBuffer *tmp = toCPU(gbuffer, 1);
        gpuFree(tmp->buffer_exc);
        gpuFree(tmp->buffer_inh);
        gpuFree(tmp->targets);
        gpuFree(tmp->mapper);
        delete tmp;
        gpuFree(gbuffer);
    }
    SpikeDenseBuffer **copy_dense_buffers_gpu(std::vector<SpikeDenseBuffer *> &cbuffers)
    {
        SpikeDenseBuffer **tmp = new SpikeDenseBuffer *[cbuffers.size()];
        for (int i = 0; i < cbuffers.size(); i++)
        {
            tmp[i] = cbuffers[i];
            // std::cout<<"addr:"<<cbuffers[i]<<std::endl;
        }
        SpikeDenseBuffer **gbuffers = toGPU(tmp, cbuffers.size());
        delete[] tmp;
        return gbuffers;
    }

    /// NetAddrs 相关 GPU数据传输

    void copy_netaddrs_gpu(GNetAddrs &gaddrs, CNetAddrs &caddrs)
    {
        gaddrs.glast_fired_addrs = toGPU(caddrs.clast_fired_addrs.data(), caddrs.clast_fired_addrs.size());
        gaddrs.gsyn_src_addrs = toGPU(caddrs.csyn_src_addrs.data(), caddrs.csyn_src_addrs.size());
        gaddrs.gsyn_weight_addrs = toGPU(caddrs.csyn_weight_addrs.data(), caddrs.csyn_weight_addrs.size());
    }
    void free_netaddrs_gpu(GNetAddrs &gaddrs)
    {
        gpuFree(gaddrs.glast_fired_addrs);
        gpuFree(gaddrs.gsyn_src_addrs);
        gpuFree(gaddrs.gsyn_weight_addrs);
    }
    void copy_subnet_cpu(GSubNet *gnet, GSubNet *cnet)
    {
        // 神经元信息
        int num = cnet->neus.size;
        toCPU(gnet->neus.I_exc, cnet->neus.I_exc, num);
        toCPU(gnet->neus.I_inh, cnet->neus.I_inh, num);
        toCPU(gnet->neus.I_buffer_exc, cnet->neus.I_buffer_exc, num);
        toCPU(gnet->neus.I_buffer_inh, cnet->neus.I_buffer_inh, num);
        toCPU(gnet->neus.fired, cnet->neus.fired, num);
        toCPU(gnet->neus.fire_cnt, cnet->neus.fire_cnt, num);
        toCPU(gnet->neus.last_fired, cnet->neus.last_fired, num);
        toCPU(gnet->neus.Refrac_state, cnet->neus.Refrac_state, num);
        toCPU(gnet->neus.rate, cnet->neus.rate, num);
        toCPU(gnet->neus.source, cnet->neus.source, num);
        toCPU(gnet->neus.type, cnet->neus.type, num);
        // 突触信息
        toCPU(gnet->syns.weight, cnet->syns.weight, cnet->syns.size);
    }

    // GSubNet *tmp = new GSubNet;
    //     int netid = cnet->id;
    //     tmp->id = netid;
    //     // tmp->max_delay=cnet->max_delay;
    //     tmp->neus_size = num;
    //     tmp->syns_size = cnet->syns_size;
    //     // 拷贝神经元数据
    //     int num = tmp->neus_size;
    //     tmp->neus.ids = toGPU(cnet->neus.ids, num);
    //     tmp->neus.V_m = toGPU(cnet->neus.V_m, num); // lif
    //     // for(int i=0;i<10;i++){
    //     //     printf("vm:%f\n",cnet->neus.V_m[i]);
    //     // }
    //     tmp->neus.I_exc = toGPU(cnet->neus.I_exc, num);
    //     tmp->neus.I_inh = toGPU(cnet->neus.I_inh, num);
    //     tmp->neus.I_buffer_exc = toGPU(cnet->neus.I_buffer_exc, num * max_delay);
    //     tmp->neus.I_buffer_inh = toGPU(cnet->neus.I_buffer_inh, num * max_delay);
    //     tmp->neus.fired = toGPU(cnet->neus.fired, num);
    //     tmp->neus.fire_cnt = toGPU(cnet->neus.fire_cnt, num);
    //     tmp->neus.last_fired = toGPU(cnet->neus.last_fired, num);
    //     tmp->neus.Refrac_state = toGPU(cnet->neus.Refrac_state, num); // lif
    //     tmp->neus.rate = toGPU(cnet->neus.rate, num);                 // poisson
    //     tmp->neus.poisson = toGPU(cnet->neus.poisson, num);
    //     tmp->neus.type = toGPU(cnet->neus.type, num);

    //     cudaMalloc(&tmp->neus.state, num * sizeof(curandState));
    //     int blocksize = 1024;
    //     srand(time(0));
    //     mgsim_init_core<<<num / blocksize + 1, blocksize>>>(num, tmp->neus.state, rand());
    //     // tmp->neus.state = toGPU(cnet->neus.state, num);

    //     // 拷贝邻接信息
    //     // int sum=0;
    //     // for(int i=0;i<num;i++){
    //     //     int axon=cnet->adjs.axon_offs[i+1]-cnet->adjs.axon_offs[i];
    //     //     sum+=axon;
    //     //     printf("offs:%d\n",axon);
    //     // }
    //     // printf("offsum:%d\n",sum);
    //     tmp->adjs.axon_offs = toGPU(cnet->adjs.axon_offs, num + 1);
    //     tmp->adjs.dend_offs = toGPU(cnet->adjs.dend_offs, num + 1);
    //     size_t axon_size = cnet->adjs.axon_offs[num];
    //     size_t dend_size = cnet->adjs.dend_offs[num];
    //     tmp->adjs.axon_refs = toGPU(cnet->adjs.axon_refs, axon_size);
    //     tmp->adjs.dend_refs = toGPU(cnet->adjs.dend_refs, dend_size);

    //     // 拷贝突触
    //     num = cnet->syns_size;
    //     tmp->syns.src = toGPU(cnet->syns.src, num);
    //     tmp->syns.tar = toGPU(cnet->syns.tar, num);
    //     tmp->syns.weight = toGPU(cnet->syns.weight, num);
    //     tmp->syns.delay = toGPU(cnet->syns.delay, num);

    //     // 记录地址信息
    //     caddrs.clast_fired_addrs[netid] = tmp->neus.last_fired;
    //     caddrs.csyn_src_addrs[netid] = tmp->syns.src;
    //     caddrs.csyn_weight_addrs[netid] = tmp->syns.weight;

    //     GSubNet *gnet = toGPU(tmp, 1);
    //     delete tmp;
    //     return gnet;

};
