#include "esamgnet.cuh"
namespace MGBrain
{
    // LIF神经元使用常量
    __constant__ real _P22 = 0.90483743;
    __constant__ real _P11exc = 0.951229453;
    __constant__ real _P11inh = 0.967216074;
    __constant__ real _P21exc = 0.0799999982;
    __constant__ real _P21inh = 0.0599999987;
    __constant__ real V_rest = 0;
    __constant__ real V_reset = 0;
    __constant__ real C_m = 0.25;
    __constant__ real Tau_m = 10.0;
    __constant__ real V_thresh = 15;
    __constant__ real I_offset = 0;
    __constant__ real Refrac_step = 4;

    // STDP突触使用常量
    __constant__ real A_LTP = 0.1;
    __constant__ real A_LTD = -0.01;
    __constant__ real TAU_LTP = 17;
    __constant__ real TAU_LTD = 34;
    __constant__ real W_max = 40;
    __constant__ real W_min = 0;

    // 神经网络仿真使用常量
    __constant__ int MAX_DELAY = 10;
    __constant__ real dt = 0.0001;

    __device__ inline size_t getSynId(size_t ref)
    {
        return ref & (0x00ffffffffffffff);
    }
    __device__ inline int getSynTarZone(size_t ref)
    {
        return ref & (0xff00000000000000) >> 56;
    }
    __device__ inline real getInput(real *buffer, int id, int step)
    {
        int loc = id * MAX_DELAY + step % MAX_DELAY;
        real res = buffer[loc];
        buffer[loc] = 0.0;
        return res;
    }
    __device__ inline void pushInnerSpike(real *buffer, int nid, int time, real value)
    {
        atomicAdd(&buffer[nid * MAX_DELAY + time % MAX_DELAY], value);
    }

    // __device__ inline short getSynTar(size_t ref,){}
    __global__ void mgsim_neus_core(GSubNet *net, int step)
    {
        int ref = blockIdx.x * blockDim.x + threadIdx.x;
        if (net->neus.type[ref] == NeuronType::POISSON)
        { /// 仿真泊松神经元
            float rand = curand_uniform(&(net->neus.state[ref]));
            net->neus.Fired[ref] = net->neus.rate[ref] > rand;
            if (net->neus.Fired[ref])
            {
                net->neus.Fire_cnt[ref]++;
                net->neus.Last_fired[ref] = step;
            }
        }
        else if (net->neus.type[ref] == NeuronType::LIF)
        { /// 仿真LIF神经元
            net->neus.Fired[ref] = false;
            if (net->neus.Refrac_state[ref] > 0)
            {
                --net->neus.Refrac_state[ref];
            }
            else
            {
                net->neus.V_m[ref] = _P22 * net->neus.V_m[ref] + net->neus.I_exc[ref] * _P21exc + net->neus.I_inh[ref] * _P21inh;
                net->neus.V_m[ref] += (1 - _P22) * (I_offset * Tau_m / C_m + V_rest);
                net->neus.I_exc[ref] *= _P11exc;
                net->neus.I_inh[ref] *= _P11inh;
                if (net->neus.V_m[ref] >= V_thresh)
                {
                    net->neus.Fired[ref] = true;
                    net->neus.Fire_cnt[ref]++;
                    net->neus.V_m[ref] = V_reset;
                    net->neus.Last_fired[ref] = step;
                    net->neus.Refrac_state[ref] = Refrac_step;
                }
                else
                {
                    net->neus.I_exc[ref] += getInput(net->neus.I_buffer_exc, ref, step);
                    net->neus.I_inh[ref] += getInput(net->neus.I_buffer_inh, ref, step);
                }
            }
        }
        else if (net->neus.type[ref] == NeuronType::LIF0)
        {
            net->neus.Fired[ref] = false;
            if (net->neus.Refrac_state[ref] > 0)
            {
                --net->neus.Refrac_state[ref];
            }
            else
            {
                real I = getInput(net->neus.I_buffer_exc, ref, step) + getInput(net->neus.I_buffer_inh, ref, step);
                net->neus.V_m[ref] += (dt / Tau_m) * (I * Tau_m / C_m - net->neus.V_m[ref]);
                if (net->neus.V_m[ref] >= V_thresh)
                {
                    net->neus.Fired[ref] = true;
                    net->neus.Fire_cnt[ref]++;
                    net->neus.V_m[ref] = V_reset;
                    net->neus.Last_fired[ref] = step;
                    net->neus.Refrac_state[ref] = Refrac_step;
                }
            }
        }
        else if (net->neus.type[ref] == NeuronType::LIFE)
        {
        }
    }
    __global__ void mgsimSyns(GSubNet *net, int step)
    {
        int ref = blockIdx.x * blockDim.x + threadIdx.x;
        int src = net->syns.src[ref];
        int tar = net->syns.tar[ref];
        real weight = net->syns.weight[ref];
        int delay = net->syns.delay[ref];
        // 发放脉冲
        if (net->neus.Fired[src])
        {
            if (weight > 0)
                pushInnerSpike(net->neus.I_buffer_exc, tar, (step + delay) % MAX_DELAY, weight);
            else
                pushInnerSpike(net->neus.I_buffer_inh, tar, (step + delay) % MAX_DELAY, weight);
        }
    }

    __global__ void mgsimOuts(GSubNet *net, int step, int index, real *spikes, int *targets, int *times, int *out_buffer_sizes)
    {
        int ref = blockIdx.x * blockDim.x + threadIdx.x;
        int src = net->outs[index].block.src[ref];
        if (net->neus.Fired[src])
        {
            int tar = net->outs[index].block.tar[ref];
            real weight = net->outs[index].block.weight[ref];
            int delay = net->outs[index].block.delay[ref];
            int loc = atomicAdd(&(out_buffer_sizes[index]), 1);
            spikes[loc] = weight;
            targets[loc] = tar;
            times[loc] = step + delay;
        }
    }
    __global__ void mgsim_recv_core(GSubNet *net, int index, SpikeBuffer **buffers)
    {
        int ref = blockIdx.x * blockDim.x + threadIdx.x;
        int tar = buffers[index]->targets[ref];
        int time = buffers[index]->times[ref];
        int spike = buffers[index]->spikes[ref];
        if (spike > 0)
        {
            pushInnerSpike(net->neus.I_buffer_exc, tar, time, spike);
        }
        else
        {
            pushInnerSpike(net->neus.I_buffer_inh, tar, time, spike);
        }
    }
    /// @brief todo:根据神经元不应期特征进行重排布
    /// @param net
    /// @param step
    /// @return
    __global__ void mgsimReloc(GSubNet *net, int step)
    {
    }
    inline void transfer(SpikeBuffer *out_buffer, int out_device, SpikeBuffer *in_buffer, int in_device, int size, cudaStream_t stream)
    {
        if (Config::SINGLE_GPU)
        {
            CUDACHECK(cudaMemcpy(in_buffer->spikes, out_buffer->spikes, size, cudaMemcpyDeviceToDevice));
            CUDACHECK(cudaMemcpy(in_buffer->targets, out_buffer->targets, size, cudaMemcpyDeviceToDevice));
            CUDACHECK(cudaMemcpy(in_buffer->times, out_buffer->times, size, cudaMemcpyDeviceToDevice));
        }
        else
        {
            CUDACHECK(cudaMemcpyPeerAsync(out_buffer->spikes, out_device, in_buffer->spikes, in_device, size, stream));
            CUDACHECK(cudaMemcpyPeerAsync(out_buffer->targets, out_device, in_buffer->targets, in_device, size, stream));
            CUDACHECK(cudaMemcpyPeerAsync(out_buffer->times, out_device, in_buffer->times, in_device, size, stream));
        }
        // std::cout<<"size:"<<size<<std::endl;
    }
    void mgsim_step(GSubNet *gnet, GSubNet *cnet, int step, int blocksize, BufferManager &manager, int turn)
    {
        /// 默认流用于仿真,其他流用于脉冲同步
        int neu_size = cnet->neus_size;
        int syn_size = cnet->syns_size;
        /// 仿真神经元

        mgsim_neus_core<<<neu_size / blocksize + 1, neu_size % blocksize>>>(gnet, step);
        /// 只同步默认流
        CUDACHECK(cudaStreamSynchronize((cudaStream_t)cudaStreamDefault));
        for (int i = 0; i < cnet->out_net_size; i++)
        {
            int out_size = cnet->outs[i].syn_size;
            /// todo: 这里可以设置多流并行
            auto buffer = manager.getCurOutGBuffer(cnet->id, i, turn);
            /// 仿真外部突触
            mgsimOuts<<<out_size / blocksize + 1, out_size % blocksize>>>(gnet, step, i, buffer->spikes, buffer->targets, buffer->times, manager.getGBufferSizeList(cnet->id, turn));
        }
        /// 仿真突触
        // SpikeBuffer **gbuffers = manager.getOutBuffer(cnet->id, turn);
        // int *gout_buffer_size_list = manager.getGBufferSizeList(cnet->id, turn);
        mgsimSyns<<<syn_size / blocksize + 1, syn_size % blocksize>>>(gnet, step);
        /// 只同步默认流
        CUDACHECK(cudaStreamSynchronize((cudaStream_t)cudaStreamDefault));
    }
    void trans_spikes(GSubNet *cnet, BufferManager &manager, int turn, std::vector<cudaStream_t> &trans_streams, std::vector<int> &mapper)
    {
        int id = cnet->id;
        /// 脉冲同步分为两步，脉冲数据传输和脉冲收集
        /// 脉冲数据传输，多流并行
        for (int k = 0; k < cnet->out_net_size; k++)
        {
            int src_device = mapper[id];
            int tar_device = mapper[manager.getTarNetId(id, k)];
            transfer(manager.getCurOutGBuffer(id, k, turn), src_device, manager.getTarInGBuffer(id, k, turn), tar_device, manager.getOutBufferSize(id, k, turn), trans_streams[k]);
        }
        /// 同步数据传输流
        for (int i = 0; i < trans_streams.size(); i++)
        {
            CUDACHECK(cudaStreamSynchronize(trans_streams[i]));
        }
    }
    void recvs_spikes(GSubNet *gnet, GSubNet *cnet, int blocksize, BufferManager &manager, int turn, std::vector<cudaStream_t> &recv_streams)
    {
        int id = cnet->id;
        /// 脉冲数据收集,多流并行
        for (int i = 0; i < cnet->in_net_size; i++)
        {
            int buffersize = manager.getInBufferSize(id, i, turn);
            auto buffer = manager.getInGGBuffer(cnet->id, turn);
            mgsim_recv_core<<<buffersize / blocksize + 1,  blocksize, 0, recv_streams[i]>>>(gnet, i, buffer);
        }
        /// 同步脉冲收集流
        for (int i = 0; i < recv_streams.size(); i++)
        {
            CUDACHECK(cudaStreamSynchronize(recv_streams[i]));
        }
    }
    void copy_consts_gpu(int max_delay, LIFConsts *lifconst, STDPConsts *stdpconst)
    {
        // CUDACHECK(cudaSetDevice(0));
        CUDACHECK(cudaMemcpyToSymbol(MAX_DELAY, &max_delay, sizeof(int)));
        if (lifconst != nullptr)
        {
            CUDACHECK(cudaMemcpyToSymbol(_P22, &(lifconst->P22), sizeof(real)));
            CUDACHECK(cudaMemcpyToSymbol(_P11exc, &(lifconst->P11exc), sizeof(real)));
            CUDACHECK(cudaMemcpyToSymbol(_P11inh, &(lifconst->P11inh), sizeof(real)));
            CUDACHECK(cudaMemcpyToSymbol(_P21exc, &(lifconst->P21exc), sizeof(real)));
            CUDACHECK(cudaMemcpyToSymbol(_P21inh, &(lifconst->P21inh), sizeof(real)));
            CUDACHECK(cudaMemcpyToSymbol(V_rest, &(lifconst->V_rest), sizeof(real)));
            CUDACHECK(cudaMemcpyToSymbol(V_reset, &(lifconst->V_reset), sizeof(real)));
            CUDACHECK(cudaMemcpyToSymbol(C_m, &(lifconst->C_m), sizeof(real)));
            CUDACHECK(cudaMemcpyToSymbol(Tau_m, &(lifconst->Tau_m), sizeof(real)));
            CUDACHECK(cudaMemcpyToSymbol(V_thresh, &(lifconst->V_thresh), sizeof(real)));
            CUDACHECK(cudaMemcpyToSymbol(I_offset, &(lifconst->I_offset), sizeof(real)));
            CUDACHECK(cudaMemcpyToSymbol(Refrac_step, &(lifconst->Refrac_step), sizeof(real)));
        }
        if (stdpconst != nullptr)
        {
            CUDACHECK(cudaMemcpyToSymbol(A_LTP, &(stdpconst->A_LTP), sizeof(real)));
            CUDACHECK(cudaMemcpyToSymbol(A_LTD, &(stdpconst->A_LTD), sizeof(real)));
            CUDACHECK(cudaMemcpyToSymbol(TAU_LTP, &(stdpconst->TAU_LTP), sizeof(real)));
            CUDACHECK(cudaMemcpyToSymbol(TAU_LTD, &(stdpconst->TAU_LTD), sizeof(real)));
            CUDACHECK(cudaMemcpyToSymbol(W_max, &(stdpconst->W_max), sizeof(real)));
            CUDACHECK(cudaMemcpyToSymbol(W_min, &(stdpconst->W_min), sizeof(real)));
        }
        CUDACHECK(cudaDeviceSynchronize());
    }
};
void MGBrain::init_gsubnet_neus(GSubNet *cnet, int max_delay)
{
    int num = cnet->neus_size;
    cnet->neus.ids = new int[num]();
    cnet->neus.V_m = new real[num]();
    cnet->neus.Fired = new bool[num]();
    cnet->neus.Fire_cnt = new int[num]();
    cnet->neus.I_buffer_exc = new real[num * max_delay]();
    cnet->neus.I_buffer_inh = new real[num * max_delay]();
    cnet->neus.I_exc = new real[num]();
    cnet->neus.I_inh = new real[num]();
    cnet->neus.Last_fired = new int[num]();
    cnet->neus.Refrac_state = new int[num]();
    cnet->neus.poisson = new bool[num]();
    cnet->neus.type = new int[num]();
    cnet->neus.rate = new real[num]();
    cnet->neus.state = new curandState[num]();
}
void MGBrain::init_gsubnet_syns(GSubNet *cnet)
{
    int num = cnet->syns_size;
    cnet->syns.src = new int[num]();
    cnet->syns.tar = new int[num]();
    cnet->syns.delay = new int[num]();
    cnet->syns.weight = new real[num]();
}
void MGBrain::initGSubNetOutSyns(SYNBlock *block, int num)
{
    block->src = new int[num];
    block->tar = new int[num];
    block->delay = new int[num];
    block->weight = new real[num];
}
void MGBrain::free_gsubnet(GSubNet *cnet)
{
    /// delete 神经元内存空间
    delete[] cnet->neus.ids;
    delete[] cnet->neus.V_m;
    delete[] cnet->neus.Fired;
    delete[] cnet->neus.Fire_cnt;
    delete[] cnet->neus.I_buffer_exc;
    delete[] cnet->neus.I_buffer_inh;
    delete[] cnet->neus.I_exc;
    delete[] cnet->neus.I_inh;
    delete[] cnet->neus.Last_fired;
    delete[] cnet->neus.Refrac_state;
    delete[] cnet->neus.poisson;
    delete[] cnet->neus.type;
    delete[] cnet->neus.rate;
    delete[] cnet->neus.state;
    /// delete 内突触内存空间
    delete[] cnet->syns.src;
    delete[] cnet->syns.tar;
    delete[] cnet->syns.delay;
    delete[] cnet->syns.weight;
    /// delete 外突触内存空间
    // for (int i = 0; i < cnet->outs_size; i++)
    // {
    //     delete[] cnet->outs[i].block.src;
    //     delete[] cnet->outs[i].block.tar;
    //     delete[] cnet->outs[i].block.delay;
    //     delete[] cnet->outs[i].block.weight;
    // }
    // delete[] cnet->outs;
    /// delete 网络内存空间
    delete cnet;
}
void MGBrain::free_gsubnet_gpu(GSubNet *gnet)
{
    GSubNet *tmp = toCPU(gnet, 1);
    /// 释放神经元显存空间
    gpuFree(tmp->neus.ids);
    gpuFree(tmp->neus.V_m);
    gpuFree(tmp->neus.Fired);
    gpuFree(tmp->neus.Last_fired);
    gpuFree(tmp->neus.Fire_cnt);
    gpuFree(tmp->neus.I_buffer_exc);
    gpuFree(tmp->neus.I_buffer_inh);
    gpuFree(tmp->neus.I_exc);
    gpuFree(tmp->neus.I_inh);
    gpuFree(tmp->neus.Refrac_state);
    gpuFree(tmp->neus.poisson);
    gpuFree(tmp->neus.type);
    gpuFree(tmp->neus.rate);
    gpuFree(tmp->neus.state);
    /// 释放内部突触显存空间
    gpuFree(tmp->syns.src);
    gpuFree(tmp->syns.tar);
    gpuFree(tmp->syns.weight);
    gpuFree(tmp->syns.delay);
    // OUTSYNBlock *outtmps = toCPU(tmp->outs, tmp->outs_size);
    // /// 释放外部突触显存空间
    // for (int i = 0; i < tmp->outs_size; i++)
    // {
    //     gpuFree(outtmps[i].block.src);
    //     gpuFree(outtmps[i].block.tar);
    //     gpuFree(outtmps[i].block.weight);
    //     gpuFree(outtmps[i].block.delay);
    // }
    // gpuFree(tmp->outs);
    // delete[] outtmps;
    delete[] tmp;
    gpuFree(gnet);
}
MGBrain::GSubNet *MGBrain::copy_subnet_gpu(GSubNet *cnet, int max_delay)
{
    GSubNet *tmp = new GSubNet;
    tmp->id = cnet->id;
    // tmp->max_delay=cnet->max_delay;
    tmp->neus_size = cnet->neus_size;
    tmp->syns_size = cnet->syns_size;
    tmp->out_net_size = cnet->out_net_size;
    // 拷贝神经元数据
    int num = tmp->neus_size;
    tmp->neus.ids = toGPU(cnet->neus.ids, num);
    tmp->neus.V_m = toGPU(cnet->neus.V_m, num); // lif
    tmp->neus.I_exc = toGPU(cnet->neus.I_exc, num);
    tmp->neus.I_inh = toGPU(cnet->neus.I_inh, num);
    tmp->neus.I_buffer_exc = toGPU(cnet->neus.I_buffer_exc, num * max_delay);
    tmp->neus.I_buffer_inh = toGPU(cnet->neus.I_buffer_inh, num * max_delay);
    tmp->neus.Fired = toGPU(cnet->neus.Fired, num);
    tmp->neus.Fire_cnt = toGPU(cnet->neus.Fire_cnt, num);
    tmp->neus.Last_fired = toGPU(cnet->neus.Last_fired, num);
    tmp->neus.Refrac_state = toGPU(cnet->neus.Refrac_state, num); // lif
    tmp->neus.rate = toGPU(cnet->neus.rate, num);                 // poisson
    tmp->neus.poisson = toGPU(cnet->neus.poisson, num);
    tmp->neus.type = toGPU(cnet->neus.type, num);
    tmp->neus.state = toGPU(cnet->neus.state, num); // poisson
    // 拷贝内部突触
    num = cnet->syns_size;
    tmp->syns.src = toGPU(cnet->syns.src, num);
    tmp->syns.tar = toGPU(cnet->syns.tar, num);
    tmp->syns.weight = toGPU(cnet->syns.weight, num);
    tmp->syns.delay = toGPU(cnet->syns.delay, num);
    // 拷贝外部突触
    tmp->out_net_size = cnet->out_net_size;
    // OUTSYNBlock *tmpouts = new OUTSYNBlock[tmp->outs_size];
    // for (int i = 0; i < tmp->outs_size; i++)
    // {
    //     tmpouts[i].syn_size = cnet->outs[i].syn_size;
    //     tmpouts[i].tar_id = cnet->outs[i].tar_id;
    //     tmpouts[i].block.src = toGPU(cnet->outs[i].block.src, cnet->outs[i].syn_size);
    //     tmpouts[i].block.tar = toGPU(cnet->outs[i].block.tar, cnet->outs[i].syn_size);
    //     tmpouts[i].block.weight = toGPU(cnet->outs[i].block.weight, cnet->outs[i].syn_size);
    //     tmpouts[i].block.delay = toGPU(cnet->outs[i].block.delay, cnet->outs[i].syn_size);
    // }
    // tmp->outs = toGPU(tmpouts, tmp->outs_size);
    GSubNet *gnet = toGPU(tmp, 1);
    // delete[] tmpouts;
    delete tmp;
    return gnet;
}
void MGBrain::init_buffer(SpikeBuffer *cbuffer, int size)
{

    cbuffer->spikes = new real[size];
    cbuffer->targets = new int[size]();
    cbuffer->times = new int[size]();
}

MGBrain::SpikeBuffer *MGBrain::copy_buffer_gpu(SpikeBuffer *cbuffer, int size)
{
    SpikeBuffer *tmp = new SpikeBuffer;
    tmp->spikes = toGPU(cbuffer->spikes, size);
    tmp->targets = toGPU(cbuffer->targets, size);
    tmp->times = toGPU(cbuffer->times, size);
    return tmp;
}
MGBrain::SpikeBuffer **MGBrain::copy_buffers_gpu(std::vector<SpikeBuffer *> &cbuffers)
{
    return toGPU(cbuffers.data(), cbuffers.size());
}
void MGBrain::free_buffer(SpikeBuffer *cbuffer)
{
    delete cbuffer->spikes;
    delete cbuffer->targets;
    delete cbuffer->times;
}
void MGBrain::free_buffer_gpu(SpikeBuffer *gbuffer)
{
    gpuFree(gbuffer->spikes);
    gpuFree(gbuffer->targets);
    gpuFree(gbuffer->times);
    delete gbuffer;
}
