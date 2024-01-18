#include "mgcore.cuh"
namespace MGBrain
{
    /**LIF神经元使用常量
     * 0:DT
     * 1:V_INIT
     * 2:V_REST
     * 3:V_RESET
     * 4:V_THRESH
     * 5:C_M
     * 6:TAU_M
     * 7:I_OFFSET
     * 8:TAU_REFRAC
     * 9:
     * 10:TAU_SYN_EXC
     * 11:TAU_SYN_INH
     * 12:P_EXC
     * 13:P_INH
     * 14:V_TMP
     * 15:C_EXC
     * 16:C_INH
     * 17:CM
     * 18:
     * 19:
     * 20:
     * 21:P11EXC
     * 22:P11INH
     * 23:P21EXC
     * 24:P21INH
     * 25:P22
     */
    __constant__ real LIFECONSTS[30] = {
        0.0001f,       // 0 DT
        0.0f,          // 1 V_INIT
        0.0f,          // 2 V_REST
        0.0f,          // 3 V_RESET
        0.02f,         // 4 V_THRESH
        0.25f,         // 5 C_M
        10.0f,         // 6 TAU_M
        0.0f,          // 7 I_OFFSET
        0.0004f,       // 8 TAU_REFRAC
        0.0f,          // 9
        1.0f,          // 10 TAU_SYN_EXC
        1.0f,          // 11 TAU_SYN_INH
        0.0f,          // 12 P_EXC
        0.0f,          // 13 P_INH
        0.0f,          // 14 V_TMP
        0.0f,          // 15 C_EXC
        0.0f,          // 16 C_INH
        0.0f,          // 17 CM
        0.0f,          // 18
        0.0f,          // 19
        0.0f,          // 20
        0.951229453f,  // 21 P11EXC
        0.967216074f,  // 22 P11INH
        0.0799999982f, // 23 P21EXC
        0.0599999987f, // 24 P21INH
        0.90483743f,   // 25 P22
        0.0f,          // 26
        0.0f,          // 27
        0.0f,          // 28
        0.0f,          // 29
    };

    /**STDP突触使用常量
     * 0: A_LTP
     * 1: A_LTD
     * 2: TAU_LTP
     * 3: TAU_LTD
     * 4: W_MAX
     * 5: W_MIN
     */
    __constant__ real STDPCONSTS[6] = {
        0.1,   // A_LTP
        -0.01, // A_LTD
        17,    // TAU_LTP
        34,    // TAU_LTD
        40,    // W_MAX
        0      // W_MIN
    };
    // 神经网络仿真使用常量
    __constant__ int MAX_DELAY = 10;
    // __device__ int out_spike = 0;
    // __device__ int in_spike = 0;
    // __device__ int fire_cnt = 0;
    __device__ inline size_t getSynId(sid ref)
    {
        return ref & (0x00ffffffffffffff);
    }
    __device__ inline int getSynZone(sid ref)
    {
        return (ref & (0xff00000000000000)) >> 56;
    }
    __device__ inline real getInput(real *buffer, int id, int step)
    {
        int loc = (id * MAX_DELAY) + (step % MAX_DELAY);
        return buffer[loc];
    }
    __device__ inline void clearInput(real *buffer, int nid, int step)
    {
        int loc = (nid * MAX_DELAY) + (step % MAX_DELAY);
        buffer[loc] = 0.0f;
    }
    // __device__ inline void setFired(bool *fired_list, int nid, int step, bool fired)
    // {
    //     fired_list[nid] = fired;
    // }
    // __device__ inline bool getFired(bool *fired_list, int nid, int step)
    // {
    //     return fired_list[nid];
    // }
    __device__ inline void pushInnerSpike(real *buffer, int nid, int time, real value)
    {
        atomicAdd(&buffer[(nid * MAX_DELAY) + ((time + MAX_DELAY) % MAX_DELAY)], value);
    }
    __global__ void mgsim_neus_core(GSubNet *net, int step)
    {
        int ref = blockIdx.x * blockDim.x + threadIdx.x;
        if (ref >= net->neus.size)
            return;
        if (net->neus.type[ref] == NeuronType::POISSON)
        { /// 仿真泊松神经元
            float rand = curand_uniform(&(net->neus.state[ref]));
            bool fired = net->neus.rate[ref] > rand;
            net->neus.fired[ref] = fired;
            if (fired)
            {
                net->neus.fire_cnt[ref]++;
                net->neus.last_fired[ref] = step;
            }
        }
        else if (net->neus.type[ref] == NeuronType::LIF)
        { /// 仿真LIF神经元
            net->neus.fired[ref] = false;
            if (net->neus.Refrac_state[ref] > 0)
            {
                --net->neus.Refrac_state[ref];
                clearInput(net->neus.I_buffer_exc, ref, step);
                clearInput(net->neus.I_buffer_inh, ref, step);
            }
            else
            {
                net->neus.V_m[ref] = LIFECONSTS[17] * net->neus.V_m[ref] + LIFECONSTS[14] + net->neus.I_exc[ref] * LIFECONSTS[12] + net->neus.I_inh[ref] * LIFECONSTS[13];
                net->neus.I_exc[ref] *= LIFECONSTS[15];
                net->neus.I_inh[ref] *= LIFECONSTS[16];
                bool fired = net->neus.V_m[ref] >= LIFECONSTS[4];
                if (fired)
                {
                    net->neus.fired[ref] = true;
                    net->neus.fire_cnt[ref]++;
                    net->neus.V_m[ref] = LIFECONSTS[3];
                    net->neus.last_fired[ref] = step;
                    net->neus.Refrac_state[ref] = ((int)std::round(LIFECONSTS[8] / LIFECONSTS[0])) - 1;
                }
                else
                {
                    net->neus.I_exc[ref] += getInput(net->neus.I_buffer_exc, ref, step);
                    net->neus.I_inh[ref] += getInput(net->neus.I_buffer_inh, ref, step);
                }
                clearInput(net->neus.I_buffer_exc, ref, step);
                clearInput(net->neus.I_buffer_inh, ref, step);
            }
        }
        else if (net->neus.type[ref] == NeuronType::LIFB)
        {
            // printf("lif0\n");
            net->neus.fired[ref] = false;

            if (net->neus.Refrac_state[ref] > 0)
            {
                --net->neus.Refrac_state[ref];
                clearInput(net->neus.I_buffer_exc, ref, step);
                clearInput(net->neus.I_buffer_inh, ref, step);
            }
            else
            {
                bool fired = net->neus.V_m[ref] >= LIFECONSTS[4];
                if (fired)
                {
                    net->neus.fired[ref] = true;
                    net->neus.fire_cnt[ref]++;
                    net->neus.V_m[ref] = LIFECONSTS[3];
                    net->neus.last_fired[ref] = step;
                    net->neus.Refrac_state[ref] = ((int)std::round(LIFECONSTS[8] / LIFECONSTS[0])) - 1;
                }
                else
                {
                    real I_e = getInput(net->neus.I_buffer_exc, ref, step);
                    real I_i = getInput(net->neus.I_buffer_inh, ref, step);
                    net->neus.V_m[ref] += I_e + I_i - 0.005f * net->neus.V_m[ref];
                    // printf("vm:%f,I_e:%f,I_i:%f\n",net->neus.V_m[ref],I_e,I_i);
                }
                clearInput(net->neus.I_buffer_exc, ref, step);
                clearInput(net->neus.I_buffer_inh, ref, step);
            }
        }
        else if (net->neus.type[ref] == NeuronType::LIFE)
        {
            net->neus.fired[ref] = false;
            net->neus.I_exc[ref] = getInput(net->neus.I_buffer_exc, ref, step) + net->neus.I_exc[ref] * (0.98f);
            net->neus.I_inh[ref] = -getInput(net->neus.I_buffer_inh, ref, step) + net->neus.I_inh[ref] * (0.99f);
            if (net->neus.Refrac_state[ref] > 0)
            {
                --net->neus.Refrac_state[ref];
                clearInput(net->neus.I_buffer_exc, ref, step);
                clearInput(net->neus.I_buffer_inh, ref, step);
            }
            else
            {
                bool fired = net->neus.V_m[ref] >= LIFECONSTS[4];
                if (fired)
                {
                    net->neus.fired[ref] = true;
                    net->neus.fire_cnt[ref]++;

                    net->neus.V_m[ref] = LIFECONSTS[3];
                    net->neus.last_fired[ref] = step;
                    net->neus.Refrac_state[ref] = ((int)std::round(LIFECONSTS[8] / LIFECONSTS[0])) - 1;
                }
                else
                {
                    // if(net->neus.ids[ref]==1)printf("vm:%f,vthres:%f,vreset:%f,refrac:%d\n",net->neus.V_m[ref],LIFECONSTS[4],LIFECONSTS[3],((int)std::round(LIFECONSTS[8] / LIFECONSTS[0])) - 1);
                    float V = net->neus.V_m[ref];
                    net->neus.V_m[ref] += ((-0.06f - V) +
                                           net->neus.I_exc[ref] * (0.0f - V) +
                                           net->neus.I_inh[ref] * (-0.08f - V) +
                                           0.02f) *
                                          0.005f;
                }
                clearInput(net->neus.I_buffer_exc, ref, step);
                clearInput(net->neus.I_buffer_inh, ref, step);
            }
        }
    }
    __global__ void mgsim_syns_core(GSubNet *net, int step, int *buffer_size_list, SpikeBuffer **buffers)
    {
        int nid = blockIdx.x * blockDim.x + threadIdx.x;
        if (nid >= net->neus.size)
            return;
        if (!net->neus.fired[nid])
            return;
        size_t start_loc = net->adjs.axon_offs[nid];
        size_t size = net->adjs.axon_offs[nid + 1] - start_loc;
        for (size_t idx = start_loc; idx < start_loc + size; idx++)
        {
            // size_t idx=i+start_loc;
            size_t ref = net->adjs.axon_refs[idx];
            size_t id = getSynId(ref);
            int zone = getSynZone(ref);
            int tar = net->syns.tar[id];
            real weight = net->syns.weight[id];
            int delay = net->syns.delay[id];
            // 发放脉冲
            if (zone == net->id) // 域内突触
            {
                if (weight > 0)
                    pushInnerSpike(net->neus.I_buffer_exc, tar, (step + delay), weight);
                else
                    pushInnerSpike(net->neus.I_buffer_inh, tar, (step + delay), weight);
            }
            else
            { // 域外突触
                int loc = atomicAdd(&(buffer_size_list[zone]), 1);
                // int loc = 0;
                buffers[zone]->spikes[loc] = weight;
                buffers[zone]->targets[loc] = tar;
                buffers[zone]->times[loc] = step + delay;
            }
        }
    }
    __global__ void mgsim_syn_fast_core(GSubNet *net, int step, int *buffer_size_list, SpikeBuffer **buffers)
    {
        int nid = blockIdx.x;
        if (nid >= net->neus.size)
            return;
        if (!net->neus.fired[nid])
            return;
        size_t start_loc = net->adjs.axon_offs[nid];
        size_t size = net->adjs.axon_offs[nid + 1] - start_loc;
        for (size_t i = threadIdx.x; i < size; i += blockDim.x)
        {
            size_t idx = i + start_loc;
            size_t ref = net->adjs.axon_refs[idx];
            size_t id = getSynId(ref);
            int zone = getSynZone(ref);
            int tar = net->syns.tar[id];
            real weight = net->syns.weight[id];
            int delay = net->syns.delay[id];
            // 发放脉冲
            if (zone == net->id) // 域内突触
            {
                if (weight > 0)
                    pushInnerSpike(net->neus.I_buffer_exc, tar, (step + delay), weight);
                else
                    pushInnerSpike(net->neus.I_buffer_inh, tar, (step + delay), weight);
            }
            else
            { // 域外突触
                int loc = atomicAdd(&(buffer_size_list[zone]), 1);
                buffers[zone]->spikes[loc] = weight;
                buffers[zone]->targets[loc] = tar;
                buffers[zone]->times[loc] = step + delay;
            }
        }
    }
    __global__ void mgsim_syn_fast_dense_core(GSubNet *net, int step, SpikeDenseBuffer **buffers)
    {
        int nid = blockIdx.x;
        if (nid >= net->neus.size)
            return;
        if (!net->neus.fired[nid])
            return;
        size_t start_loc = net->adjs.axon_offs[nid];
        size_t size = net->adjs.axon_offs[nid + 1] - start_loc;
        for (size_t i = threadIdx.x; i < size; i += blockDim.x)
        {
            size_t idx = i + start_loc;
            size_t ref = net->adjs.axon_refs[idx];
            size_t id = getSynId(ref);
            int zone = getSynZone(ref);
            int tar = net->syns.tar[id];
            real weight = net->syns.weight[id];
            int delay = net->syns.delay[id];
            // 发放脉冲
            if (zone == net->id) // 域内突触
            {
                if (weight > 0)
                    pushInnerSpike(net->neus.I_buffer_exc, tar, (step + delay), weight);
                else
                    pushInnerSpike(net->neus.I_buffer_inh, tar, (step + delay), weight);
            }
            else
            { // 域外突触
                // if(buffers==nullptr)return;
                // if(buffers[zone]==nullptr)return;
                int loc = buffers[zone]->mapper[tar - buffers[zone]->start];
                if (loc < 0 || loc >= buffers[zone]->neusize)
                    return;
                if (weight > 0)
                    pushInnerSpike(buffers[zone]->buffer_exc, loc, (step + delay), weight);
                else
                    pushInnerSpike(buffers[zone]->buffer_inh, loc, (step + delay), weight);
            }
        }
    }
    __device__ real stdp_cal_weight(int dt, real weight)
    {
        real dw = 0;
        if (dt < 0)
        {
            dw = STDPCONSTS[0] * exp(dt / STDPCONSTS[2]);
        }
        else if (dt > 0)
        {
            dw = STDPCONSTS[1] * exp(-dt / STDPCONSTS[3]);
        }
        else
        {
            dw = 0;
        }
        real nweight = weight + dw;
        nweight = (nweight > STDPCONSTS[4]) ? STDPCONSTS[4] : ((nweight < STDPCONSTS[5]) ? STDPCONSTS[5] : nweight);
        return nweight;
    }
    __global__ void mgsim_stdp_core(GSubNet *net, int step, int **last_fired_addrs, int **syn_src_addrs, real **syn_weight_addrs)
    {
        int nid = blockIdx.x * blockDim.x + threadIdx.x;
        if (nid >= net->neus.size)
            return;
        if (!net->neus.fired[nid])
            return;
        size_t axon_offset = net->adjs.axon_offs[nid];
        size_t axon_size = net->adjs.axon_offs[nid + 1] - axon_offset;
        for (int i = 0; i < axon_size; i++)
        {
            size_t ref = net->adjs.axon_refs[axon_offset + i];
            size_t id = getSynId(ref);
            int zone = getSynZone(ref);
            if (zone == net->id)
            {
                int tar = net->syns.tar[id];
                int dt = net->neus.last_fired[nid] - net->neus.last_fired[tar];
                atomicExch(&(net->syns.weight[id]), stdp_cal_weight(dt, net->syns.weight[id]));
            }
            else
            {
                int tar = net->syns.tar[id];
                int dt = net->neus.last_fired[nid] - last_fired_addrs[zone][tar];
                atomicExch(&(net->syns.weight[id]), stdp_cal_weight(dt, net->syns.weight[id]));
            }
        }
        size_t dend_offset = net->adjs.dend_offs[nid];
        size_t dend_size = net->adjs.dend_offs[nid + 1] - dend_offset;
        for (int i = 0; i < dend_size; i++)
        {
            size_t ref = net->adjs.dend_refs[dend_offset + i];
            size_t id = getSynId(ref);
            int zone = getSynZone(ref);
            if (zone == net->id)
            {
                int src = net->syns.src[id];
                int dt = net->neus.last_fired[src] - net->neus.last_fired[nid];
                atomicExch(&(net->syns.weight[id]), stdp_cal_weight(dt, net->syns.weight[id]));
            }
            else
            {
                int src = syn_src_addrs[zone][id];
                int dt = last_fired_addrs[zone][src] - net->neus.last_fired[nid];
                atomicExch(&(syn_weight_addrs[zone][id]), stdp_cal_weight(dt, syn_weight_addrs[zone][id]));
            }
        }
    }
    __global__ void mgsim_recv_core(GSubNet *net, int index, size_t buffer_size, SpikeBuffer **buffers)
    {
        int ref = blockIdx.x * blockDim.x + threadIdx.x;
        if (ref >= buffer_size)
            return;
        // atomicAdd(&in_spike, 1);
        int tar = buffers[index]->targets[ref];
        int time = buffers[index]->times[ref];
        real spike = buffers[index]->spikes[ref];
        if (spike > 0)
        {
            pushInnerSpike(net->neus.I_buffer_exc, tar, time, spike);
        }
        else
        {
            pushInnerSpike(net->neus.I_buffer_inh, tar, time, spike);
        }
    }

    __global__ void mgsim_recv_dense_core(GSubNet *net, int step, int zone, int neus_size, SpikeDenseBuffer **buffers)
    {
        int ref = blockIdx.x * blockDim.x + threadIdx.x;
        if (ref >= neus_size||buffers[zone]==nullptr)
            return;
        int tar = buffers[zone]->targets[ref];
        for (int i = 0; i < MAX_DELAY; i++)
        {
            int offset = (step + i + MAX_DELAY) % MAX_DELAY;
            real i_exc = buffers[zone]->buffer_exc[ref * MAX_DELAY + offset];
            atomicAdd(&(net->neus.I_buffer_exc[tar * MAX_DELAY + offset]), i_exc);

            real i_inh = buffers[zone]->buffer_inh[ref * MAX_DELAY + offset];
            atomicAdd(&(net->neus.I_buffer_inh[tar * MAX_DELAY + offset]), i_inh);
        }
    }

    __global__ void mgsim_init_core(int size, curandState *state, unsigned long seed)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= size)
            return;
        curand_init(seed, tid, 0, &state[tid]);
    }
    __global__ void sum_spikes(int *buffer_size_list, int size)
    {
        int sum = 0;
        for (int i = 0; i < size; i++)
        {
            sum += buffer_size_list[i];
        }
        printf("%d,\n", sum);
    }
    __global__ void print_test()
    {
        // printf("in_fire:%d\n", in_spike);
        // out_spike = 0;
    }
    inline void transfer_sparse(SpikeBuffer &out_buffer, int out_device, SpikeBuffer &in_buffer, int in_device, int size, cudaStream_t stream)
    {
        // std::cout<<"transfer size:"<<size<<std::endl;
        if (size == 0)
            return;
        if (Config::SINGLE_GPU)
        {
            CUDACHECK(cudaMemcpy(in_buffer.spikes, out_buffer.spikes, sizeof(real) * size, cudaMemcpyDeviceToDevice));
            CUDACHECK(cudaMemcpy(in_buffer.targets, out_buffer.targets, sizeof(int) * size, cudaMemcpyDeviceToDevice));
            CUDACHECK(cudaMemcpy(in_buffer.times, out_buffer.times, sizeof(int) * size, cudaMemcpyDeviceToDevice));
        }
        else
        {
            // printf("out:%d,in:%d,size:%d,outp:%p,inp:%p\n",out_device,in_device,size,out_buffer.spikes,in_buffer.spikes);
            CUDACHECK(cudaMemcpyPeerAsync(in_buffer.spikes, in_device, out_buffer.spikes, out_device, sizeof(real) * size, stream));
            CUDACHECK(cudaMemcpyPeerAsync(in_buffer.targets, in_device, out_buffer.targets, out_device, sizeof(int) * size, stream));
            CUDACHECK(cudaMemcpyPeerAsync(in_buffer.times, in_device, out_buffer.times, out_device, sizeof(int) * size, stream));
        }
    }

    void trans_spikes_sparse(GSubNet *cnet, BufferManager &manager, std::vector<int> &mapper, int turn, cudaStream_t &trans_stream)
    {
        int id = cnet->id;
        /// 脉冲同步分为两步，脉冲数据传输和脉冲收集
        /// 脉冲数据传输，多流并行
        for (int i = 0; i < cnet->npart; i++)
        {
            int index = i;
            if (Config::STAGGERED)
            {
                if(i==0)continue;
                index = cnet->out_net_id_list[i];
            }
            int buffersize = manager.getOutBufferSize(id, index, turn);
            if (buffersize == 0 || !manager.out_valid[id][index])
                continue;
            // printf("buffersize:%d\n",buffersize);
            int src_device = mapper[id];
            int tar_device = mapper[manager.getTarNetId(id, index)];
            
            SpikeBuffer &curoutbuffer = manager.getCurOutCBuffer(id, index, turn);
            SpikeBuffer &tarinbuffer = manager.getTarInCBuffer(id, index, turn);
            transfer_sparse(curoutbuffer, src_device, tarinbuffer, tar_device, buffersize, trans_stream);
        }
    }
    void recvs_spikes_sparse(GSubNet *gnet, GSubNet *cnet, int blocksize, BufferManager &manager, int turn, std::vector<cudaStream_t> &recv_streams)
    {
        int id = cnet->id;
        /// 脉冲数据收集,多流并行
        for (int i = 0; i < cnet->npart; i++)
        {
            size_t buffersize = manager.getInBufferSize(id, i, turn);
            if (buffersize == 0)
                continue;
            auto buffer = manager.getInGGBuffer(cnet->id, turn);
            mgsim_recv_core<<<buffersize / blocksize + 1, blocksize, 0, recv_streams[i]>>>(gnet, i, buffersize, buffer);
        }
        /// 同步脉冲收集流
        for (int i = 0; i < cnet->npart; i++)
        {
            if (manager.getInBufferSize(id, i, turn) == 0)
                continue;
            CUDACHECK(cudaStreamSynchronize(recv_streams[i]));
        }
    }
    void mgsim_step_sparse(GSubNet *gnet, GSubNet *cnet, int step, int blocksize, BufferManager &manager, GNetAddrs &gaddrs, int turn,cudaStream_t& sim_stream)
    {
        /// 默认流用于仿真,其他流用于脉冲同步
        int neu_size = cnet->neus.size;
        int gridsize = neu_size / blocksize + 1;
        /// 仿真神经元
        mgsim_neus_core<<<gridsize, blocksize,0,sim_stream>>>(gnet, step);
        /// 仿真突触
        SpikeBuffer **gbuffers = manager.getOutGGBuffer(cnet->id, turn);
        int *gout_buffer_size_list = manager.getGBufferSizeList(cnet->id, turn);
        mgsim_syn_fast_core<<<neu_size, 64,0,sim_stream>>>(gnet, step, gout_buffer_size_list, gbuffers);
        /// 仿真STDP
        if (Config::STDP)
        {
            int **last_fired_addrs = gaddrs.glast_fired_addrs;
            int **syn_src_addrs = gaddrs.gsyn_src_addrs;
            real **syn_weight_addrs = gaddrs.gsyn_weight_addrs;
            mgsim_stdp_core<<<gridsize, blocksize,0,sim_stream>>>(gnet, step, last_fired_addrs, syn_src_addrs, syn_weight_addrs);
        }
    }

    inline void transfer_dense(SpikeDenseBuffer *out_buffer, int out_device, SpikeDenseBuffer *in_buffer, int in_device, int size, int max_delay, cudaStream_t stream)
    {
        if (Config::SINGLE_GPU)
        {
            CUDACHECK(cudaMemcpy(in_buffer->buffer_exc, out_buffer->buffer_exc, sizeof(real) * size * max_delay, cudaMemcpyDeviceToDevice));
            CUDACHECK(cudaMemcpy(in_buffer->buffer_inh, out_buffer->buffer_inh, sizeof(real) * size * max_delay, cudaMemcpyDeviceToDevice));
        }
        else
        {
            // if(in_buffer==nullptr||out_buffer==nullptr||size==0)return;
            CUDACHECK(cudaMemcpyPeerAsync(in_buffer->buffer_exc, in_device, out_buffer->buffer_exc, out_device, sizeof(real) * size * max_delay, stream));
            CUDACHECK(cudaMemcpyPeerAsync(in_buffer->buffer_inh, in_device, out_buffer->buffer_inh, out_device, sizeof(real) * size * max_delay, stream));
        }
    }
    inline void clear_dense(SpikeDenseBuffer* out_buffer, int out_device, SpikeDenseBuffer* in_buffer, int in_device, int size, int max_delay, cudaStream_t stream)
    {
        if (Config::SINGLE_GPU)
        {
            CUDACHECK(cudaMemset(out_buffer->buffer_exc, 0.0f, sizeof(real) * size * max_delay));
            CUDACHECK(cudaMemset(out_buffer->buffer_inh, 0.0f, sizeof(real) * size * max_delay));
        }
        else
        {
            // if(in_buffer==nullptr||out_buffer==nullptr||size==0)return;
            CUDACHECK(cudaMemsetAsync(out_buffer->buffer_exc, 0.0f, sizeof(real) * size * max_delay, stream));
            CUDACHECK(cudaMemsetAsync(out_buffer->buffer_inh, 0.0f, sizeof(real) * size * max_delay, stream));
        }
    }
    void trans_spikes_dense(GSubNet *cnet, DenseBufferManager &manager, int turn, cudaStream_t &trans_streams, std::vector<int> &mapper, int max_delay)
    {
        int id = cnet->id;
        /// 脉冲同步分为两步，脉冲数据传输和脉冲收集
        /// 脉冲数据传输，多流并行
        for (int i = 0; i < cnet->npart; i++)
        {
            int index = i;
            if (Config::STAGGERED)
            {
                if(i==0)continue;
                index = cnet->out_net_id_list[i];
            }
            int outneusize = manager.getOutNeuSize(id, index);
            if (outneusize == 0 || !manager.out_valid[id][index])
                continue;
            // printf("buffersize:%d\n",buffersize);
            int src_device = mapper[id];
            int tar_device = mapper[manager.getTarNetId(id, index)];
            SpikeDenseBuffer* curoutbuffer = manager.getCurOutCBuffer(id, index, turn);
            SpikeDenseBuffer* tarinbuffer = manager.getTarInCBuffer(id, index, turn);
            // std::cout<<"trans:src_net:"<<id<<"["<<index<<"]"<<"->"<<"tar_net:"<<manager.getTarNetId(id,index)<<"["<<manager.getTarNetIdx(id,index)<<"]"<<"="<<outneusize<<std::endl;
            transfer_dense(curoutbuffer, src_device, tarinbuffer, tar_device, outneusize, max_delay, trans_streams);
        }
        for (int i = 0; i < cnet->npart; i++)
        {
            int index = i;
            if (Config::STAGGERED)
            {
                if(i==0)continue;
                index = cnet->out_net_id_list[i];
            }
            int outneusize = manager.getOutNeuSize(id, index);
            if (outneusize == 0 || !manager.out_valid[id][index])
                continue;
            // printf("buffersize:%d\n",buffersize);
            int src_device = mapper[id];
            int tar_device = mapper[manager.getTarNetId(id, index)];
            SpikeDenseBuffer* curoutbuffer = manager.getCurOutCBuffer(id, index, turn);
            SpikeDenseBuffer* tarinbuffer = manager.getTarInCBuffer(id, index, turn);
            clear_dense(curoutbuffer, src_device, tarinbuffer, tar_device, outneusize, max_delay, trans_streams);
        }
    }
    void recvs_spikes_dense(GSubNet *gnet, GSubNet *cnet, int step, int blocksize, DenseBufferManager &manager, int turn, std::vector<cudaStream_t> &recv_streams)
    {
        int id = cnet->id;
        /// 脉冲数据收集,多流并行
        for (int i = 0; i < cnet->npart; i++)
        {
            int inneusize = manager.getInNeuSize(id, i);
            if (inneusize == 0)
                continue;
            auto buffer = manager.getInGGBuffer(cnet->id, turn);
            mgsim_recv_dense_core<<<inneusize / blocksize + 1, blocksize, 0, recv_streams[i]>>>(gnet, step, i, inneusize, buffer);
        }
        // 同步脉冲收集流
        for (int i = 0; i < recv_streams.size(); i++)
        {
            if (manager.getInNeuSize(id, i) == 0)
                continue;
            CUDACHECK(cudaStreamSynchronize(recv_streams[i]));
        }
    }
    void mgsim_step_dense(GSubNet *gnet, GSubNet *cnet, int step, int blocksize, DenseBufferManager &manager, GNetAddrs &gaddrs, int turn)
    {
        int neu_size = cnet->neus.size;
        int gridsize = neu_size / blocksize + 1;
        cudaStream_t &sim_stream = manager.sim_streams[cnet->id];
        mgsim_neus_core<<<gridsize, blocksize, 0, sim_stream>>>(gnet, step);
        /// 仿真突触
        SpikeDenseBuffer **gbuffers = manager.getOutGGBuffer(cnet->id, turn);
        mgsim_syn_fast_dense_core<<<neu_size, 64, 0, sim_stream>>>(gnet, step, gbuffers);
        /// 仿真STDP
        if (Config::STDP)
        {
            int **last_fired_addrs = gaddrs.glast_fired_addrs;
            int **syn_src_addrs = gaddrs.gsyn_src_addrs;
            real **syn_weight_addrs = gaddrs.gsyn_weight_addrs;
            mgsim_stdp_core<<<gridsize, blocksize, 0, sim_stream>>>(gnet, step, last_fired_addrs, syn_src_addrs, syn_weight_addrs);
        }
    }
    void init_state(int num,curandState* gstates){
        srand(time(0));
        int blocksize = 1024;
        mgsim_init_core<<<num / blocksize + 1, blocksize>>>(num, gstates, rand());
    }

    void test(int i)
    {
        CUDACHECK(cudaSetDevice(i));
        print_test<<<1, 1>>>();
    }
    void test2(int *buffer, int size)
    {
        sum_spikes<<<1, 1>>>(buffer, size);
    }
    /// @brief 拷贝仿真常量到GPU中
    /// @param max_delay
    /// @param dt
    /// @param lifconst
    /// @param stdpconst
    void copy_consts_gpu(int max_delay, real dt)
    {
        // CUDACHECK(cudaSetDevice(0));
        // std::cout<<"maxdelay:"<<max_delay<<std::endl;
        CUDACHECK(cudaMemcpyToSymbol(MAX_DELAY, &max_delay, sizeof(int)));
        real lifconsts[30] = {
            dt,            // 0 DT
            0.0f,          // 1 V_INIT
            0.0f,          // 2 V_REST
            0.0f,          // 3 V_RESET
            15.0f,         // 4 V_THRESH
            0.25f,         // 5 C_M
            10.0f,         // 6 TAU_M
            0.0f,          // 7 I_OFFSET
            0.0004f,       // 8 TAU_REFRAC
            0.0f,          // 9
            1.0f,          // 10 TAU_SYN_EXC
            1.0f,          // 11 TAU_SYN_INH
            0.0f,          // 12 P_EXC
            0.0f,          // 13 P_INH
            0.0f,          // 14 V_TMP
            0.0f,          // 15 C_EXC
            0.0f,          // 16 C_INH
            0.0f,          // 17 CM
            0.0f,          // 18
            0.0f,          // 19
            0.0f,          // 20
            0.951229453f,  // 21 P11EXC
            0.967216074f,  // 22 P11INH
            0.0799999982f, // 23 P21EXC
            0.0599999987f, // 24 P21INH
            0.90483743f,   // 25 P22
            0.0f,          // 26
            0.0f,          // 27
            0.0f,          // 28
            0.0f,          // 29
        };
        CUDACHECK(cudaMemcpyToSymbol(LIFECONSTS, lifconsts, sizeof(real) * 30));

        real stdpconsts[6] = {
            0.1,   // A_LTP
            -0.01, // A_LTD
            17,    // TAU_LTP
            34,    // TAU_LTD
            40,    // W_MAX
            0      // W_MIN
        };
        CUDACHECK(cudaMemcpyToSymbol(STDPCONSTS, stdpconsts, sizeof(real) * 6));
        CUDACHECK(cudaDeviceSynchronize());
    }
    void copy_consts_gpu(int max_delay, real dt, bool nlifconst, std::array<real, 30> lifconst, bool nstdpconst, std::array<real, 6> stdpconst)
    {
        // std::cout<<"maxdelay:"<<max_delay<<std::endl;
        CUDACHECK(cudaMemcpyToSymbol(MAX_DELAY, &max_delay, sizeof(int)));
        if (nlifconst)
        {
            CUDACHECK(cudaMemcpyToSymbol(LIFECONSTS, lifconst.data(), sizeof(real) * 30));
        }
        if (nstdpconst)
        {
            CUDACHECK(cudaMemcpyToSymbol(STDPCONSTS, stdpconst.data(), sizeof(real) * 6));
        }
        CUDACHECK(cudaDeviceSynchronize());
    }

};
