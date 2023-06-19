#include "uvamgnet.cuh"
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
    __device__ inline real getInput(real *buffer, int id, int step)
    {
        int loc=id * MAX_DELAY + step % MAX_DELAY;
        real res=buffer[loc];
        buffer[loc]=0.0;
        return res;
    }
    __device__ inline void pushSpike(real *buffer, int nid, int step, real value)
    {
        atomicAdd(&buffer[nid * MAX_DELAY + step], value);
    }
    // 仿真神经元
    __global__ void mgsimNeusUVA(GSubNetUVA *net, int step)
    {
        int ref = blockIdx.x * blockDim.x + threadIdx.x;
        if (net->neus.poisson[ref])
        {
            /// 仿真泊松神经元
            float rand = curand_uniform(&(net->neus.state[ref]));
            net->neus.Fired[ref] = net->neus.rate[ref] > rand;
            if (net->neus.Fired[ref])
            {
                net->neus.Fire_cnt[ref]++;
                net->neus.Last_fired[ref] = step;
            }
            // if(id==0){
            //     printf("%d poisson' rate:%f\n",id,net->neus.rate[id]);
            // }
        }
        else
        {
            /// 仿真LIF神经元
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
            // if(net->neus.id[ref]==12){
            //     printf("step:%d,neu%d' vm0:%f\n",step,net->neus.id[ref],net->neus.V_m[ref]);
            // }
        }
    }
    // 仿真子网络内部突触
    __global__ void mgsimSynsUVA(GSubNetUVA *net, int step)
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
                pushSpike(net->neus.I_buffer_exc, tar, (step + delay) % MAX_DELAY, weight);
            else
                pushSpike(net->neus.I_buffer_inh, tar, (step + delay) % MAX_DELAY, weight);
        }

        // STDP权重更新
        if (!net->neus.Fired[src] && !net->neus.Fired[tar])
            return;
        int dt = net->neus.Last_fired[src] - net->neus.Last_fired[tar];
        real dw = 0;
        if (dt < 0)
        {
            dw = A_LTP * exp(dt / TAU_LTP);
        }
        else if (dt > 0)
        {
            dw = A_LTD * exp(-dt / TAU_LTD);
        }
        else
        {
            dw = 0;
        }
        real nweight = net->syns.weight[ref] + dw;
        nweight = (nweight > W_max) ? W_max : ((nweight < W_min) ? W_min : nweight);
        atomicExch(&(net->syns.weight[ref]), nweight);
    }
    // 仿真外部突触，也即脉冲同步操作
    __global__ void mgsimOutsUVA(GSubNetUVA *net, int index, int step, real *tar_exc_buffer_addr, real *tar_inh_buffer_addr, bool *tar_fired_addr, int *tar_last_fired_addr)
    {
        int ref = blockIdx.x * blockDim.x + threadIdx.x;
        int src = net->outs[index].block.src[ref];
        int tar = net->outs[index].block.tar[ref];
        real weight = net->outs[index].block.weight[ref];
        int delay = net->outs[index].block.delay[ref];

        // 向外发放脉冲
        if (net->neus.Fired[src])
        {
            if (weight > 0)
            {
                pushSpike(tar_exc_buffer_addr, tar, (step + delay) % MAX_DELAY, weight);
            }
            else
            {
                pushSpike(tar_inh_buffer_addr, tar, (step + delay) % MAX_DELAY, weight);
            }
        }
        // STDP权重更新
        if (!net->neus.Fired[src] && !tar_fired_addr[tar])
            return;
        int dt = net->neus.Last_fired[src] - tar_last_fired_addr[tar];
        real dw = 0;
        if (dt < 0)
            dw = A_LTP * exp(dt / TAU_LTP);
        else if (dt > 0)
            dw = A_LTD * exp(-dt / TAU_LTD);
        else
            dw = 0;
        real nweight = net->syns.weight[ref] + dw;
        nweight = (nweight > W_max) ? W_max : ((nweight < W_min) ? W_min : nweight);
        atomicExch(&(net->syns.weight[ref]), nweight);
        // if (ref == 0)
            // printf("outs\n");
    }
    
    /// @brief UVA方式仿真单个时间片
    void mgsimStepUVA(GSubNetUVA *gnet, GSubNetUVA *cnet, int step, int blocksize, std::vector<GSubNetAddrsUVA *> addrs,std::vector<cudaStream_t> streams)
    {
        int neu_size = cnet->neus_size;
        int syn_size = cnet->syns_size;
        #ifdef STREAM_EVENT_RECORD
            std::vector<std::pair<cudaEvent_t,cudaEvent_t>> recorder(cnet->outs_size);
            cudaEvent_t syn_s,syn_e,all_s,all_e,neu_s,neu_e;
            cudaEventCreate(&syn_s);
            cudaEventCreate(&syn_e);
            cudaEventCreate(&all_s);
            cudaEventCreate(&all_e);
            cudaEventCreate(&neu_s);
            cudaEventCreate(&neu_e);
            for(int i=0;i<recorder.size();i++){
                cudaEventCreate(&(recorder[i].first));
                cudaEventCreate(&(recorder[i].second));
            }
            cudaEventRecord(all_s);
            cudaEventRecord(neu_s);
        #endif
        /// 仿真神经元
        mgsimNeusUVA<<<neu_size / blocksize + 1, neu_size % blocksize>>>(gnet, step);
        CUDACHECK(cudaDeviceSynchronize());
        
        #ifdef STREAM_EVENT_RECORD
            cudaEventRecord(neu_e);
        #endif
        // 仿真外部突触
        for (int i = 0; i < cnet->outs_size; i++)
        {
            int out_size = cnet->outs[i].syn_size;
            #ifdef STREAM_EVENT_RECORD
            cudaEventRecord(recorder[i].first,streams[i]);
            #endif
            // 仿真外部突触核函数
            mgsimOutsUVA<<<out_size / blocksize + 1, out_size % blocksize,0,streams[i]>>>(gnet, i, step, addrs[i]->tar_exc_buffer, addrs[i]->tar_inh_buffer, addrs[i]->tar_fired_buffer, addrs[i]->tar_last_fired_buffer);
            
            #ifdef STREAM_EVENT_RECORD
            cudaEventRecord(recorder[i].second,streams[i]);
            #endif
        }
        
        #ifdef STREAM_EVENT_RECORD
            cudaEventRecord(syn_s);
        #endif
        // 仿真内部突触
        mgsimSynsUVA<<<syn_size / blocksize + 1, syn_size % blocksize>>>(gnet, step);
        #ifdef STREAM_EVENT_RECORD 
            cudaEventRecord(syn_e);
        #endif
        CUDACHECK(cudaDeviceSynchronize());
        #ifdef STREAM_EVENT_RECORD
            std::vector<float> times(recorder.size(),0.0f);
            std::cout<<"====device("<<cnet->id<<"),step("<<step<<")===="<<std::endl;
            for(int i=0;i<recorder.size();i++){
                cudaEventElapsedTime(&(times[i]),recorder[i].first,recorder[i].second);
                std::cout<<"outs("<<i<<")elapsed:"<<times[i]<<std::endl;
            }
            float syn_time=0;
            cudaEventElapsedTime(&(syn_time),syn_s,syn_e);
            std::cout<<"syns elapsed:"<<syn_time<<std::endl;
            float neu_time=0;
            cudaEventElapsedTime(&(neu_time),neu_s,neu_e);
            std::cout<<"neus elapsed:"<<neu_time<<std::endl;
            float all_time=0;
            cudaEventRecord(all_e);
            cudaEventElapsedTime(&(all_time),all_s,all_e);
            std::cout<<"alls elapsed:"<<all_time<<std::endl;
        #endif
        
    }
    void copy_constsuva_gpu(int max_delay,LIFConsts* lifconst,STDPConsts* stdpconst){
        CUDACHECK(cudaMemcpyToSymbol(MAX_DELAY,&max_delay,sizeof(int)));
        if(lifconst!=nullptr){
            CUDACHECK(cudaMemcpyToSymbol(_P22,&(lifconst->P22),sizeof(real)));
            CUDACHECK(cudaMemcpyToSymbol(_P11exc,&(lifconst->P11exc),sizeof(real)));
            CUDACHECK(cudaMemcpyToSymbol(_P11inh,&(lifconst->P11inh),sizeof(real)));
            CUDACHECK(cudaMemcpyToSymbol(_P21exc,&(lifconst->P21exc),sizeof(real)));
            CUDACHECK(cudaMemcpyToSymbol(_P21inh,&(lifconst->P21inh),sizeof(real)));
            CUDACHECK(cudaMemcpyToSymbol(V_rest,&(lifconst->V_rest),sizeof(real)));
            CUDACHECK(cudaMemcpyToSymbol(V_reset,&(lifconst->V_reset),sizeof(real)));
            CUDACHECK(cudaMemcpyToSymbol(C_m,&(lifconst->C_m),sizeof(real)));
            CUDACHECK(cudaMemcpyToSymbol(Tau_m,&(lifconst->Tau_m),sizeof(real)));
            CUDACHECK(cudaMemcpyToSymbol(V_thresh,&(lifconst->V_thresh),sizeof(real)));
            CUDACHECK(cudaMemcpyToSymbol(I_offset,&(lifconst->I_offset),sizeof(real)));
            CUDACHECK(cudaMemcpyToSymbol(Refrac_step,&(lifconst->Refrac_step),sizeof(real)));
        }
        if(stdpconst!=nullptr){
            CUDACHECK(cudaMemcpyToSymbol(A_LTP,&(stdpconst->A_LTP),sizeof(real)));
            CUDACHECK(cudaMemcpyToSymbol(A_LTD,&(stdpconst->A_LTD),sizeof(real)));
            CUDACHECK(cudaMemcpyToSymbol(TAU_LTP,&(stdpconst->TAU_LTP),sizeof(real)));
            CUDACHECK(cudaMemcpyToSymbol(TAU_LTD,&(stdpconst->TAU_LTD),sizeof(real)));
            CUDACHECK(cudaMemcpyToSymbol(W_max,&(stdpconst->W_max),sizeof(real)));
            CUDACHECK(cudaMemcpyToSymbol(W_min,&(stdpconst->W_min),sizeof(real)));
        }
        CUDACHECK(cudaDeviceSynchronize());
    }

};
void MGBrain::initGSubNetUVANeus(GSubNetUVA *cnet, int max_delay)
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
    cnet->neus.rate = new real[num]();
    cnet->neus.state = new curandState[num]();
}
void MGBrain::initGSubNetUVASyns(GSubNetUVA *cnet)
{
    int num = cnet->syns_size;
    cnet->syns.src = new int[num]();
    cnet->syns.tar = new int[num]();
    cnet->syns.delay = new int[num]();
    cnet->syns.weight = new real[num]();
}
void MGBrain::initGSubNetUVAOutSyns(SYNBlock *block, int num)
{
    block->src = new int[num];
    block->tar = new int[num];
    block->delay = new int[num];
    block->weight = new real[num];
}
void MGBrain::freeGSubNetUVA(GSubNetUVA *cnet)
{
    ///delete 神经元内存空间
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
    delete[] cnet->neus.rate;
    delete[] cnet->neus.state;
    ///delete 内突触内存空间
    delete[] cnet->syns.src;
    delete[] cnet->syns.tar;
    delete[] cnet->syns.delay;
    delete[] cnet->syns.weight;
    ///delete 外突触内存空间
    for (int i = 0; i < cnet->outs_size; i++)
    {
        delete[] cnet->outs[i].block.src;
        delete[] cnet->outs[i].block.tar;
        delete[] cnet->outs[i].block.delay;
        delete[] cnet->outs[i].block.weight;
    }
    delete[] cnet->outs;
    ///delete 网络内存空间
    delete cnet;
}
MGBrain::GSubNetUVA *MGBrain::copy_subnetuva_gpu(GSubNetUVA *cnet, GSubNetAddrsUVA *addr, int max_delay)
{
    GSubNetUVA *tmp = new GSubNetUVA;
    tmp->id = cnet->id;
    // tmp->max_delay=cnet->max_delay;
    tmp->neus_size = cnet->neus_size;
    tmp->syns_size = cnet->syns_size;
    tmp->outs_size = cnet->outs_size;
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
    tmp->neus.state = toGPU(cnet->neus.state, num); // poisson
    // 记录子网络神经元信息的相关地址
    addr->tar_exc_buffer = tmp->neus.I_buffer_exc;
    addr->tar_inh_buffer = tmp->neus.I_buffer_inh;
    addr->tar_fired_buffer = tmp->neus.Fired;
    addr->tar_last_fired_buffer = tmp->neus.Last_fired;
    // 拷贝内部突触
    num = cnet->syns_size;
    tmp->syns.src = toGPU(cnet->syns.src, num);
    tmp->syns.tar = toGPU(cnet->syns.tar, num);
    tmp->syns.weight = toGPU(cnet->syns.weight, num);
    tmp->syns.delay = toGPU(cnet->syns.delay, num);
    // 拷贝外部突触
    tmp->outs_size = cnet->outs_size;
    OUTSYNBlock *tmpouts = new OUTSYNBlock[tmp->outs_size];
    for (int i = 0; i < tmp->outs_size; i++)
    {
        tmpouts[i].syn_size = cnet->outs[i].syn_size;
        tmpouts[i].tar_id = cnet->outs[i].tar_id;
        tmpouts[i].block.src = toGPU(cnet->outs[i].block.src, cnet->outs[i].syn_size);
        tmpouts[i].block.tar = toGPU(cnet->outs[i].block.tar, cnet->outs[i].syn_size);
        tmpouts[i].block.weight = toGPU(cnet->outs[i].block.weight, cnet->outs[i].syn_size);
        tmpouts[i].block.delay = toGPU(cnet->outs[i].block.delay, cnet->outs[i].syn_size);
    }
    tmp->outs = toGPU(tmpouts, tmp->outs_size);
    GSubNetUVA *gnet = toGPU(tmp, 1);
    delete tmp;
    delete[] tmpouts;
    return gnet;
}
void MGBrain::gpuFreeGSubNetUVA(GSubNetUVA *gnet)
{
    GSubNetUVA *tmp = toCPU(gnet, 1);
    ///释放神经元显存空间
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
    gpuFree(tmp->neus.rate);
    gpuFree(tmp->neus.state);
    ///释放内部突触显存空间
    gpuFree(tmp->syns.src);
    gpuFree(tmp->syns.tar);
    gpuFree(tmp->syns.weight);
    gpuFree(tmp->syns.delay);
    OUTSYNBlock *outtmps = toCPU(tmp->outs, tmp->outs_size);
    ///释放外部突触显存空间
    for (int i = 0; i < tmp->outs_size; i++)
    {
        gpuFree(outtmps[i].block.src);
        gpuFree(outtmps[i].block.tar);
        gpuFree(outtmps[i].block.weight);
        gpuFree(outtmps[i].block.delay);
    }
    gpuFree(tmp->outs);
    delete[] outtmps;
    delete[] tmp;
    gpuFree(gnet);
}
