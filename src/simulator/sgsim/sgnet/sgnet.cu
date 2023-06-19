#include "sgnet.cuh"

namespace MGBrain
{
    // LIF神经元使用常量
    __constant__ real DT=0.0001;
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
    __constant__ real Tau_refrac = 4;

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
        return buffer[id * MAX_DELAY + step % MAX_DELAY];
    }
    __device__ inline void pushSpike(real *buffer, int nid, int step, real value)
    {
        atomicAdd(&buffer[nid * MAX_DELAY + step], value);
    }
    __global__ void simNeus(GNetwork *net, int step)
    {
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        if (net->neus.poisson[id])
        {
            /// 仿真泊松神经元
            float rand = curand_uniform(&(net->neus.state[id]));
            net->neus.Fired[id] = net->neus.rate[id] > rand;
            if (net->neus.Fired[id])
            {
                net->neus.Fire_cnt[id]++;
                net->neus.Last_fired[id] = step;
            }
            // if(id==0){
            //     printf("%d poisson' rate:%f\n",id,net->neus.rate[id]);
            // }

        }else{
            /// 仿真LIF神经元
            net->neus.Fired[id] = false;
            if (net->neus.Refrac_state[id] > 0)
            {
                --net->neus.Refrac_state[id];
            }
            else
            {
                net->neus.V_m[id] = _P22 * net->neus.V_m[id] + net->neus.I_exc[id] * _P21exc + net->neus.I_inh[id] * _P21inh;
                net->neus.V_m[id] += (1 - _P22) * (I_offset * Tau_m / C_m + V_rest);
                net->neus.I_exc[id] *= _P11exc;
                net->neus.I_inh[id] *= _P11inh;
                if (net->neus.V_m[id] >= V_thresh)
                {
                    net->neus.Fired[id] = true;
                    net->neus.Fire_cnt[id]++;
                    net->neus.V_m[id] = V_reset;
                    net->neus.Last_fired[id] = step;
                    net->neus.Refrac_state[id] = std::round(Tau_refrac/DT);
                }
                else
                {
                    net->neus.I_exc[id] += getInput(net->neus.I_buffer_exc, id, step);
                    net->neus.I_inh[id] += getInput(net->neus.I_buffer_inh, id, step);
                }
            }
            // if(id==12){
            //     printf("%d neu' vm0:%f\n",id,net->neus.V_m[id]);
            // }
            
        }
    }
    __global__ void simSyns(GNetwork *net, int step)
    {
        int ref = blockIdx.x * blockDim.x + threadIdx.x;
        int src = net->syns.src[ref];
        int tar = net->syns.tar[ref];
        real weight = net->syns.weight[ref];
        int delay = net->syns.delay[ref];
        if (!net->neus.Fired[src] && !net->neus.Fired[tar])
            return;
        // 发放脉冲
        if (net->neus.Fired[net->syns.src[ref]])
        {
            if (weight > 0)
                pushSpike(net->neus.I_buffer_exc, tar, (step + delay) % MAX_DELAY, weight);
            else
                pushSpike(net->neus.I_buffer_inh, tar, (step + delay) % MAX_DELAY, weight);
        }
        // STDP权重更新
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
    void simStep(GNetwork *gnet,int step,int blocksize,int neunum,int synnum){
        simNeus<<<neunum / blocksize + 1, neunum % blocksize>>>(gnet, step);
        simSyns<<<synnum / blocksize + 1, synnum % blocksize>>>(gnet, step);
    }
    void init_gpu_consts(int max_delay,real dt,LIFConsts* lifconst,STDPConsts* stdpconst){
        CUDACHECK(cudaMemcpyToSymbol(MAX_DELAY,&max_delay,sizeof(int)));
        CUDACHECK(cudaMemcpyToSymbol(DT,&dt,sizeof(int)));
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
            CUDACHECK(cudaMemcpyToSymbol(Tau_refrac,&(lifconst->Tau_refrac),sizeof(real)));
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
/// gnetwork
void MGBrain::initNeus(GNetwork* cnet,int max_delay)
{
    int nums = cnet->neus_size;
    cnet->neus.ids =new int[nums]();
    cnet->neus.V_m = new real[nums]();
    cnet->neus.Fired = new bool[nums]();
    cnet->neus.Fire_cnt = new int[nums]();
    cnet->neus.Last_fired = new int[nums]();
    cnet->neus.I_exc = new real[nums]();
    cnet->neus.I_inh = new real[nums]();
    cnet->neus.I_buffer_exc = new real[nums * max_delay]();
    cnet->neus.I_buffer_inh = new real[nums * max_delay]();
    cnet->neus.Refrac_state = new int[nums]();
    cnet->neus.poisson = new bool[nums]();
    cnet->neus.rate = new real[nums]();
    cnet->neus.state = new curandState[nums]();
}
void MGBrain::initSyns(GNetwork* cnet)
{
    int nums = cnet->syns_size;
    cnet->syns.src = new int[nums]();
    cnet->syns.tar = new int[nums]();
    cnet->syns.weight = new real[nums]();
    cnet->syns.delay = new int[nums]();
}
void MGBrain::initLIFConsts(LIFConsts* lifconst){
    lifconst->P22= 0.90483743;
    lifconst->P11exc = 0.951229453;
    lifconst->P11inh = 0.967216074;
    lifconst->P21exc = 0.0799999982;
    lifconst->P21inh = 0.0599999987;
    lifconst->V_rest = 0;
    lifconst->V_reset = 0;
    lifconst->C_m = 0.25;
    lifconst->Tau_m = 10.0;
    lifconst->V_thresh = 15;
    lifconst->I_offset = 0;
    lifconst->Tau_refrac = 0.0004;
}
void MGBrain::initSTDPConsts(STDPConsts* stdpconst){
    stdpconst->A_LTP = 0.1;
    stdpconst->A_LTD = -0.01;
    stdpconst->TAU_LTP = 17;
    stdpconst->TAU_LTD = 34;
    stdpconst->W_max = 40;
    stdpconst->W_min = 0;
    int max_delay = 10;
}
void MGBrain::freeNet(GNetwork* cnet)
{
    /// 清除神经元数据
    delete[] cnet->neus.ids;
    delete[] cnet->neus.V_m;
    delete[] cnet->neus.Fire_cnt;
    delete[] cnet->neus.Fired;
    delete[] cnet->neus.Last_fired;
    delete[] cnet->neus.I_exc;
    delete[] cnet->neus.I_inh;
    delete[] cnet->neus.I_buffer_exc;
    delete[] cnet->neus.I_buffer_inh;
    delete[] cnet->neus.Refrac_state;
    // poisson
    delete[] cnet->neus.rate;
    delete[] cnet->neus.poisson;
    delete[] cnet->neus.state;
    // 清除突触数据
    delete[] cnet->syns.src;
    delete[] cnet->syns.tar;
    delete[] cnet->syns.delay;
    delete[] cnet->syns.weight;
    delete[] cnet;
}
MGBrain::GNetwork *MGBrain::copyNetToGPU(GNetwork *cnet,int max_delay)
{
    GNetwork *tmp = new GNetwork;
    int num = cnet->neus_size;
    tmp->neus.ids = toGPU(cnet->neus.ids, num);
    tmp->neus.V_m = toGPU(cnet->neus.V_m, num);
    tmp->neus.I_exc = toGPU(cnet->neus.I_exc, num);
    tmp->neus.I_inh = toGPU(cnet->neus.I_inh, num);
    tmp->neus.I_buffer_exc = toGPU(cnet->neus.I_buffer_exc, num * max_delay);
    tmp->neus.I_buffer_inh = toGPU(cnet->neus.I_buffer_inh, num * max_delay);
    tmp->neus.Fired = toGPU(cnet->neus.Fired, num);
    tmp->neus.Fire_cnt = toGPU(cnet->neus.Fire_cnt, num);
    tmp->neus.Last_fired = toGPU(cnet->neus.Last_fired, num);
    tmp->neus.Refrac_state = toGPU(cnet->neus.Refrac_state, num);

    tmp->neus.rate = toGPU(cnet->neus.rate, num);
    tmp->neus.poisson = toGPU(cnet->neus.poisson, num);
    tmp->neus.state = toGPU(cnet->neus.state, num);

    num = cnet->syns_size;
    tmp->syns.src = toGPU(cnet->syns.src, num);
    tmp->syns.tar = toGPU(cnet->syns.tar, num);
    tmp->syns.weight = toGPU(cnet->syns.weight, num);
    tmp->syns.delay = toGPU(cnet->syns.delay, num);
    GNetwork *gnet = toGPU(tmp, 1);
    delete tmp;
    return gnet;
}
void MGBrain::gpuFreeNet(GNetwork *gnet)
{
    GNetwork *tmp = toCPU(gnet, 1);
    gpuFree(tmp->neus.ids);
    gpuFree(tmp->neus.V_m);
    gpuFree(tmp->neus.I_exc);
    gpuFree(tmp->neus.I_inh);
    gpuFree(tmp->neus.I_buffer_exc);
    gpuFree(tmp->neus.I_buffer_inh);
    gpuFree(tmp->neus.Fired);
    gpuFree(tmp->neus.Fire_cnt);
    gpuFree(tmp->neus.Last_fired);
    gpuFree(tmp->neus.Refrac_state);
    gpuFree(tmp->neus.state);
    gpuFree(tmp->neus.rate);
    gpuFree(tmp->neus.poisson);
    gpuFree(tmp->syns.src);
    gpuFree(tmp->syns.tar);
    gpuFree(tmp->syns.weight);
    gpuFree(tmp->syns.delay);
    delete[] tmp;
    gpuFree(gnet);
}
// NEUBlock *initLIFData(int num, int steps)
// {
//     NEUBlock *block = new NEUBlock;
//     block->num = num;
//     block->V_m = new real[num]();
//     block->Fired = new bool[num]();
//     block->Fire_cnt = new int[num]();
//     block->Last_fired = new int[num]();
//     block->I_exc = new real[num]();
//     block->I_inh = new real[num]();
//     block->I_buffer_exc = new real[num * Max_delay]();
//     block->I_buffer_inh = new real[num * Max_delay]();
//     block->Refrac_state = new int[num]();

//     block->steps = steps;
//     block->source = new bool[num]();
//     block->rand = new real[num * steps]();
//     block->rate = new real[num]();
//     return block;
// }

// void freeLIFData(NEUBlock *block)
// {
//     delete[] block->V_m;
//     delete[] block->Fire_cnt;
//     delete[] block->Fired;
//     delete[] block->Last_fired;
//     delete[] block->I_exc;
//     delete[] block->I_inh;
//     delete[] block->I_buffer_exc;
//     delete[] block->I_buffer_inh;
//     delete[] block->Refrac_state;

//     delete[] block->rand;
//     delete[] block->rate;
//     delete[] block->source;
//     delete block;
// }
// NEUBlock *copyLIF2GPU(NEUBlock *cblock)
// {
//     NEUBlock *gblock;
//     NEUBlock *tmp = new NEUBlock;
//     int num = cblock->num;
//     tmp->num = num;

//     tmp->V_m = toGPU(cblock->V_m, num);
//     tmp->I_exc = toGPU(cblock->I_exc, num);
//     tmp->I_inh = toGPU(cblock->I_inh, num);
//     tmp->I_buffer_exc = toGPU(cblock->I_buffer_exc, num * Max_delay);
//     tmp->I_buffer_inh = toGPU(cblock->I_buffer_inh, num * Max_delay);
//     tmp->Fired = toGPU(cblock->Fired, num);
//     tmp->Fire_cnt = toGPU(cblock->Fire_cnt, num);
//     tmp->Last_fired = toGPU(cblock->Last_fired, num);
//     tmp->Refrac_state = toGPU(cblock->Refrac_state, num);

//     int steps = cblock->steps;
//     tmp->steps = steps;
//     tmp->rate = toGPU(cblock->rate, num);
//     tmp->source = toGPU(cblock->source, num);
//     tmp->rand = toGPU(cblock->rand, num * steps);

//     gblock = toGPU(tmp, 1);
//     return gblock;
// }
// void freeGLIF(NEUBlock *gblock)
// {
//     NEUBlock *tmp = toCPU(gblock, 1);
//     gpuFree(tmp->V_m);
//     gpuFree(tmp->I_exc);
//     gpuFree(tmp->I_inh);
//     gpuFree(tmp->I_buffer_exc);
//     gpuFree(tmp->I_buffer_inh);
//     gpuFree(tmp->Fired);
//     gpuFree(tmp->Fire_cnt);
//     gpuFree(tmp->Last_fired);
//     gpuFree(tmp->Refrac_state);

//     gpuFree(tmp->rand);
//     gpuFree(tmp->rate);
//     gpuFree(tmp->source);
//     gpuFree(gblock);
//     delete tmp;
// }