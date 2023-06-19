#pragma once
// #include "../../model/model.h"
#include "cneuron.h"
namespace MGBrain
{
    class SpikePipe
    {
    public:
        real *data;
        int index;
        int size;
        SpikePipe()
        {
            data = nullptr;
            index = -1;
            size = 0;
        }
        SpikePipe(int _size, real d)
        {
            initialize(_size, d);
        }
        ~SpikePipe()
        {
            index = 0;
            size = 0;
            delete data;
        }
        void initialize(int _size, real d)
        {
            size = _size;
            index = -1;
            data = new real[_size];
            std::fill_n(data, _size, d);
        };
        real push(real d)
        {
            index = (index + 1) % size;
            real pre = data[index];
            data[index] = d;
            return pre;
        }
        real front() const
        {
            int f = (index + 1) % size;
            return data[f];
        }
        real back() const
        {
            return data[index];
        }
    };
    class CBaseSynapse
    {
    public:
        int src;
        int tar;
        real weight;
        real delay;
        SpikePipe spikes;

    public:
        virtual real update(CBaseNeuron &pre, CBaseNeuron &post)
        {
            real i = weight * pre.isFired();
            return spikes.push(i) * weight;
        }
    };
    class STDPSynapse : public CBaseSynapse
    {
    private:
        real A_LTP = 0.1;
        real A_LTD = -0.01;
        real TAU_LTP = 17;
        real TAU_LTD = 34;
        real W_max = 40;
        real W_min = 0;

    public:
        virtual real update(CBaseNeuron &pre, CBaseNeuron &post)
        {

            real o = CBaseSynapse::update(pre, post);
            /// STDP
            CRecordNeuron &pre_r = *dynamic_cast<CRecordNeuron *>(&pre);
            CRecordNeuron &post_r = *dynamic_cast<CRecordNeuron *>(&post);
            if (pre.isFired() || post.isFired())
            {
                int dt = pre_r.getLastFired() - post_r.getLastFired();
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
                weight += dw;
                weight = (weight > W_max) ? W_max : ((weight < W_min) ? W_min : weight);
            }
            return o;
        }
    };
}