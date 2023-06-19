#pragma once
#include "model.h"
#include "constgen.h"
namespace MGBrain{
    class ModelMaker
{
private:
    /* data */
public:
    static void make_brunel(Model &m, int nsyn)
    {
        float con_prob = 0.1;
        int pop_size = static_cast<int>(std::sqrt((float)nsyn / (con_prob * 0.5)));
        Population p = m.create_pop(pop_size * 5 / 10, NeuronType::POISSON, true);
        Population e = m.create_pop(pop_size * 4 / 10, NeuronType::LIFB, false);
        Population i = m.create_pop(pop_size * 1 / 10, NeuronType::LIFB, false);
        real w1 = 0.0001 * 20000 / pop_size;
        real w2 = -0.0005 * 20000 / pop_size;
        real d1 = 0.0016f;
        m.set_lif_const(ConstGen::gen_life_const2());
        m.connect(p, e, w1, d1, con_prob);
        m.connect(p, i, w1, d1, con_prob);
        m.connect(e, e, w1, d1, con_prob);
        m.connect(e, i, w1, d1, con_prob);
        m.connect(i, i, w2, d1, con_prob);
        m.connect(i, e, w2, d1, con_prob);
    }
    static void make_vogel(Model &m, int nsyn)
    {
        float con_prob=0.02;
        int pop_size=static_cast<int>(std::sqrt((float)nsyn/(con_prob)));
        Population pe = m.create_pop(pop_size * 8 / 10, NeuronType::LIFE, false);
        Population pi = m.create_pop(pop_size * 2 / 10, NeuronType::LIFE, false);
        real w1 = 0.4 * 16000000 / pop_size /pop_size ;
        real w2 = -5.1 * 16000000 / pop_size /pop_size ;
        real d1 = 0.0008f;
        m.set_lif_const(ConstGen::gen_life_const1());
        m.connect(pe, pe, w1, d1, con_prob);
        m.connect(pe, pi, w1, d1, con_prob);
        m.connect(pi, pe, w2, d1, con_prob);
        m.connect(pi, pi, w2, d1, con_prob);
    }
};
};

