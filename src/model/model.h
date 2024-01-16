//
// Created by 15838 on 2023/3/6.
//
#pragma once
#include "../macro.h"
#ifndef SIMPLECPUSIM_MODEL_H
#define SIMPLECPUSIM_MODEL_H
namespace MGBrain
{

    /// @brief 族群模型
    struct Population
    {
        /// @brief 族群id
        int id;
        /// @brief 族群内神经元数量
        int num;
        /// @brief 是否是输入神经元
        bool source;
        /// @brief 神经元类型
        NeuronType type;
        Population(int _id,int _num,bool _source,NeuronType _type):id(_id),num(_num),source(_source),type(_type){

        }
    };
    struct Projection
    {
        /// @brief 源族群id
        int src;
        /// @brief 目标族群id
        int tar;
        /// @brief 权重范围
        std::array<real, 2> wrange;
        /// @brief 延迟范围
        std::array<real, 2> drange;
        /// @brief 连接类型
        float ctype;
        /// @brief 突触类型
        SynapseType stype;
        Projection(int _src,int _tar,std::array<real,2> _wrange,std::array<real,2> _drange,float _ctype,SynapseType _stype):src(_src),tar(_tar),wrange(_wrange),drange(_drange),ctype(_ctype),stype(_stype){}
    };

    class Model
    {
    public:
        std::vector<Population> pops;
        std::vector<Projection> pros;
        std::array<real,30> lifconst;
        bool nlifconst=false;
        std::array<real,6> stdpconst;
        bool nstdpconst=false;
        int indexer;
        Model();
        void set_lif_const(std::array<real,30> lifconst);
        void set_stdp_const(std::array<real,6> stdpconst);
        Population &create_pop(int num, NeuronType type, bool isSource);
        bool connect(Population &src, Population tar, std::array<real, 2> _wrange, std::array<real, 2> _drange, float type,SynapseType _stype);
        bool connect(Population &src, Population tar, real weight, real delay, float type,SynapseType _stype);
        bool connect(Population &src, Population tar, std::array<real, 2> _wrange, std::array<real, 2> _drange, float type);
        bool connect(Population &src, Population tar, real weight, real delay, float type);
    };
}

#endif // SIMPLECPUSIM_MODEL_H
