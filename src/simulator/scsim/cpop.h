
#include "../../macro.h"
#pragma once
namespace MGBrain
{
    struct Pop
    {
        /// @brief 族群id
        int id;
        /// @brief 族群内神经元数量
        int num;
        /// @brief 是否是输入神经元
        bool source;
        /// @brief 神经元类型
        NeuronType type;
        /// @brief 神经元列表
        std::vector<int> neurons;
        Pop(int _id,int _num,bool _source,NeuronType _type):id(_id),num(_num),source(_source),type(_type){
            neurons.resize(_num);
        }
    };
};