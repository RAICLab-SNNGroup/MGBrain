//
// Created by 15838 on 2023/3/7.
//
#pragma once

#include <chrono>
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <math.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <time.h>
#include <stdio.h>
#include <array>
#include <random>
#include <memory.h>

// #define STREAM_EVENT_RECORD
// #define FULL_DEVICE_PERFORMANCE
// #define ENABLE_PEER_ACCESS
// #define UVA_MULTINET
// #define TIME_DEBUG
namespace MGBrain
{

    const float CONN_TYPE_O2O = 0;
    const float CONN_TYPE_FULL = 1;
#ifdef DOUBLE_REAL
    typedef double real;
#else
    typedef float real;
    typedef unsigned long sid;
#endif
    enum NeuronType
    {
        RAW,
        LIF0,
        POISSON,
        LIF,
        LIFE,
        LIFB
    };
    enum SynapseType
    {
        STATIC,
        STDP
    };
    enum PartType
    {
        METIS,
        MODEL,
        BSIM,
        SIMPLE,
    };
    /// @brief LIF神经元模型常量参数
    struct LIFConsts
    {
        real P22;
        real P11exc;
        real P11inh;
        real P21exc;
        real P21inh;
        real C_m;
        real V_reset;
        real V_rest;
        real V_thresh;
        real Tau_m;
        real Tau_exc;
        real Tau_inh;
        real I_offset;
        real Tau_refrac;
    };
    /// @brief STDP突触模型常量参数
    struct STDPConsts
    {
        real A_LTP;
        real A_LTD;
        real TAU_LTP;
        real TAU_LTD;
        real W_max;
        real W_min;
    };
    class Config
    {
    public:
        static real STEP;
        /// @brief 时间片长度 ms
        static real DT;
        /// @brief 权重变化量
        static real DW;
        static bool STDP;
        static bool SINGLE_GPU;
        static bool PEER_ACCESS;
        static bool SHUFFLE_GEN;
        static bool FIRE_CHECK;
        static bool DENSE_SPIKE;
        static bool STAGGERED;
        static float MIN_GPUMEM;
        static bool SEQUENCE;
        
        /// @brief 设置时间片长度
        /// @param time_step 时间片时长 ms
        static void set_timestep(real time_step)
        {
            STEP = time_step;
            DT = time_step;
        }
        /// @brief 获取时间片数
        /// @param time 输入仿真时长 s
        /// @return
        static int get_steps(real time)
        {
            return round(time * 1000 / STEP);
        }
        static void set_dw(real dw)
        {
            DW = dw;
        }
        static void enable_stdp()
        {
            STDP = true;
        }
        static void enable_single_gpu()
        {
            SINGLE_GPU = true;
            PEER_ACCESS = false;
        }
        static void enable_shuffle_gen()
        {
            SHUFFLE_GEN = true;
        }
        static void enable_fire_check(){
            FIRE_CHECK=true;
        }
        static void enable_dense_spike(){
            DENSE_SPIKE=true;
        }
        static void disable_peer_access()
        {
            PEER_ACCESS = false;
        }
        static void enable_staggered(){
            STAGGERED=true;
        }
        static void set_min_gpumem(float mem){
            MIN_GPUMEM=mem;
        }
        static void enable_sequence(){
            SEQUENCE=true;
        }
    };

    using namespace std::chrono;

    class timer
    {
        time_point<high_resolution_clock> s;

    public:
        timer() { s = high_resolution_clock::now(); }
        double stop()
        {
            return duration_cast<microseconds>(high_resolution_clock::now() - s).count() * 1e-6;
        }
    };

}
