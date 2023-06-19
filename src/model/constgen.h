#pragma once
#include "../macro.h"

namespace MGBrain
{
    class ConstGen
    {
    public:
        static std::array<real, 30> gen_life_const(real v_init, real v_rest, real v_reset, real cm, real tau_m, real tau_refrac, real tau_syn_E, real tau_syn_I, real v_thresh, real i_offset)
        {
            real dt = Config::DT;
            real rm = 1.0;
            real _Cm, _CE, _CI, _v_tmp, _i_offset, _C_E, _C_I;

            if (fabs(cm) > 0.00001)
            {
                std::cout << "cm" << std::endl;
                rm = tau_m / cm;
            }
            if (tau_m > 0)
            {
                _Cm = exp(-dt / tau_m);
            }
            else
            {
                _Cm = 0.0;
            }

            if (tau_syn_E > 0)
            {
                _CE = exp(-dt / tau_syn_E);
            }
            else
            {
                _CE = 0.0;
            }

            if (tau_syn_I > 0)
            {
                _CI = exp(-dt / tau_syn_I);
            }
            else
            {
                _CI = 0.0;
            }

            _v_tmp = _i_offset * rm + v_rest;
            _v_tmp *= (1 - _Cm);

            _C_E = rm * tau_syn_E / (tau_syn_E - tau_m);
            _C_I = rm * tau_syn_I / (tau_syn_I - tau_m);

            _C_E = _C_E * (_CE - _Cm);
            _C_I = _C_I * (_CI - _Cm);
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
            std::array<real, 30> life{};
            life[0] = dt;
            life[1] = v_init;
            life[2] = v_rest;
            life[3] = v_reset;
            life[4] = v_thresh;
            life[5] = cm;
            life[6] = tau_m;
            life[7] = i_offset;
            life[8] = tau_refrac;
            life[10] = tau_syn_E;
            life[11] = tau_syn_I;
            life[12] = _C_E;
            life[13] = _C_I;
            life[14] = _v_tmp;
            life[15] = _CE;
            life[16] = _CI;
            life[17] = _Cm;
            life[21] = 0.951229453f;
            life[22] = 0.967216074f;
            life[23] = 0.0799999982f;
            life[24] = 0.0599999987f;
            life[25] = 0.90483743f;
            return life;
        }
        static std::array<real, 30> gen_lif0_const(real v_init, real v_rest, real v_reset, real cm, real tau_m, real tau_refrac, real tau_syn_E, real tau_syn_I, real v_thresh, real i_offset)
        {
            std::array<real, 30> life{};
            life[0] = Config::DT;
            life[1] = v_init;
            life[2] = v_rest;
            life[3] = v_reset;
            life[4] = v_thresh;
            life[5] = cm;
            life[6] = tau_m;
            life[7] = i_offset;
            life[8] = tau_refrac;
            life[10] = tau_syn_E;
            life[11] = tau_syn_I;
            life[12] = 0.9999f;
            life[13] = 0.9999f;
            life[14] = 0.0f;
            life[15] = 0.9999f;
            life[16] = 0.9999f;
            life[17] = 0.0f;
            life[21] = 0.951229453f;
            life[22] = 0.967216074f;
            life[23] = 0.0799999982f;
            life[24] = 0.0599999987f;
            life[25] = 0.90483743f;
            return life;
        }
        static std::array<real, 30> gen_life_const1()
        {
            return gen_life_const(-0.06f, -0.06f, -0.06f, 0, 0, 0.005f, 1, 1, -0.05f, 0);
        }
        static std::array<real, 30> gen_life_const2()
        {
            return gen_life_const(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.002f, 1.0f, 1.0f, 0.02f, 0.0f);
        }
        static std::array<real, 30> gen_life_const3()
        {
            return gen_life_const(0.0f, 0.0f, 0.0f, 0.25f, 10.0f, 0.0004f, 1.0f, 1.0f, 15.0f, 0.0f);
        }

        static std::array<real, 6> gen_stdp_const(real a_ltp, real a_ltd, real tau_ltp, real tau_ltd, real w_max, real w_min)
        {
            return {a_ltp, a_ltd, tau_ltp, tau_ltd, w_max, w_min};
        }
        static std::array<real, 6> gen_stdp_const1()
        {
            return {0.1, -0.01, 17, 34, 40, 0};
        }
        static std::array<real, 6> gen_stdp_const2()
        {
            return {};
        }
    };
};