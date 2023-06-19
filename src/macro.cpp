#include "macro.h"
namespace MGBrain{
    real Config::STEP = 0.001;
    real Config::DT = 0.001;
    real Config::DW = 0.01;
    bool Config::STDP = false;
    bool Config::SINGLE_GPU =false;
    bool Config::PEER_ACCESS =true;
    bool Config::SHUFFLE_GEN=false;
    bool Config::FIRE_CHECK=false;
    bool Config::DENSE_SPIKE=false;
    bool Config::STAGGERED =false;
    float Config::MIN_GPUMEM=3.0f;
    bool Config::SEQUENCE=false;
};