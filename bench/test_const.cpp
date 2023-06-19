#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <time.h>
#include "../src/mgbrain.h"
MGBrain::real random0001(){
    return rand() % (999 + 1) / (float)(999 + 1);
}

int main()
{
    config::set_timestep(0.0001);
    // std::array<real,30>life2=constgen::gen_life_const(0, 0, 0, 0, 0, 0.002f, 1, 1, 0.02f, 0);
    // std::array<real,30>life2=constgen::gen_life_const(-0.06f, -0.06f, -0.06f, 0, 0, 0.005f, 1, 1, -0.05f, 0);
    std::array<real,30> life2=constgen::gen_life_const1();
    for(int i=0;i<30;i++){
        std::cout<<i<<":"<<life2[i]<<std::endl;
    }
    return 0;
}