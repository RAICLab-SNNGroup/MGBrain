# MGBrain

> MGBrain is a multi-GPU based spiking neural network simulator, which is optimized in terms of communication and calculation.

### Prerequisites
* C++: 14
* G++: 7.5
* CMAKE: 3.16
* CUDA: 10.0
* Metis,GKLib

### Usage
A typical SNN is defined as follows:
```c++
#include "../src/mgbrain.h"
int main()
{
    //configuration
    config::set_timestep(0.0001);
    config::enable_single_gpu();

    ///population modeling
    model m;
    float con_prob=0.1;
    //set neuron parameters
    m.set_lif_const(constgen::gen_life_const2());
    //build Population
    pop p=m.create_pop(10,neuType::POISSON,true);//Define a population of 10 POISSON neurons.
    pop e=m.create_pop(10,neuType::LIF0,false);//Define a population of 10 LIF neurons.
    pop i=m.create_pop(10,neuType::LIF0,false);//Define a population of 10 LIF neurons.
    //build Connection
    m.connect(p,e,0.1,0.0014f,con_prob);// randomly connect population. p->e with prob. 10%. weight:0.1 delay:0.0014
    m.connect(e,i,0.1,0.0014f,con_prob);// randomly connect population. e->i with prob. 10%. weight:0.1 delay:0.0014

    ///generate network
    network net;
    netgen::gen_net(net,m);
    
    ///partition network
    int npart=2;//partition number
    std::vector<int> part=parter::part_network(net,
    npart,
    PartType::MODEL//partition method
    );
    
    ///initialize the simulator
    mgsim sim(net,part,npart);
    
    ///start simulation
    sim.simulate(1);
    return 0;
}
```
Put your customized examples under `examples` directory, and add corresponding content of `CMakeLists.txt` file. Then use cmake to compile.

### Bench
All experimental test programs in the article are in the `bench` directory and all test scripts are in the `shell` directory. you can 
