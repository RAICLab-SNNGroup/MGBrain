#include <iostream>
#include "../src/mgbrain.h"
int main(int argc,char* argv[]){
    parser p(argc,argv);
    std::cout<<p.getNum(parser::NSYN)<<std::endl;
    std::cout<<p.getNum(parser::NPART)<<std::endl;
    std::cout<<p.getStr(parser::MODEL)<<std::endl;
    float a=p.getFloat(parser::RATE);
    std::cout<<a<<std::endl;
    return 0;
}