//
// Created by 15838 on 2023/3/3.
//
typedef float real;
#include <vector>
#include <iostream>
#define a 0.02
#define b 0.2
#define c -65
#define d 8
#define si 1000
#define ei 2000
#define ii 40
#define vm -65
#define um 0
#define T 4000
#define dt 0.01

class Timer
{
public:
    Timer(const char* funcName)
    {
        _funcname = funcName;
        _begin = clock();
    }

    ~Timer()
    {
        _end = clock();
        _elapsed = _end - _begin;

        printf("Function name: %s\nElapsed : %f\n", _funcname ,_elapsed);
    }

private:
    double        _begin;
    double        _end;
    double        _elapsed;
    const char*   _funcname;
};
struct IzhiP{
    real v;
    real u;
    IzhiP(real _v, real _u): v(_v), u(_u){}
    IzhiP(IzhiP& p): v(p.v), u(p.u){}
    IzhiP operator*(real k){
        return IzhiP(v * k, u * k);
    }
    IzhiP operator+(IzhiP p){
        return IzhiP(v + p.v, u + p.u);
    }
};

IzhiP izhi(real i, IzhiP p){
    real dv=0.04*p.v*p.v+5*p.v+140-p.u+i;
    real du=a*(b*p.v-p.u);
    return IzhiP(dv, du);
}

IzhiP izhiEulerP(real h, real i, IzhiP p){
    IzhiP dp= izhi(i, p);
    return p+dp*h;
}
IzhiP izhiRKP(real h, real i, IzhiP p){
    real _6=(1.0f/6.0f);
    real _2=0.5;
    IzhiP k1= izhi(i, p) * h;
    IzhiP k2= izhi(i, p + (k1 * _2)) * h;
    IzhiP k3= izhi(i, p + (k2 * _2)) * h;
    IzhiP k4= izhi(i, p + k3) * h;
    IzhiP v= p + (k1 + (k2 * 2.0) + (k3 * 2.0) + k4) * _6;
    return v;
}

real izhiFv(real i, real v, real u){
    return 0.04*v*v+5*v+140-u+i;
}

real izhiFu(real i, real v, real u){
    return a*(b*v-u);
}
IzhiP izhiRK(real h, real i, IzhiP p){

    real _2=0.5*h;
    real _6=(1.0/6.0)*h;
    real fv1= izhiFv(i, p.v, p.u);
    real fu1= izhiFu(i, p.v, p.u);
    real fv2= izhiFv(i, p.v + _2 * fv1, p.u + _2 * fu1);
    real fu2= izhiFu(i, p.v + _2 * fv1, p.u + _2 * fu1);
    real fv3= izhiFv(i, p.v + _2 * fv2, p.u + _2 * fu2);
    real fu3= izhiFu(i, p.v + _2 * fv2, p.u + _2 * fu2);
    real fv4= izhiFv(i, p.v + h * fv3, p.u + h * fu3);
    real fu4= izhiFu(i, p.v + h * fv3, p.u + h * fu3);
    real nv=p.v+_6*(fv1+2*fv2+2*fv3+fv4);
    real nu=p.u+_6*(fu1+2*fu2+2*fu3+fu4);
    return IzhiP(nv, nu);
}
void checkRK(){
    Timer timer(__FUNCTION__ );
    IzhiP p=IzhiP(vm, um);
    for(int i=0;i<10000000;i++)
        p= izhiRK(0.01, 0.1, p);
}
void checkRKP(){
    Timer timer(__FUNCTION__ );
    IzhiP p=IzhiP(vm, um);
    for(int i=0;i<10000000;i++)
        p= izhiRKP(0.01, 0.1, p);
}
void checkRKRKP(){
    IzhiP p=IzhiP(vm, um);
    IzhiP p2=IzhiP(vm, um);
    for(int i=0;i<1000;i++){
        p= izhiRKP(0.01, 0.1, p);
        p2= izhiRK(0.01,0.1,p2);
        if(abs(p.v-p2.v)>1e-5){
            std::cout<<"error"<<std::endl;
        }
    }


}
void gen(){
    IzhiP p=IzhiP(vm, um);
    real I=0;
    std::vector<real> res1;
    res1.push_back(p.v);
    for(int i=1;i<T;i++){
        if(i==si){
            I=ii;
        }
        if(i==ei){
            I=0;
        }
        p= izhiRK(dt, I, p);
        if(p.v>30){
            res1.push_back(30);
            p.v=c;
            p.u=p.u+d;
        }else{
            res1.push_back(p.v);
        }
    }
    for(int i=0;i<res1.size();i++){
        std::cout<<res1[i]<<",";
    }
    std::cout<<std::endl;
}

int main(){
    gen();
//    checkRK();
//    checkRKRKP();
    return 0;
}

