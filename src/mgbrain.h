
#include "macro.h"
#include "model/model.h"
#include "model/network.h"
#include "model/netgen.h"
#include "model/maker.h"
#include "model/constgen.h"
#include "partition/parter.h"
#include "simulator/sim.h"
#include "util/parser.h"
typedef MGBrain::Config config;
typedef MGBrain::Model model;
typedef MGBrain::Population pop;
typedef MGBrain::Projection prj;
typedef MGBrain::Network network;
typedef MGBrain::NeuronType neuType;
typedef MGBrain::Neuron neuron;
typedef MGBrain::Synapse synapse;
typedef MGBrain::NetGen netgen;
typedef MGBrain::Partitioner parter;
typedef MGBrain::PartType PartType;
typedef MGBrain::SGSimultor sgsim;
// typedef MGBrain::MGSimulatorUVA mgsimuva;
typedef MGBrain::MGSimulator mgsim;
typedef MGBrain::NeuronType NeuType;
typedef MGBrain::real real;
typedef MGBrain::PartitionAnalysis panalysis;
typedef MGBrain::ConstGen constgen;
typedef MGBrain::ModelMaker maker;
typedef MGBrain::ParamsParser parser;
typedef MGBrain::timer timer;
