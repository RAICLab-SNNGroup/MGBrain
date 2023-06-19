#include <metis.h>
#include <vector>
#include "../model/network.h"
#include "metisutil.h"
#include "analysis.h"
#include "calculator.h"
namespace MGBrain
{

    class Partitioner
    {
        private:
        /// @brief 按照网络结构使用METIS算法划分
        /// @param network 网络结构
        /// @param nparts 划分区域数量
        /// @param part 划分结果
        static void part_with_net(Network &network, int nparts,std::vector<int>& part);
        /// @brief 按照MGBrain负载模型使用METIS算法划分
        /// @param network 网络结构
        /// @param nparts 划分区域数量
        /// @param part 划分结果
        static void part_with_model(Network &network, int nparts,std::vector<int>& part);
        /// @brief 按照Bsim划分方式划分,每个区域的突触数量大致相同
        /// @param network 网络结构
        /// @param nparts 划分区域数量
        /// @param part 划分结果
        static void part_with_bsim(Network &network, int nparts,std::vector<int>& part);
        /// @brief 按照顺序直接划分，每个区域顶点数量大致相同
        /// @param network 网络结构
        /// @param nparts 划分区域数量
        /// @param part 划分结果
        static void part_with_simple(Network &network, int nparts,std::vector<int>& part);
        
        public:
        static std::vector<int> part_network(Network &network, int nparts, PartType type);
    };
}