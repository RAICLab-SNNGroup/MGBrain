#include <metis.h>
#include <vector>
namespace MGBrain{
    class MetisUtil
    {
    public:
        /// @brief METIS算法划分一般图的方法封装
        /// @param nCells 图的顶点数
        /// @param nPart 需要划分的区域数
        /// @param xadj 邻接列表索引
        /// @param adjncy 邻接顶点列表数据
        /// @param part 划分数组（结果）
        /// @return 
        static int metis_part_normal_graph(idx_t nCells, int nPart, std::vector<idx_t> &xadj, std::vector<idx_t> &adjncy, std::vector<idx_t> &part);
        /// @brief METIS算法划分带权重图的方法封装
        /// @param nvtxs 图的顶点数
        /// @param nparts 需要划分的区域数
        /// @param xadj 邻接列表索引
        /// @param adjncy 邻接顶点列表
        /// @param vwgt 点权重列表
        /// @param adjwgt 边权重列表
        /// @param part 划分数组（结果）
        /// @return 
        static int metis_part_weighted_graph(int nvtxs, int nparts, std::vector<idx_t> &xadj, std::vector<idx_t> &adjncy, std::vector<idx_t> &vwgt,
                              std::vector<idx_t> &adjwgt, std::vector<idx_t> &part);
    };
    
};