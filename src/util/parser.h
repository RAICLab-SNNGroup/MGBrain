#include <string>
#include <unordered_map>
#include <iostream>
#include <sstream>
namespace MGBrain
{
    class ParamsParser
    {
    private:
        std::unordered_map<std::string, std::string> params;

    public:
        // 突触数量
        static std::string NSYN;
        // 划分区域数量
        static std::string NPART;
        // 划分方式
        static std::string MODEL;
        // 输出文件名
        static std::string FILE;
        // 输出文件路径
        static std::string PREFIX;
        static std::string RATE;
        ParamsParser(int args, char *argv[])
        {
            // 默认突触数量:10000
            params[NSYN] = "10000";
            // 默认划分区域:1
            params[NPART] = "1";
            // 默认划分方式:model
            params[MODEL] = "model";
            // 默认输出文件
            params[FILE] = "tmp.txt";
            // 默认输出文件路径
            params[PREFIX] = "../../benchdata/";
            params[RATE] = "0.002";
            // 获取输入参数
            for (int i = 1; i < args; i++)
            {
                char *arg = argv[i];
                std::string name;
                std::string value;
                bool first = true;
                for (int j = 0; arg[j] != '\0'; j++)
                {
                    if (arg[j] == '=')
                    {
                        first = false;
                        continue;
                    }
                    if (first)
                        name.push_back(arg[j]);
                    else
                        value.push_back(arg[j]);
                }
                params[name] = value;
            }
        }
        ~ParamsParser() {}
        std::string getStr(std::string name)
        {
            if (!params.count(name))
                return "";
            return params[name];
        }
        size_t getNum(std::string name)
        {
            if (!params.count(name))
                return -1;
            std::stringstream ss(params[name]);
            size_t res;
            ss >> res;
            return res;
        }
        float getFloat(std::string name)
        {
            if (!params.count(name))
                return -1;
            std::stringstream ss(params[name]);
            float res;
            ss >> res;
            return res;
        }
        bool getBool(std::string name)
        {
            if (!params.count(name))
                return false;
            return (params[name] == "y");
        }
    };
    std::string ParamsParser::NSYN = "--nsyn";
    std::string ParamsParser::NPART = "--npart";
    std::string ParamsParser::MODEL = "--model";
    std::string ParamsParser::FILE = "--file";
    std::string ParamsParser::PREFIX = "--prefix";
    std::string ParamsParser::RATE = "--rate";
};
