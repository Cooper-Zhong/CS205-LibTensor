#include "benchmark/api.hpp"

namespace bm{
    std::vector<int> int_shape(const std::vector<size_t> &shape)
    {
        std::vector<int> ret;
        for (auto &i : shape)
        {
            ret.push_back(static_cast<int>(i));
        }
        return ret;
    }

    std::vector<std::vector<int>> int_slices(const std::vector<std::pair<size_t, size_t>> &slices)
    {
        std::vector<std::vector<int>> ret;

        for (auto &i : slices)
        {
            ret.push_back({static_cast<int>(i.first), static_cast<int>(i.second)});
        }
        return ret;
    }
}