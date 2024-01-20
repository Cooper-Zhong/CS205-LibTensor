#include "xt_adaptor.hpp"

#include <spdlog/spdlog.h>

namespace xtada {

    template<>
    bool equals(const float &a, const float &b) {
        return std::isnan(a) && std::isnan(b)
               || std::abs(a - b) < 1e-6;
    }
}
