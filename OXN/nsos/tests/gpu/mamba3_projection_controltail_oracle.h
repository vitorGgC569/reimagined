#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace controltail_oracle {
inline std::uint32_t bits(float value) {
    std::uint32_t result;std::memcpy(&result,&value,sizeof(result));return result;
}
// Independent scalar form of the original wave32 dot contract: assigning K in
// ascending order to K%32 lanes, then the five shuffle-down reduction stages.
inline float dot(const std::vector<float>& a,const std::vector<float>& b,
    bool ta,bool tb,int rows,int cols,int k,int row,int col,int begin) {
    std::array<float,32> lanes{};
    for(int x=begin;x<k;++x) {
        const int lane=(x-begin)%32;
        lanes[lane]=std::fma(a[ta?std::size_t(x)*rows+row:std::size_t(row)*k+x],
            b[tb?std::size_t(col)*k+x:std::size_t(x)*cols+col],lanes[lane]);
    }
    for(int delta:{16,8,4,2,1})
        for(int lane=0;lane<32-delta;++lane) lanes[lane]+=lanes[lane+delta];
    return lanes[0];
}
}
