#pragma once
#include <bit>
#include <cstdint>

namespace garch_map_capture {
inline std::uint64_t ordered(double value) {
    const auto bits=std::bit_cast<std::uint64_t>(value);
    return bits>>63 ? ~bits : bits | (std::uint64_t{1}<<63);
}
inline std::uint64_t ulps(double left,double right) {
    const auto a=ordered(left),b=ordered(right);
    return a>b ? a-b : b-a;
}
template<std::size_t N>
void array(const char* name,const std::array<double,N>& values) {
    std::printf("\"%s\":[",name);
    for(std::size_t i=0;i<N;++i) std::printf("%s\"%a\"",i ? "," : "",values[i]);
    std::printf("]");
}
}

#if defined(_MSC_VER)
#define SCIVOL_TEST_NOINLINE __declspec(noinline)
#else
#define SCIVOL_TEST_NOINLINE __attribute__((noinline))
#endif
template<std::size_t P,std::size_t Q,class Shape,std::size_t K>
SCIVOL_TEST_NOINLINE void capture_hessian(unsigned density,
    const std::array<double,K>& z,const std::array<double,K>& theta,
    const std::array<double,K>& gradient,const std::array<double,K*K>& theta_hessian,
    const std::array<double,K*K>& jacobian,const std::array<double,K*K>& runtime_jacobian,
    const std::array<double,K*K>& fixed,
    const std::array<double,K*K>& runtime,
    const vlk::experiment::generated::SimplexTailPoint& point,
    const vlk::experiment::generated::SimplexTailPoint& runtime_point) {
    using Transform=vlk::experiment::generated::GarchTransform<P,Q,Shape>;
    static_assert(K==Transform::parameters);
    std::size_t first=0;
    while(first<K*K && fixed[first]==runtime[first]) ++first;
    if(first==K*K) {
        std::printf("Hessian observer saw equal materialized arrays; original assertion remains active\n");
        std::fflush(stdout);return;
    }
    std::printf("{\"kind\":\"hessian\",\"p\":%zu,\"q\":%zu,\"density\":%u,\"width\":%zu,",
        P,Q,density,K);
    garch_map_capture::array("z",z);std::printf(",");
    garch_map_capture::array("theta",theta);std::printf(",");
    garch_map_capture::array("theta_gradient",gradient);std::printf(",");
    garch_map_capture::array("theta_hessian",theta_hessian);std::printf(",");
    garch_map_capture::array("jacobian",jacobian);std::printf(",");
    garch_map_capture::array("runtime_jacobian",runtime_jacobian);std::printf(",");
    garch_map_capture::array("fixed_hessian",fixed);std::printf(",");
    garch_map_capture::array("runtime_hessian",runtime);
    std::array<double,P+Q> weights{},shifted{},runtime_weights{},runtime_shifted{};
    for(std::size_t i=0;i<P+Q;++i) {
        weights[i]=point.weight(i);shifted[i]=point.shifted_weights[i];
        runtime_weights[i]=runtime_point.weight(i);
        runtime_shifted[i]=runtime_point.shifted_weights[i];
    }
    std::printf(",");garch_map_capture::array("weights",weights);
    std::printf(",");garch_map_capture::array("shifted_weights",shifted);
    std::printf(",");garch_map_capture::array("runtime_weights",runtime_weights);
    std::printf(",");garch_map_capture::array("runtime_shifted_weights",runtime_shifted);
    std::printf(",\"dominant\":%zu,\"denominator\":\"%a\",\"dominant_complement\":\"%a\",",
        point.dominant,point.denominator,point.dominant_complement);
    std::printf("\"runtime_dominant\":%zu,\"runtime_denominator\":\"%a\",\"runtime_dominant_complement\":\"%a\",",
        runtime_point.dominant,runtime_point.denominator,runtime_point.dominant_complement);
    std::printf("\"first_unequal\":{\"row\":%zu,\"column\":%zu,\"fixed\":\"%a\",\"runtime\":\"%a\",\"ulp\":%llu},",
        first/K,first%K,fixed[first],runtime[first],
        static_cast<unsigned long long>(garch_map_capture::ulps(fixed[first],runtime[first])));
    // These are observations of the local map, not the independent expectation.
    std::printf("\"local_map_hessians\":[");
    for(std::size_t output=0;output<K;++output) {
        std::array<double,K> unit{};
        std::array<double,K*K> zero{},curvature{};
        unit[output]=1;
        Transform::transform_hessian(theta.data(),z.data(),unit.data(),zero.data(),curvature.data(),&point);
        if(output) std::printf(",");
        std::printf("[");
        for(std::size_t i=0;i<K*K;++i) std::printf("%s\"%a\"",i ? "," : "",curvature[i]);
        std::printf("]");
    }
    std::printf("]}\n");std::fflush(stdout);
}

#undef SCIVOL_TEST_NOINLINE
