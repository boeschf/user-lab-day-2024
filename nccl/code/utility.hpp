#pragma once

#include <cuda_runtime_api.h>
#include <mpi.h>
#include <nccl.h>

#include <iostream>
//#include <format>
#include <source_location>
#include <string>
#include <utility>
#include <variant>

namespace detail {

template<typename T>
class exception {
  private:
    std::string msg_;
    T payload_;
    std::source_location loc_;

  public:
    exception(std::string msg, T payload, std::source_location loc = std::source_location::current())
    : msg_{std::move(msg)}
    , payload_{std::move(payload)}
    , loc_{std::move(loc)}
    {}

    auto &      what()        noexcept { return msg_; }
    auto const& what()  const noexcept { return msg_; }
    auto &      data()        noexcept { return payload_; }
    auto const& data()  const noexcept { return payload_; }
    auto const& where() const noexcept { return loc_; }
};

} // namespace detail

struct cuda_status {
    cudaError_t status;
    operator cudaError_t() const noexcept { return status; }
    std::string to_string() const noexcept { return {cudaGetErrorString (status)}; }
};

struct mpi_status {
    int status;
    operator int() const noexcept { return status; }
    std::string to_string() const noexcept {
        char err_str[MPI_MAX_ERROR_STRING];
        int len;
        MPI_Error_string(status, err_str, &len);
        return {err_str, err_str+len};
    }
};

struct nccl_status {
    ncclResult_t status;
    operator ncclResult_t() const noexcept { return status; }
    std::string to_string() const noexcept { return {ncclGetErrorString(status)}; }
};

struct runtime_status {
    std::string to_string() const noexcept { return "runtime error"; }
};

using cuda_exception = detail::exception<cuda_status>;
using mpi_exception = detail::exception<mpi_status>;
using nccl_exception = detail::exception<nccl_status>;
using runtime_exception = detail::exception<runtime_status>;

using exception = std::variant<cuda_exception, mpi_exception, nccl_exception, runtime_exception>;

template<typename R>
void throw_(R result, std::string&& msg,std::source_location&& loc) {
    throw exception{std::in_place_type<detail::exception<R>>,
        msg + " (" + result.to_string() + ")", result, std::move(loc)};
}

template<typename R, typename T>
void check_status(R result, std::string&& msg, T expected, std::source_location&& loc) {
    if (result != expected)
        throw_(result, msg + " (" + result.to_string() + ")", std::move(loc));
}

inline void check_mpi(int result, std::string msg, int expected = MPI_SUCCESS,
    std::source_location loc = std::source_location::current()) {
    check_status(mpi_status{result}, std::move(msg), expected, std::move(loc));
}

inline void check_nccl(ncclResult_t result, std::string msg, ncclResult_t expected = ncclSuccess,
    std::source_location loc = std::source_location::current()) {
    check_status(nccl_status{result}, std::move(msg), expected, std::move(loc));
}

inline void check_cuda(cudaError_t result, std::string msg, cudaError_t expected = cudaSuccess,
    std::source_location loc = std::source_location::current()) {
    check_status(cuda_status{result}, std::move(msg), expected, std::move(loc));
}

//namespace std {
//template<typename T>
//struct formatter<detail::exception<T>, char> : formatter<string> {
//    template<typename FmtContext>
//    auto format(detail::exception<T> const & e, FmtContext& ctx) const {
//        string buffer;
//        format_to(back_inserter(buffer), "{}\n{}({}:{}), function `{}`",
//            e.what(),
//            e.where().file_name(),
//            e.where().line(),
//            e.where().column(),
//            e.where().function_name()
//        );
//        return formatter<string>::format(std::move(buffer), ctx);
//    }
//};
//
//struct formatter<exception, char> : formatter<string> {
//    template<typename FmtContext>
//    auto format(exception const & e, FmtContext& ctx) const {
//        std::visit([]<template T>(T const& e) {
//          return formatter<T>::format(e, ctx);
//          }, e);
//    }
//};
//} // namespace std
//
//
//template<typename T>
//std::ostream& operator<<(std::ostream& os, detail::exception<T> const & e) {
//    os << std::format("{}", e);
//    return os;
//}
//

