#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <mpi.h>
#include <nccl.h>

#include <algorithm>
#include <cstdlib>
#include <format>
#include <memory>
#include <source_location>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace detail {

// exception class with payload
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

// cuda exception payload type
struct cuda_status {
    cudaError_t status;
    operator cudaError_t() const noexcept { return status; }
    std::string to_string() const noexcept { return {cudaGetErrorString(status)}; }
};

// cuda driver exception payload type
struct cu_status {
    CUresult status;
    operator CUresult() const noexcept { return status; }
    std::string to_string() const noexcept {
        const char* str = nullptr;
        cuGetErrorString(status, &str);
        return {str};
    }
};

// MPI exception payload type
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

// NCCL exception payload type
struct nccl_status {
    ncclResult_t status;
    operator ncclResult_t() const noexcept { return status; }
    std::string to_string() const noexcept { return {ncclGetErrorString(status)}; }
};

// general payload type
struct runtime_status {
    std::string to_string() const noexcept { return "runtime error"; }
};

// possible exception types
using cuda_exception = detail::exception<cuda_status>;
using cuda_driver_exception = detail::exception<cu_status>;
using mpi_exception = detail::exception<mpi_status>;
using nccl_exception = detail::exception<nccl_status>;
using runtime_exception = detail::exception<runtime_status>;

// an exception is a variant of all possible exceptions
using exception = std::variant<cuda_exception, cuda_driver_exception, mpi_exception, nccl_exception, runtime_exception>;

// helper functions to throw an exception
template<typename R>
void throw_(R result, std::string msg, std::source_location loc = std::source_location::current()) {
    throw exception{std::in_place_type<detail::exception<R>>,
        msg + " (" + result.to_string() + ")", result, std::move(loc)};
}

inline void throw_(std::string msg, std::source_location loc = std::source_location::current()) {
    throw_(runtime_status{}, std::move(msg), std::move(loc));
}

template<typename R> requires (!std::is_convertible_v<R, std::string>)
void throw_(R result, std::source_location loc = std::source_location::current()) {
    throw exception{std::in_place_type<detail::exception<R>>,
        " (" + result.to_string() + ")", result, std::move(loc)};
}

// helper functions to check the return status of a library function call
template<typename R, typename T>
void check_status(R result, std::string&& msg, T expected, std::source_location&& loc) {
    if (result != expected)
        throw_(result, std::move(msg), std::move(loc));
}

inline void check_mpi(int result, std::string msg, int expected = MPI_SUCCESS,
    std::source_location loc = std::source_location::current()) {
    check_status(mpi_status{result}, std::move(msg), expected, std::move(loc));
}

inline void check_mpi(int result, int expected = MPI_SUCCESS,
    std::source_location loc = std::source_location::current()) {
    check_mpi(result, "", expected, std::move(loc));
}

inline void check_nccl(ncclResult_t result, std::string msg, ncclResult_t expected = ncclSuccess,
    std::source_location loc = std::source_location::current()) {
    check_status(nccl_status{result}, std::move(msg), expected, std::move(loc));
}

inline void check_nccl(ncclResult_t result, ncclResult_t expected = ncclSuccess,
    std::source_location loc = std::source_location::current()) {
    check_nccl(result, "", expected, std::move(loc));
}

inline void check_cuda(cudaError_t result, std::string msg, cudaError_t expected = cudaSuccess,
    std::source_location loc = std::source_location::current()) {
    check_status(cuda_status{result}, std::move(msg), expected, std::move(loc));
}

inline void check_cuda(cudaError_t result, cudaError_t expected = cudaSuccess,
    std::source_location loc = std::source_location::current()) {
    check_cuda(result, "", expected, std::move(loc));
}

inline void check_cu(CUresult result, std::string msg, CUresult expected = CUDA_SUCCESS,
    std::source_location loc = std::source_location::current()) {
    check_status(cu_status{result}, std::move(msg), expected, std::move(loc));
}

inline void check_cu(CUresult result, CUresult expected = CUDA_SUCCESS,
    std::source_location loc = std::source_location::current()) {
    check_cu(result, "", expected, std::move(loc));
}

// specialization of formatter to print exceptions
template<typename T>
struct std::formatter<detail::exception<T>, char> : std::formatter<string> {
    template<typename FmtContext>
    auto format(detail::exception<T> const & e, FmtContext& ctx) const {
        std::string buffer;
        std::format_to(std::back_inserter(buffer), "{}\n{}({}:{}), function `{}`",
            e.what(),
            e.where().file_name(),
            e.where().line(),
            e.where().column(),
            e.where().function_name()
        );
        return std::formatter<string>::format(std::move(buffer), ctx);
    }
};

template<>
struct std::formatter<::exception> : std::formatter<string> {
    template<typename FmtContext>
    auto format(::exception const & e, FmtContext& ctx) const {
        return std::formatter<string>::format(
            std::visit([&ctx]<typename E>(const E& e){
                std::string buffer;
                std::format_to(std::back_inserter(buffer), "{}", e);
                return buffer;
            },e),
            ctx);
    }
};

// RAII class for setting a cuda device
// restores the previously set device on destruction
struct device_guard {
    int old_device;

    device_guard(int dev) {
        check_cuda(cudaGetDevice(&old_device));
        check_cuda(cudaSetDevice(dev));
    }
    ~device_guard() {
        cudaSetDevice(old_device);
    }

    device_guard(device_guard const&) = delete;
    device_guard(device_guard &&) = delete;
    device_guard& operator=(device_guard const&) = delete;
    device_guard& operator=(device_guard &&) = delete;
};

// stores information about the number of local and global ranks
// and the number of associated devices. Creates a unique nccl id
// on construction.
struct world_info {
    int rank;
    int size;
    int local_rank;
    int local_size;
    int num_devices;
    std::vector<int> device_ids;
    ncclUniqueId id;

    world_info(MPI_Comm world_comm) {
        check_mpi(MPI_Comm_rank(world_comm, &rank));
        check_mpi(MPI_Comm_size(world_comm, &size));

        // split communicator to contain ranks with shared memory
        MPI_Comm local_comm;
        check_mpi(MPI_Comm_split_type(world_comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm));
        check_mpi(MPI_Comm_rank(local_comm, &local_rank));
        check_mpi(MPI_Comm_size(local_comm, &local_size));
        check_mpi(MPI_Comm_free(&local_comm));

        // get number of GPUs
        check_cuda(cudaGetDeviceCount(&num_devices));
        if (num_devices == 0) {
            throw_("no device found");
        }
        // 1 visible device per rank
        else if (num_devices == 1) {
            device_ids.push_back(0);
        }
        else {
            // N visible devices:
            // - local size must be N -> either exactly 1 device per local rank, or
            // - local size must be 1 -> N devices per rank
            if (num_devices < local_size)
                throw_("too many ranks per node");
            if (local_size > 1 && num_devices > local_size)
                throw_("too few ranks per node");
            if (local_size > 1) {
                device_ids.push_back(local_rank);
            }
            else {
                for (int i=0; i<num_devices; ++i)
                    device_ids.push_back(i);
            }
        }

        // generate unique nccl id and broadcast it
        if (rank == 0) check_nccl(ncclGetUniqueId(&id));
        check_mpi(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, world_comm));
    }
};

// specialize formatter for world_info
template<>
struct std::formatter<world_info> : std::formatter<string> {
    template<typename FmtContext>
    auto format(world_info const& w, FmtContext& ctx) const {
        std::string buffer;
        std::format_to(std::back_inserter(buffer),
            "world_info:\n"
            "  rank       = {:6}\n"
            "  size       = {:6}\n"
            "  local rank = {:6}\n"
            "  local size = {:6}\n"
            "  num gpus   = {:6}\n",
            w.rank, w.size, w.local_rank, w.local_size, w.num_devices);
        return std::formatter<string>::format(std::move(buffer), ctx);
    }
};

// RAII communicator
struct comm {
    // store state on the heap
    struct state {
        ncclComm_t comm;
        int device_id;
        struct deleter {
            void operator()(state* s) {
                device_guard g(s->device_id);
                ncclCommDestroy(s->comm);
                ::delete s;
            }
        };
    };
    std::unique_ptr<state, state::deleter> m_;

    comm() noexcept = default;
    comm(world_info const& info, int device_index = 0, bool blocking = false) : m_{::new state{}, state::deleter{}} {
        // check if device_index is valid, and set device id
        int devices_per_rank = info.device_ids.size();
        if (device_index < 0 || device_index >= devices_per_rank)
            throw_("invalid device_index");
        m_->device_id = info.device_ids[device_index];

        // activate device
        device_guard g(m_->device_id);
        // create communicator
        ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
        config.blocking = (blocking ? 1 : 0);
        auto state = ncclCommInitRankConfig(&(m_->comm),
            info.size*devices_per_rank,
            info.id,
            info.rank*devices_per_rank+device_index,
            &config);
        if (state != ncclSuccess && state != ncclInProgress)
            throw_(nccl_status{state});
        while(state == ncclInProgress) {
            check_nccl(ncclCommGetAsyncError(m_->comm, &state));
        }
        check_nccl(state);
    }
    comm(comm const&) = delete;
    comm& operator=(comm const&) = delete;
    comm(comm&& other) noexcept = default;
    comm& operator=(comm&& other) noexcept = default;

    operator ncclComm_t() const noexcept { return m_->comm; }
    int device_id() const noexcept { return m_->device_id; }

    [[nodiscard]] device_guard set_device() const noexcept { return {m_->device_id}; }
};

// helper functions for starting and ending a group
template<typename CommRange>
void wait(CommRange&& comm_range) {
    for (auto& c : comm_range) {
        auto g = c.set_device();
        ncclResult_t state = ncclInProgress;
        do {
            check_nccl(ncclCommGetAsyncError(c, &state));
        }
        while (state == ncclInProgress);
        check_nccl(state);
    }
}

template<typename CommRange>
auto ncclGroupEndNB(CommRange&& comm_range) {
    if (auto ret = ncclGroupEnd(); ret == ncclInProgress) {
        wait(std::forward<CommRange>(comm_range));
        return ncclSuccess;
    }
    else {
        return ret;
    }
}

inline auto ncclGroupEndNB(comm& c) {
    return ncclGroupEndNB(std::span<comm,1>{&c,1});
}

template<typename Comms>
void end_group(Comms&& comms) {
    check_nccl(ncclGroupEndNB(std::forward<Comms>(comms)));
}

inline void start_group() {
    check_nccl(ncclGroupStart());
}

enum class alloc {
    system,
    device
};

namespace detail {

template<typename T, alloc A>
struct deleter;

template<typename T>
struct deleter<T, alloc::system> {
    void operator()(T* ptr) { std::free(ptr); }
    static T* allocate(std::size_t size) {
        auto ptr = std::malloc(sizeof(T)*size);
        if (!ptr) throw_("system allocation failed");
        return reinterpret_cast<T*>(ptr);
    }
};

template<typename T>
struct deleter<T, alloc::device> {
    void operator()(T* ptr) { cudaFree(ptr); }
    static T* allocate(std::size_t size) {
        void* ptr = nullptr;
        check_cuda(cudaMalloc(&ptr, sizeof(T)*size));
        return reinterpret_cast<T*>(ptr);
    }
};

} // namespace detail

template<typename T, alloc A>
using buffer = std::unique_ptr<T[], detail::deleter<T, A>>;

template<typename T, alloc A>
auto make_buffer(std::size_t size) { return buffer<T, A>{ detail::deleter<T, A>::allocate(size) }; }
