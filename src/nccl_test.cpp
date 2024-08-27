#include "utility.hpp"

#include <barrier>
#include <future>
#include <iostream>
#include <memory>
#include <syncstream>
#include <thread>

inline constexpr std::size_t buffer_size = 256;
inline constexpr alloc allocation = alloc::system;
inline constexpr bool use_threads = true;

// setup communicators, streams and buffers and do an allreduce
template<typename Barrier>
void test(world_info const& info, int thread_id, bool blocking, Barrier barrier) {
    // synchronize at startup
    barrier();

    // calculate number of GPUs for this thread
    int devices_per_rank = info.device_ids.size();
    int num_comms = ((devices_per_rank > 1 && thread_id >= 0) ? 1 : devices_per_rank);
    thread_id = thread_id < 0 ? 0 : thread_id;
    int device_rank_offset = devices_per_rank*info.rank + thread_id;

    // communicators (one per GPU) and streams
    std::vector<comm> comms;
    comms.reserve(num_comms);
    std::vector<cudaStream_t> streams(num_comms);

    // make communicators
    start_group();
    for (int i=0; i<num_comms; ++i) {
        std::osyncstream(std::cout) << std::format("rank = {:6}, local rank = {:2}, thread = {:2}: "
            "making communicator for device {:6}, blocking = {:5}\n",
            info.rank, info.local_rank, thread_id, device_rank_offset+i, blocking);
        comms.emplace_back(info, thread_id+i, blocking);
        device_guard g(comms.back().device_id());
        check_cuda(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
    }
    end_group(comms);
    wait(comms);
    barrier();

    // make buffers
    using T = float;
    using buffer_type = buffer<T, allocation>;
    auto make_buffers = [&]() {
        std::vector<buffer_type> vec;
        vec.reserve(num_comms);
        for (auto& c : comms) {
            vec.push_back(make_buffer<T, allocation>(buffer_size));
            device_guard g(c.device_id());
            check_cuda(cudaMemset(vec.back().get(), 0, sizeof(T)*buffer_size));
        }
        return vec;
    };
    auto send_allreduce_buffers = make_buffers();
    auto recv_allreduce_buffers = make_buffers();

    // allreduce
    start_group();
    for (auto i=0u; i<comms.size(); ++i) {
        auto& c = comms[i];
        device_guard g(c.device_id());
        if constexpr (allocation == alloc::system) {
            send_allreduce_buffers[i][0] = 1;
        }
        check_nccl(ncclAllReduce(send_allreduce_buffers[i].get(), recv_allreduce_buffers[i].get(),
            buffer_size, ncclFloat, ncclSum, c, streams[i]));
    }
    end_group(comms);
    wait(comms);
    for (auto i=0u; i<comms.size(); ++i) {
        auto& c = comms[i];
        device_guard g(c.device_id());
        check_cuda(cudaStreamSynchronize(streams[i]));
    }

    // print reduced buffer
    if constexpr (allocation == alloc::system) {
        for (int i=0; i<num_comms; ++i) {
            std::osyncstream(std::cout) << std::format("rank = {:6}, local rank = {:2}, thread = {:2}, device = {:6}:  "
                "allreduce = [{:4}, ... ]\n",
                info.rank, info.local_rank, thread_id, device_rank_offset+i, recv_allreduce_buffers[i][0]);
        }
    }

    // destroy streams
    for (auto i=0u; i<comms.size(); ++i) {
        auto& c = comms[i];
        device_guard g(c.device_id());
        check_cuda(cudaStreamDestroy(streams[i]));
    }

    // communicators are destroyed at end of scope
};


int main(int argc, char** argv) {
    try {
        // initialize MPI with serialized thread support
        int required_thread_support = MPI_THREAD_SERIALIZED;
        int provided_thread_support = 0;
        check_mpi(MPI_Init_thread(&argc, &argv, required_thread_support, &provided_thread_support));
        check_mpi(provided_thread_support, required_thread_support);
        check_mpi(MPI_Barrier(MPI_COMM_WORLD));

        // query world
        world_info info(MPI_COMM_WORLD);
        if (info.rank == 0)
            std::osyncstream(std::cout) << std::format("\n\n{}\n", info);

        if (info.device_ids.size() > 1) {
            if (use_threads) {
                if (info.rank == 0)
                    std::osyncstream(std::cout)  << std::format("\nrunning with 4 GPUs per rank: 1 GPU per thread\n"
                                                                  "==============================================\n\n");
                auto num_threads = info.device_ids.size();
                auto thread_barrier = std::barrier(num_threads, [](){ check_mpi(MPI_Barrier(MPI_COMM_WORLD)); });
                auto barrier = [&]() { thread_barrier.arrive_and_wait(); };

                // use packaged tasks to get proper exception transport to main
                std::vector<std::future<void>> results;
                results.reserve(num_threads);
                {
                    std::vector<std::jthread> threads;
                    threads.reserve(num_threads);
                    for (auto i=0u; i<num_threads; ++i) {
                        std::packaged_task t{test<decltype(barrier)>};
                        results.emplace_back(t.get_future());
                        threads.emplace_back(std::move(t), std::cref(info), i, false, barrier);
                    }
                }
                for (auto& f : results) f.get();
            }
            else {
                if (info.rank == 0)
                    std::osyncstream(std::cout)  << std::format("\nrunning with 4 GPUs per rank: single thread\n"
                                                                  "===========================================\n\n");
                test(info, -1, false, [](){ check_mpi(MPI_Barrier(MPI_COMM_WORLD)); });
            }
        }
        else {
            if (info.rank == 0)
                std::osyncstream(std::cout)  << std::format("\nrunning with one GPU per rank\n"
                                                              "=============================\n\n");
            test(info, -1, false, [](){ check_mpi(MPI_Barrier(MPI_COMM_WORLD)); });
        }

        // finalize
        check_mpi(MPI_Barrier(MPI_COMM_WORLD));
        check_mpi(MPI_Finalize());
    }
    catch (const exception& e) {
        // print exception
        std::cout << std::format("caught exception:\n{}\n\n", e);
        return 1;
    }
    return 0;
}
