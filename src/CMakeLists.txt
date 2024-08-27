find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(nccl REQUIRED)

add_executable(nccl_test nccl_test.cpp)

target_compile_options(nccl_test PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options -Wall,-Wextra>
)

target_link_libraries(nccl_test PRIVATE
    MPI::MPI_CXX
    CUDA::cudart
    CUDA::cuda_driver
    nccl::nccl
)
