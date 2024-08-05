if (${BUILD_SLIDES})
    add_marp_slides(nccl nccl.md)
endif()



if (${BUILD_CODE})
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/code")
    find_package(MPI REQUIRED)

    find_package(MPI REQUIRED)
    find_package(CUDAToolkit REQUIRED)
    find_package(NCCL REQUIRED)

    add_executable(nccl_test code/nccl_test.cpp)

    target_compile_options(nccl_test PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options -Wall,-Wextra>
    )

    target_link_libraries(nccl_test PRIVATE
        MPI::MPI_CXX
        CUDA::cudart
        nccl:nccl
    )
endif()
