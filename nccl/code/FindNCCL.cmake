if(NOT nccl_FOUND)
    find_path(nccl_INCLUDE_DIR NAMES nccl.h)
    find_library(nccl_LIBRARY NAMES nccl)

    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(nccl
        REQUIRED_VARS nccl_INCLUDE_DIR nccl_LIBRARY
        VERSION_VAR nccl_VERSION
    )

    if(nccl_FOUND)
        set(nccl_INCLUDE_DIRS ${nccl_INCLUDE_DIR})
        set(nccl_LIBRARIES ${nccl_LIBRARY})

        # Create imported target
        add_library(nccl::nccl UNKNOWN IMPORTED)
        set_target_properties(nccl::nccl PROPERTIES
            IMPORTED_LOCATION ${nccl_LIBRARY}
            INTERFACE_INCLUDE_DIRECTORIES ${nccl_INCLUDE_DIR}
        )
    endif()

    mark_as_advanced(nccl_INCLUDE_DIR nccl_LIBRARY)
endif()

