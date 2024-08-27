cmake_minimum_required(VERSION 3.24)

set(CMAKE_CUDA_ARCHITECTURES "90" CACHE STRING "CUDA architectures")

project(user_lab_day_2024 CXX)

#------------------------------------------------------------------------------
# Setup Output directories
#------------------------------------------------------------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_BINARY_DIR}/bin CACHE PATH "Single Directory for all Executables."
)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
    ${PROJECT_BINARY_DIR}/lib CACHE PATH "Single Directory for all Libraries"
)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY
    ${PROJECT_BINARY_DIR}/lib CACHE PATH "Single Directory for all static libraries."
)

option(BUILD_SLIDES "Wether to build slides." ON)
option(BUILD_SLIDES_PDF "Wether to build slides." ON)
option(BUILD_CODE "Wether to build code." OFF)
option(BUILD_JOBREPORT "Wether to build the jobreport tool." OFF)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (${BUILD_SLIDES})
    include(cmake/Marp.cmake)
endif()

include(CTest)

add_subdirectory(nccl)

if (${BUILD_JOBREPORT})
    add_subdirectory(ext/alps-jobreport)
    set(COPY_DESTINATION "${CMAKE_SOURCE_DIR}/jobreport")
    add_custom_command(
        OUTPUT "${COPY_DESTINATION}"
        COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:jobreport>" "${COPY_DESTINATION}"
        COMMENT "Copying jobreport to top-level directory"
        DEPENDS jobreport
    )
    add_custom_target(copy_jobreport ALL DEPENDS "${COPY_DESTINATION}")
endif()
