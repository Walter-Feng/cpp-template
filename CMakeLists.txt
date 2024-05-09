cmake_minimum_required(VERSION 3.27)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

### TODO: replace the PROJECT VERSION ###
set(PROJECT_VERSION 1.0.0)

### TODO: replace the project name ###
project(cpptemplate
        VERSION ${PROJECT_VERSION}
        LANGUAGES CXX CUDA
        DESCRIPTION "A generic template for C++ projects")

IF (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.git)
    FIND_PACKAGE(Git)
    IF (GIT_FOUND)
        EXECUTE_PROCESS(
                COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                OUTPUT_VARIABLE "TEMPLATE_BUILD_VERSION"
                ERROR_QUIET
                OUTPUT_STRIP_TRAILING_WHITESPACE)
        MESSAGE(STATUS "Git version: ${TEMPLATE_BUILD_VERSION}")
    ELSE (GIT_FOUND)
        SET(TEMPLATE_BUILD_VERSION PROJECT_VERSION)
    ENDIF (GIT_FOUND)
ENDIF (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.git)

CONFIGURE_FILE(
        ${CMAKE_CURRENT_SOURCE_DIR}/src/version/version.h.in
        ${CMAKE_CURRENT_BINARY_DIR}/include/version.h
)

include(GNUInstallDirs)

include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/data)

enable_testing()

# Change the standard to your need if necessary
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

add_definitions(-DTEMPLATE_ROOT=\"${CMAKE_CURRENT_SOURCE_DIR}\")

set(CMAKE_STATIC_LIBRARY_PREFIX "")
# TODO: Rename
add_library(libTemplate
        src/util/util.cpp
)

#TODO: Rename
add_executable(template
        src/main.cpp
)

target_link_libraries(template libTemplate)
####################
#  find Libraries  #
####################

#####MPI######
find_package(
        MPI
        REQUIRED
)

if (${MPI_FOUND})
    include_directories(${MPI_INCLUDE_DIRS})
    target_link_libraries(libTemplate ${MPI_LIBRARIES})
endif ()

#####Boost#####
find_package(
        Boost
        REQUIRED
        COMPONENTS mpi serialization
)
if (${Boost_FOUND})
    include_directories(${Boost_INCLUDE_DIRS})
    target_link_libraries(libTemplate ${Boost_LIBRARIES})
endif ()

#####HDF5######

find_package(
        HDF5
)

if (${HDF5_FOUND})
    message(STATUS "Found HDF5 Library,")
    message(STATUS "HDF5_VERSION: " ${HDF5_VERSION})
    message(STATUS "HDF5_LIBRARIES   : " ${HDF5_LIBRARIES})
    message(STATUS "HDF5_INCLUDE_DIRS: " ${HDF5_INCLUDE_DIRS})

    include_directories(${HDF5_INCLUDE_DIRS})
    target_link_libraries(libTemplate ${HDF5_LIBRARIES})
endif ()

######OpenMP######

FIND_PACKAGE(OpenMP)

if (${OPENMP_FOUND})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif ()


######CUDA######
set(CMAKE_CUDA_ARCHITECTURES 80)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(libTemplate CUDA::cublas CUDA::cudart)

#########nccl#########
find_package(NCCL REQUIRED)
if (${NCCL_FOUND})
    include_directories(${NCCL_INCLUDE_DIRS})
    target_link_libraries(libTemplate ${NCCL_LIBRARIES})
endif ()


################################
#  Compile external libraries  #
################################


#######Catch2#######
add_subdirectory(external/Catch2)
include_directories("external/Catch2/src")
target_link_libraries(libTemplate Catch2WithMain)

########args#########
add_subdirectory(external/args)
include_directories("external/args")
target_link_libraries(libTemplate args)

#########json#########
add_subdirectory(external/json)
include_directories("external/json/include")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

####################
#       Test       #
####################

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test)

# TODO: Rename
add_executable(test_template
        src/util/vectorization_test.cpp
        src/util/print_test.cu
)

target_link_libraries(test_template libTemplate)