cmake_minimum_required(VERSION 3.0)
include("./utils.cmake")

set(MY_PROJECT_NAME "project" CACHE STRING "Project name")

set(MY_SELECTED_PLATFORM "linux" CACHE STRING "User defined selected platform for the compilation.")
set(IS_CROSS_COMPILATION FALSE CACHE BOOL "Indicates cross compilation")
set(MY_DEFAULT_BUILD_TYPE "Debug")

set(MY_ROOT_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE PATH "Root")
set(MY_COMMON_DIR ${MY_ROOT_SRC_DIR}/common CACHE PATH "Commons")
set(MY_SRC_DIR ${MY_ROOT_SRC_DIR}/src CACHE PATH "SRC")

# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${MY_ROOT_SRC_DIR}/bin)
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/debug)
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/release)

# set(CMAKE_DEBUG_POSTFIX "-d")
set(CMAKE_VERBOSE_MAKEFILE ON)
# execute_process(COMMAND "${CMAKE_COMMAND}" "-E" "environment")

include("./multiplatform.cmake")

# get the version number from version.id
file(STRINGS version.id VERSION_RAW)
string(REPLACE "version:" "" VERSION ${VERSION_RAW})

# make a list
string(REPLACE "." ";" VERSION_LIST ${VERSION})

# filter version numbers
list(GET VERSION_LIST 0 VERSION_MAJOR)
list(GET VERSION_LIST 1 VERSION_MINOR)
list(GET VERSION_LIST 2 VERSION_BUILD)

# define it for the source files
add_definitions(-DVERSION_MAJOR=${VERSION_MAJOR}
                -DVERSION_MINOR=${VERSION_MINOR}
                -DVERSION_BUILD=${VERSION_BUILD}
)
   

####################################################################################################
# Compilers definitions
####################################################################################################
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${MY_DEFAULT_BUILD_TYPE}' as none was specified.")
  set(CMAKE_BUILD_TYPE ${MY_DEFAULT_BUILD_TYPE} CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# set(CMAKE_BUILD_TYPE ${MY_DEFAULT_BUILD_TYPE} CACHE STRING "" FORCE)
message(STATUS "CMAKE_BUILD_TYPE: '${CMAKE_BUILD_TYPE}'")
message(STATUS "CMAKE_CONFIGURATION_TYPES: '${CMAKE_CONFIGURATION_TYPES}'")

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-Wall -Wno-reorder -Wno-unknown-pragmas")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_CUDA_FLAGS)
set(CMAKE_CUDA_FLAGS_DEBUG "-g")
set(CMAKE_CUDA_FLAGS_MINSIZEREL "-O1 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
set(CMAKE_CUDA_HOST_COMPILER)

set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS}
    -std=c++14
    -default-stream per-thread
    # -gencode=arch=compute_60,code=sm_60 \     #PASCAL
    # -gencode=arch=compute_61,code=sm_61 \ 
    # -gencode=arch=compute_62,code=sm_62 \ 
    # -gencode=arch=compute_70,code=sm_70 \     #VOLTA
    # -gencode=arch=compute_72,code=sm_72 \
    -gencode=arch=compute_75,code=sm_75       #TURING
)

if(MY_SELECTED_PLATFORM STREQUAL "linux")
    include("./app.linux.cmake")
    print_target_properties(app)
endif()

# print_cache (".*")¨
# print_cache (CMAKE_CXX_FLAGS)
# print_cache (CMAKE_CXX_STANDARD)
# print_cache (CMAKE_CUDA_FLAGS)
# print_cache (CUDA_NVCC_FLAGS)