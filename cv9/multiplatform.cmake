# set(MY_SELECTED_PLATFORM "jetson" CACHE STRING "User defined selected platform for the compilation.")
# # set(MY_SELECTED_PLATFORM_VALUES "linux;windows;jetson" CACHE INTERNAL "List of possible values for the SelectedPlatform.")
# # set_property(CACHE MY_SELECTED_PLATFORM PROPERTY STRINGS ${MY_SELECTED_PLATFORM_VALUES})
# message(STATUS "MY_SELECTED_PLATFORM='${MY_SELECTED_PLATFORM}'")

if(MY_SELECTED_PLATFORM STREQUAL "linux")
    
    set(CMAKE_SYSTEM_NAME Linux)

    project(MY_PROJECT_NAME C CXX CUDA)

    
    set(CMAKE_C_COMPILER /usr/bin/gcc-8)
    set(CMAKE_CXX_COMPILER /usr/bin/g++-8)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)


    # find_package(OpenGL REQUIRED)
    find_package(CUDA REQUIRED)
    # find_package(X11 REQUIRED)

elseif (MY_SELECTED_PLATFORM STREQUAL "windows")
    
    set(CMAKE_SYSTEM_NAME Windows)

endif()

print_cache("^\(OPENGL\)|\(OpenGL\).*")
print_cache("^\(CUDA\)|\(Cuda\).*")
print_cache("^\(X11.*LIB$\)|\(X11_LIBRARIES\)")

# print_cache(".*")
