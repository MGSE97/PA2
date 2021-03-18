file(GLOB MY_APP_SOURCE_FILES
    ${MY_SRC_DIR}/*.h
    ${MY_SRC_DIR}/*.cpp
    ${MY_SRC_DIR}/*.cuh
    ${MY_SRC_DIR}/*.cu
)

add_executable(app ${MY_APP_SOURCE_FILES})

# We need to explicitly state that we need all CUDA files in the app to be built with -dc as the member functions could be called by other libraries and executables
set_target_properties(app 
    PROPERTIES 
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
)

target_include_directories(app PRIVATE 
	${CUDA_INCLUDE_DIRS}
    ${MY_SRC_DIR}
    ${MY_COMMON_DIR}/pg/
    ${MY_COMMON_DIR}/Glew/${MY_SELECTED_PLATFORM}/inc/
    ${MY_COMMON_DIR}/FreeGlut/${MY_SELECTED_PLATFORM}/inc/
    ${MY_COMMON_DIR}/FreeImage/${MY_SELECTED_PLATFORM}/inc/
    ${MY_COMMON_DIR}/CUDA_10-2_SDK/${MY_SELECTED_PLATFORM}/inc/
)

find_package(OpenGL REQUIRED)
find_package(X11 REQUIRED)

target_link_libraries(app PRIVATE "stdc++fs"

    ${OPENGL_LIBRARY} # filled by "find_package(OpenGL REQUIRED)"
    ${CUDA_LIBRARIES}
    
    ${MY_COMMON_DIR}/Glew/${MY_SELECTED_PLATFORM}/lib/libGLEW.a
    ${MY_COMMON_DIR}/FreeGlut/${MY_SELECTED_PLATFORM}/lib/libglut.a
    ${MY_COMMON_DIR}/FreeImage/${MY_SELECTED_PLATFORM}/lib/libfreeimage.a

    ${X11_LIBRARIES}  # filled by "find_package(X11 REQUIRED)"  
    ${X11_Xrandr_LIB}
    ${X11_Xxf86vm_LIB}
    ${X11_Xi_LIB}
)

#find_package(OpenMP)
#    if(OpenMP_CXX_FOUND)
#    target_link_libraries(app PUBLIC OpenMP::OpenMP_CXX)
#endif()

# add_custom_command(
#     TARGET app
#     POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E copy ${MY_SRC_DIR}/config.json $<TARGET_FILE_DIR:app>/config.json
#     COMMENT "Copying to output directory"
# )