function(print_target_properties tgt)
    if(NOT TARGET ${tgt})
        message(STATUS "There is no target named '${tgt}'")
        return()
    endif()

    # this list of properties can be extended as needed
    set(CMAKE_PROPERTY_LIST SOURCE_DIR BINARY_DIR COMPILE_DEFINITIONS
             COMPILE_OPTIONS INCLUDE_DIRECTORIES LINK_LIBRARIES)

    message(STATUS "\r\n= TARGET CONFIGURATIONS (${tgt}) ====================================================================================================\r\n ")

    foreach (prop ${CMAKE_PROPERTY_LIST})
        get_property(propval TARGET ${tgt} PROPERTY ${prop} SET)
        if (propval)
            get_target_property(propval ${tgt} ${prop})
            string(REGEX REPLACE  ";" ";\r\n\t" newvalues "${propval}")
            message (STATUS "${prop} = ${newvalues}")
        endif()
    endforeach(prop)

    message(STATUS "\r\n=========================================================================================================================================\r\n ")

endfunction(print_target_properties)

function(print_cache regex)
    get_cmake_property(_variableNames VARIABLES)

    message(STATUS "\r\n= CACHE ('${regex}') ==================================================================================================\r\n ")
    foreach (_variableName ${_variableNames})
        if(_variableName MATCHES ${regex})
            #message(STATUS "${_variableName}=${${_variableName}}")
            string(REGEX REPLACE  ";" ";\r\n\t" newvalues "${${_variableName}}")
            message(STATUS "${_variableName}=${newvalues}")
        endif()
    endforeach()

    message(STATUS "\r\n======================================================================================================================\r\n ")
endfunction(print_cache)

function(print_glob globname)
    message(STATUS "\r\n= GLOB FILES ('${globname}') ====================================================================================================\r\n ")
    foreach (item ${${globname}})
        message (STATUS "${item}")
    endforeach(item)
    message(STATUS "\r\n======================================================================================================================\r\n ")
endfunction(print_glob)
