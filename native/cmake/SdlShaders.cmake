# Shader tools are build-time executables, isolated from both renderer libraries.
include(ExternalProject)
set(shader_tools "${CMAKE_CURRENT_BINARY_DIR}/shader-tools")
ExternalProject_Add(mojive_sdl_shader_tools
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shader_tools"
    BINARY_DIR "${shader_tools}"
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    BUILD_COMMAND ${CMAKE_COMMAND} --build . --config Release --parallel 4
        --target glslang-standalone spirv-cross
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS "${shader_tools}/bin/glslang${CMAKE_EXECUTABLE_SUFFIX}"
        "${shader_tools}/bin/spirv-cross${CMAKE_EXECUTABLE_SUFFIX}")
if(WIN32)
    find_program(MOJIVE_DXC dxc REQUIRED)
endif()
set(sdl_shader_outputs)
foreach(entry IN ITEMS scene_vertex color_fragment data_fragment ui_vertex ui_fragment)
    if(entry MATCHES "vertex$")
        set(stage vert)
        set(profile vs_6_0)
    else()
        set(stage frag)
        set(profile ps_6_0)
    endif()
    set(base "${CMAKE_CURRENT_BINARY_DIR}/shaders/${entry}")
    set(source "${CMAKE_CURRENT_SOURCE_DIR}/shaders/sdl/${entry}.${stage}")
    add_custom_command(OUTPUT "${base}.spv" "${base}.msl" "${base}.hlsl"
        COMMAND "${shader_tools}/bin/glslang${CMAKE_EXECUTABLE_SUFFIX}" -V
            --target-env vulkan1.0 -S ${stage} "${source}" -o "${base}.spv"
        COMMAND "${shader_tools}/bin/spirv-cross${CMAKE_EXECUTABLE_SUFFIX}" "${base}.spv"
            --msl --msl-version 20100 --rename-entry-point main ${entry} ${stage} --output "${base}.msl"
        COMMAND "${shader_tools}/bin/spirv-cross${CMAKE_EXECUTABLE_SUFFIX}" "${base}.spv"
            --hlsl --shader-model 60 --output "${base}.hlsl"
        DEPENDS mojive_sdl_shader_tools "${source}" VERBATIM)
    list(APPEND sdl_shader_outputs "${base}.spv" "${base}.msl" "${base}.hlsl")
    if(WIN32)
        add_custom_command(OUTPUT "${base}.dxil"
            COMMAND "${MOJIVE_DXC}" -T ${profile} -E main -O3 -Fo "${base}.dxil" "${base}.hlsl"
            DEPENDS "${base}.hlsl" VERBATIM)
        list(APPEND sdl_shader_outputs "${base}.dxil")
    endif()
endforeach()
add_custom_target(mojive_sdl_shaders ALL DEPENDS ${sdl_shader_outputs})
add_dependencies(mojive_backend_sdl mojive_sdl_shaders)
