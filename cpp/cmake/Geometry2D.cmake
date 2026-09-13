mojive_dependency(libtess2)
add_library(mojive_tessellator STATIC
    "${libtess2_SOURCE_DIR}/Source/bucketalloc.c"
    "${libtess2_SOURCE_DIR}/Source/dict.c"
    "${libtess2_SOURCE_DIR}/Source/geom.c"
    "${libtess2_SOURCE_DIR}/Source/mesh.c"
    "${libtess2_SOURCE_DIR}/Source/priorityq.c"
    "${libtess2_SOURCE_DIR}/Source/sweep.c"
    "${libtess2_SOURCE_DIR}/Source/tess.c")
set_target_properties(mojive_tessellator PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(mojive_tessellator PUBLIC "${libtess2_SOURCE_DIR}/Include")
add_library(mojive_geometry2d STATIC
    src/geometry2d/Path.cpp src/geometry2d/Stroke.cpp src/geometry2d/Tessellator.cpp)
set_target_properties(mojive_geometry2d PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(mojive_geometry2d PUBLIC mojive_render_contract PRIVATE mojive_tessellator)
