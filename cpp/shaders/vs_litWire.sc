$input a_position, a_normal, a_texcoord0, a_texcoord1, i_data0, i_data1, i_data2, i_data3, i_data4, i_data5, i_data6, i_data7
$output v_color0, v_normal, v_depth, v_texcoord0, v_world, v_litMaterial, v_litCube, v_litIdentity, v_bary
#include <bgfx_shader.sh>
#define WIREFRAME 1
#include "litVertex.sh"
