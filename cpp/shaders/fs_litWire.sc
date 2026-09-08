$input v_color0, v_normal, v_depth, v_texcoord0, v_world, v_litMaterial, v_litCube, v_litIdentity, v_bary
#include <bgfx_shader.sh>
SAMPLER2D(s_image, 0);
SAMPLERCUBE(s_cube, 1);
#include "lighting.sh"
#include "reflections.sh"
#define WIREFRAME 1
#include "litFragment.sh"
