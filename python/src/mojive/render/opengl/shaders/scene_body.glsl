#ifndef DEBUG_VIEW
#define DEBUG_VIEW 0
#endif

#define OVERDRAW_STEP (1.0 / 16.0)

#include "common.glsl"
#include "lighting.glsl"

in VertexData {
    vec3 world;
    vec3 normal;
    vec2 uv;
    vec3 cube;
    float cube_on;
    vec4 color;
    vec3 material;
    float reflect;
    float view_depth;
    float selected;
} v;

#ifdef WIREFRAME
in vec3 v_bary;
uniform vec3 u_wire_color;
uniform float u_wire_width;   // Pixels
#endif

layout(location = 0) out vec4 o_color;

uniform sampler2D u_texture;
uniform samplerCube u_cube_texture;
uniform float u_exposure;
uniform int u_tonemap;
uniform vec2 u_depth_range;        // near, far
uniform vec3 u_highlight_color;
uniform vec2 u_highlight;          // blend, emission

uniform sampler2D u_reflection0;
uniform sampler2D u_reflection1;
uniform sampler2D u_reflection2;
uniform sampler2D u_reflection3;

// A small positive bias moves grazing, high-frequency surfaces toward the
// next trilinear mip level before their texels become sub-pixel.  This avoids
// the large moire bands produced by repeated floor textures while preserving
// anisotropic detail along the surface.
const float ALBEDO_MINIFICATION_LOD_BIAS = 1.0;
uniform vec2 u_reflection_size;
uniform int u_linear_out;
uniform vec4 u_fog;              // start, end, fog enabled, haze density
uniform vec3 u_fog_color;
uniform vec3 u_haze_color;


void main() {
    vec4 texel = v.cube_on > 0.5
        ? texture(u_cube_texture, v.cube)
        : texture(u_texture, v.uv, ALBEDO_MINIFICATION_LOD_BIAS);
    vec3 surface = v.color.rgb;
    if (u_classic_lighting != 0) {
        surface = gamma_encode(surface);
        texel.rgb = linear_to_srgb(texel.rgb);
    }
    vec4 base = vec4(surface * texel.rgb, v.color.a * texel.a);
    vec3 albedo = base.rgb;
    float alpha = base.a;
    float emission = v.material.x;

    if (v.selected > 0.5) {
        albedo = mix(albedo, u_highlight_color, u_highlight.x);
        emission += u_highlight.y;
    }

#if DEBUG_VIEW == 1
    o_color = vec4(u_classic_lighting != 0 ? albedo : gamma_encode(albedo), alpha);
#elif DEBUG_VIEW == 2
    o_color = vec4(normalize(v.normal) * 0.5 + 0.5, alpha);
#elif DEBUG_VIEW == 3
    o_color = vec4(vec3(1.0 - depth_view(v.view_depth, u_depth_range.x, u_depth_range.y)), alpha);
#elif DEBUG_VIEW == 4
    o_color = vec4(vec3(OVERDRAW_STEP), 0.0);
#else
    vec3 lit = shade(
        albedo, v.normal, v.world,
        emission, v.material.y, v.material.z, v.view_depth,
        texel.rgb
    );
    // Classic OpenGL clamps primary lighting before texture modulation.
    if (u_classic_lighting != 0) {
        lit = clamp(lit, vec3(0.0), texel.rgb);
    }

    if (v.reflect < 0.0 && u_reflection_size.x > 0.0) {
        float code = -v.reflect;
        int layer = int(floor(code * 0.25));
        float reflectance = code - float(layer * 4);
        vec2 reflection_uv = gl_FragCoord.xy / u_reflection_size;
        vec3 reflected;
        if (layer == 0) {
            reflected = texture(u_reflection0, reflection_uv).rgb;
        } else if (layer == 1) {
            reflected = texture(u_reflection1, reflection_uv).rgb;
        } else if (layer == 2) {
            reflected = texture(u_reflection2, reflection_uv).rgb;
        } else {
            reflected = texture(u_reflection3, reflection_uv).rgb;
        }
        lit += reflectance * reflected;
    }

    float fog = u_fog.z * smoothstep(u_fog.x, max(u_fog.y, u_fog.x + 1e-6), v.view_depth);
    float haze = 1.0 - exp(-max(u_fog.w, 0.0) * max(v.view_depth, 0.0));
    vec3 fog_color = u_classic_lighting != 0 ? u_fog_color : srgb_to_linear(u_fog_color);
    vec3 haze_color = u_classic_lighting != 0 ? u_haze_color : srgb_to_linear(u_haze_color);
    lit = mix(lit, fog_color, fog);
    lit = mix(lit, haze_color, haze);

    vec3 rgb = (u_linear_out != 0)
        ? lit
        : (u_classic_lighting != 0
            ? clamp(lit * u_exposure, 0.0, 1.0)
            : finish_color(lit, u_exposure, u_tonemap != 0));
  #ifdef WIREFRAME
    float d = min(v_bary.x, min(v_bary.y, v_bary.z));
    float w = max(fwidth(d), 1e-6) * max(u_wire_width, 0.1);
    rgb = mix(u_wire_color, rgb, smoothstep(0.0, w, d));
  #endif
    o_color = vec4(rgb, alpha);
#endif
}
