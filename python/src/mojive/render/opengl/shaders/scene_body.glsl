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

uniform vec2 u_reflection_size;
uniform int u_linear_out;
uniform vec4 u_fog;              // start, end, fog enabled, haze density
uniform vec3 u_fog_color;
uniform vec3 u_haze_color;


// Use the same bounded anisotropic footprint in each graphics API. Hardware
// anisotropy may choose different LODs even with identical mips and sampler limits.
vec4 sampleAlbedo(vec2 uv) {
    vec2 dimensions = vec2(textureSize(u_texture, 0));
    vec2 dx = dFdx(uv) * dimensions;
    vec2 dy = dFdy(uv) * dimensions;
    // Principal axes of the pixel footprint keep the filter stable under rotation.
    float a = dx.x * dx.x + dy.x * dy.x;
    float b = dx.x * dx.y + dy.x * dy.y;
    float c = dx.y * dx.y + dy.y * dy.y;
    float discriminant = sqrt(max((a-c)*(a-c) + 4.0*b*b, 0.0));
    float majorSquared = max(0.5 * (a+c+discriminant), 1e-12);
    float majorLength = sqrt(majorSquared);
    float minorLength = max(sqrt(max(0.5 * (a+c-discriminant), 0.0)), majorLength / 16.0);
    vec2 axis = a >= c ? vec2(majorSquared-c, b) : vec2(b, majorSquared-a);
    axis *= inversesqrt(max(dot(axis, axis), 1e-12));
    vec2 span = axis * majorLength / dimensions;
    float taps = ceil(clamp(majorLength / max(minorLength, 1.0), 1.0, 16.0));
    float lod = max(log2(max(minorLength, 1e-8)) + 1.0, 0.0);
    vec4 color = vec4(0.0);
    for (int i = 0; i < 16; ++i) {
        if (float(i) >= taps) break;
        color += textureLod(u_texture, uv + span * ((float(i)+0.5)/taps-0.5), lod);
    }
    return color / taps;
}

void main() {
    vec4 texel = v.cube_on > 0.5
        ? texture(u_cube_texture, v.cube)
        : sampleAlbedo(v.uv);
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
