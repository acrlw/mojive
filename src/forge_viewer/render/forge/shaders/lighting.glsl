#ifndef FORGE_LIGHTING_GLSL
#define FORGE_LIGHTING_GLSL

#include "common.glsl"

#define FORGE_MAX_LIGHTS 100

uniform int  u_light_count;
layout(std140) uniform ForgeLights {
    vec4 u_light_pos[FORGE_MAX_LIGHTS];       // xyz position, w type
    vec4 u_light_dir[FORGE_MAX_LIGHTS];       // xyz ray direction, w cutoff cosine
    vec4 u_light_diffuse[FORGE_MAX_LIGHTS];   // rgb linear, w spot exponent
    vec4 u_light_specular[FORGE_MAX_LIGHTS];
    vec4 u_light_atten[FORGE_MAX_LIGHTS];     // constant, linear, quadratic, range
};

uniform vec3 u_ambient;
uniform vec4 u_headlight_diffuse;       // rgb diffuse, w enabled
uniform vec3 u_headlight_specular;
uniform vec3 u_camera_pos;
uniform vec3 u_camera_dir;

uniform int u_shadow_light;

float shadow_factor(vec3 world_pos, vec3 normal, float view_depth);
float local_shadow(int kind, int slot, vec3 world_pos, vec3 normal);

vec3 forge_light_term(
    vec3 albedo, vec3 n, vec3 l, vec3 view_dir,
    vec3 diffuse_rgb, vec3 specular_rgb,
    float specular, float shininess, float atten, float shadow
) {
    float ndl = max(dot(n, l), 0.0);
    if (ndl <= 0.0 || atten <= 0.0) return vec3(0.0);
    vec3 h = normalize(l + view_dir);
    float spec = specular * pow(max(dot(n, h), 0.0), max(shininess * 128.0, 1e-3));
    return atten * shadow * ndl * (diffuse_rgb * albedo + specular_rgb * spec);
}

vec3 shade(
    vec3 albedo, vec3 normal, vec3 world_pos,
    float emission, float specular, float shininess, float view_depth
) {
    vec3 n = normalize(normal);
    vec3 view_dir = normalize(u_camera_pos - world_pos);

    vec3 color = ambient_linear(u_ambient) * albedo;

    for (int i = 0; i < FORGE_MAX_LIGHTS; ++i) {
        if (i >= u_light_count) break;
        int kind = int(u_light_pos[i].w + 0.5);
        vec3 l;
        float atten = 1.0;
        if (kind == 0) {
            l = -normalize(u_light_dir[i].xyz);
        } else {
            vec3 to_light = u_light_pos[i].xyz - world_pos;
            float dist = length(to_light);
            l = to_light / max(dist, 1e-6);
            vec3 k = u_light_atten[i].xyz;
            atten = 1.0 / max(k.x + k.y * dist + k.z * dist * dist, 1e-6);
            if (u_light_atten[i].w > 0.0 && dist > u_light_atten[i].w) atten = 0.0;
            if (kind == 2) {
                float cd = dot(-l, normalize(u_light_dir[i].xyz));
                atten *= (cd < u_light_dir[i].w) ? 0.0 : pow(max(cd, 0.0), u_light_diffuse[i].w);
            }
        }
        float shadow = 1.0;
#ifdef USE_SHADOW
        if (i == u_shadow_light) shadow = shadow_factor(world_pos, n, view_depth);
        if (kind != 0) shadow *= local_shadow(kind, u_local_slot[i], world_pos, n);
#endif
        color += forge_light_term(
            albedo, n, l, view_dir,
            u_light_diffuse[i].rgb, u_light_specular[i].rgb,
            specular, shininess, atten, shadow
        );
    }

    if (u_headlight_diffuse.w > 0.5) {
        color += forge_light_term(
            albedo, n, -normalize(u_camera_dir), view_dir,
            u_headlight_diffuse.rgb, u_headlight_specular,
            specular, shininess, 1.0, 1.0
        );
    }

    return color + emission * albedo;
}

#endif
