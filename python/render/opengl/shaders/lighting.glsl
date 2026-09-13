#ifndef MOJIVE_LIGHTING_GLSL
#define MOJIVE_LIGHTING_GLSL

#include "common.glsl"

#define MOJIVE_MAX_LIGHTS 100

uniform int  u_light_count;
layout(std140) uniform OpenGLLights {
    vec4 u_light_pos[MOJIVE_MAX_LIGHTS];       // xyz position, w type
    vec4 u_light_dir[MOJIVE_MAX_LIGHTS];       // xyz ray direction, w cutoff cosine
    vec4 u_light_diffuse[MOJIVE_MAX_LIGHTS];   // rgb linear, w spot exponent
    vec4 u_light_specular[MOJIVE_MAX_LIGHTS];
    vec4 u_light_atten[MOJIVE_MAX_LIGHTS];     // constant, linear, quadratic, range
};

uniform vec3 u_ambient;
uniform vec4 u_headlight_diffuse;       // rgb diffuse, w enabled
uniform vec3 u_headlight_specular;
uniform vec3 u_camera_pos;
uniform vec3 u_camera_dir;
uniform samplerCube u_image_light_texture;
uniform vec2 u_image_light;  // intensity gain, maximum mip level
uniform int u_classic_lighting;

uniform int u_shadow_light;

float shadow_factor(vec3 world_pos, vec3 normal, float view_depth);
float local_shadow(int light_type, int slot, vec3 world_pos, vec3 normal);

vec3 mojive_lighting_color(vec3 c) {
    return u_classic_lighting != 0 ? linear_to_srgb(c) : c;
}

vec3 mojive_light_term(
    vec3 albedo, vec3 n, vec3 l, vec3 view_dir,
    vec3 diffuse_rgb, vec3 specular_rgb,
    vec3 specular_mod,
    float specular, float shininess, float atten, float shadow
) {
    float ndl = max(dot(n, l), 0.0);
    if (ndl <= 0.0 || atten <= 0.0) return vec3(0.0);
    vec3 h = normalize(l + view_dir);
    float spec = specular * pow(max(dot(n, h), 0.0), max(shininess * 128.0, 1e-3));
    diffuse_rgb = mojive_lighting_color(diffuse_rgb);
    specular_rgb = mojive_lighting_color(specular_rgb);
    if (u_classic_lighting != 0) {
        return atten * shadow
            * (ndl * diffuse_rgb * albedo + specular_rgb * spec * specular_mod);
    }
    return atten * shadow * ndl
        * (diffuse_rgb * albedo + specular_rgb * spec * specular_mod);
}

vec3 shade(
    vec3 albedo, vec3 normal, vec3 world_pos,
    float emission, float specular, float shininess, float view_depth,
    vec3 texture_color
) {
    vec3 n = normalize(normal);
    vec3 view_dir = normalize(u_camera_pos - world_pos);

    vec3 ambient = u_classic_lighting != 0 ? clamp(u_ambient, 0.0, 1.0) : ambient_linear(u_ambient);
    vec3 color = ambient * albedo;
    vec3 specular_mod = u_classic_lighting != 0 ? texture_color : vec3(1.0);

    if (u_image_light.x > 0.0) {
        vec3 cube_n = vec3(n.x, n.z, -n.y);
        vec3 reflected = reflect(-view_dir, n);
        vec3 cube_r = vec3(reflected.x, reflected.z, -reflected.y);
        vec3 diffuse_ibl = textureLod(u_image_light_texture, cube_n, u_image_light.y).rgb;
        float roughness = 1.0 - clamp(shininess, 0.0, 1.0);
        vec3 specular_ibl = textureLod(
            u_image_light_texture, cube_r, roughness * u_image_light.y
        ).rgb;
        diffuse_ibl = mojive_lighting_color(diffuse_ibl);
        specular_ibl = mojive_lighting_color(specular_ibl);
        color += u_image_light.x
            * (diffuse_ibl * albedo + specular * specular_ibl * specular_mod);
    }

    for (int i = 0; i < MOJIVE_MAX_LIGHTS; ++i) {
        if (i >= u_light_count) break;
        int light_type = int(u_light_pos[i].w + 0.5);
        vec3 l;
        float atten = 1.0;
        if (light_type == 0) {
            l = -normalize(u_light_dir[i].xyz);
        } else {
            vec3 to_light = u_light_pos[i].xyz - world_pos;
            float dist = length(to_light);
            l = to_light / max(dist, 1e-6);
            vec3 k = u_light_atten[i].xyz;
            atten = 1.0 / max(k.x + k.y * dist + k.z * dist * dist, 1e-6);
            if (
                u_classic_lighting == 0
                && u_light_atten[i].w > 0.0
                && dist > u_light_atten[i].w
            ) atten = 0.0;
            if (light_type == 2) {
                float cd = dot(-l, normalize(u_light_dir[i].xyz));
                atten *= (cd < u_light_dir[i].w)
                    ? 0.0
                    : pow(max(cd, 0.0), u_light_diffuse[i].w);
            }
        }
        float shadow = 1.0;
#ifdef USE_SHADOW
        if (i == u_shadow_light) shadow = shadow_factor(world_pos, n, view_depth);
        if (light_type != 0) {
            shadow *= local_shadow(light_type, u_local_slot[i], world_pos, n);
        }
#endif
        color += mojive_light_term(
            albedo, n, l, view_dir,
            u_light_diffuse[i].rgb, u_light_specular[i].rgb,
            specular_mod,
            specular, shininess, atten, shadow
        );
    }

    if (u_headlight_diffuse.w > 0.5) {
        color += mojive_light_term(
            albedo, n, -normalize(u_camera_dir), view_dir,
            u_headlight_diffuse.rgb, u_headlight_specular,
            specular_mod,
            specular, shininess, 1.0, 1.0
        );
    }

    return color + emission * albedo;
}

#endif
