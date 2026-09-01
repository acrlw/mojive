#ifndef MOJIVE_SHADOW_SAMPLE_GLSL
#define MOJIVE_SHADOW_SAMPLE_GLSL

#define SHADOW_MAX_CASCADES 3
#define SHADOW_MAX_LOCAL 8
#define SHADOW_MAX_LIGHTS 100
#define LOCAL_PCF_RADIUS 1
#define AREA_PCF_RADIUS 3

#ifndef SHADOW_PCF_RADIUS
#define SHADOW_PCF_RADIUS 1
#endif

const vec2 MOJIVE_SHADOW_BIAS = vec2(1.0, 2.5);
const float MOJIVE_SHADOW_MIN_NDL = 0.15;
// Local distance maps are R16F.  One relative half-float ULP prevents a
// receiver from alternately falling above and below its quantized floor
// distance, which otherwise appears as concentric self-shadow rings.
const float LOCAL_DISTANCE_QUANTIZATION_BIAS = 1.0 / 1024.0;

float pcf_tent_weight(int offset, int radius) {
    return float(radius + 1 - abs(offset));
}

uniform sampler2D u_shadow_atlas;
uniform mat4 u_shadow_matrix[SHADOW_MAX_CASCADES];
uniform vec3 u_shadow_splits;
uniform vec3 u_shadow_texel;
uniform vec4 u_shadow_tile[SHADOW_MAX_CASCADES];
uniform int u_shadow_count;
uniform vec2 u_shadow_bias;

uniform sampler2DArray u_local_shadow;
uniform mat4 u_local_matrix[SHADOW_MAX_LOCAL];
uniform vec4 u_local_pos[SHADOW_MAX_LOCAL];       // xyz position, w range
uniform float u_local_texel[SHADOW_MAX_LOCAL];
uniform float u_local_radius[SHADOW_MAX_LOCAL];
uniform int u_local_layer[SHADOW_MAX_LOCAL];
uniform int u_local_slot[SHADOW_MAX_LIGHTS];
uniform int u_local_count;

float shadow_factor(vec3 world_pos, vec3 normal, float view_depth) {
    int count = min(u_shadow_count, SHADOW_MAX_CASCADES);
    if (count <= 0) return 1.0;

    int c = count - 1;
    for (int i = 0; i < SHADOW_MAX_CASCADES; ++i) {
        if (i >= count) break;
        if (view_depth < u_shadow_splits[i]) { c = i; break; }
    }

    vec3 p = vec3(0.0);
    bool inside = false;
    for (; c < SHADOW_MAX_CASCADES; ++c) {
        if (c >= count) break;
        vec4 clip = u_shadow_matrix[c] * vec4(world_pos, 1.0);
        p = (clip.xyz / clip.w) * 0.5 + 0.5;
        if (all(greaterThanEqual(p, vec3(0.0))) && all(lessThanEqual(p, vec3(1.0)))) {
            inside = true;
            break;
        }
    }
    if (!inside) return 1.0;

    vec3 axis = vec3(u_shadow_matrix[c][0][2], u_shadow_matrix[c][1][2], u_shadow_matrix[c][2][2]);
    float depth_per_world = 0.5 * length(axis);   // = 1 / (far − near)
    vec3 n = normalize(normal);
    float ndl = dot(n, -normalize(axis));
    if (ndl <= 0.0) return 0.0;

    vec2 k = (u_shadow_bias.x > 0.0) ? u_shadow_bias : MOJIVE_SHADOW_BIAS;
    float tan_theta = sqrt(max(1.0 - ndl * ndl, 0.0)) / max(ndl, MOJIVE_SHADOW_MIN_NDL);
    float bias = u_shadow_texel[c] * (k.x + k.y * tan_theta) * depth_per_world;

    vec2 texel_uv = 1.0 / vec2(textureSize(u_shadow_atlas, 0));
    vec4 tile = u_shadow_tile[c];
    vec2 margin = (float(SHADOW_PCF_RADIUS) + 0.5) * texel_uv;
    vec2 uv = mix(tile.xy - margin, tile.zw + margin, p.xy);

    float lit = 0.0;
    float taps = 0.0;
    float ref = p.z - bias;
    for (int y = -SHADOW_PCF_RADIUS; y <= SHADOW_PCF_RADIUS; ++y) {
        for (int x = -SHADOW_PCF_RADIUS; x <= SHADOW_PCF_RADIUS; ++x) {
            vec2 s = clamp(uv + vec2(float(x), float(y)) * texel_uv, tile.xy, tile.zw);
            float weight = pcf_tent_weight(x, SHADOW_PCF_RADIUS) *
                           pcf_tent_weight(y, SHADOW_PCF_RADIUS);
            lit += step(ref, texture(u_shadow_atlas, s).r) * weight;
            taps += weight;
        }
    }
    return lit / taps;
}

float local_bias(int slot, float dist, vec3 normal, vec3 l) {
    float ndl = max(dot(normalize(normal), l), MOJIVE_SHADOW_MIN_NDL);
    float tan_theta = sqrt(max(1.0 - ndl * ndl, 0.0)) / ndl;
    vec2 k = (u_shadow_bias.x > 0.0) ? u_shadow_bias : MOJIVE_SHADOW_BIAS;
    return dist * (
        u_local_texel[slot] * (k.x + k.y * tan_theta)
        + LOCAL_DISTANCE_QUANTIZATION_BIAS
    );
}

float local_spot_shadow(int slot, vec3 world_pos, vec3 normal) {
    vec4 clip = u_local_matrix[slot] * vec4(world_pos, 1.0);
    if (clip.w <= 0.0) return 1.0;
    vec3 p = (clip.xyz / clip.w) * 0.5 + 0.5;
    if (any(lessThan(p.xy, vec2(0.0))) || any(greaterThan(p.xy, vec2(1.0)))) return 1.0;
    if (p.z < 0.0 || p.z > 1.0) return 1.0;
    vec3 to_light = u_local_pos[slot].xyz - world_pos;
    float dist = length(to_light);
    if (u_local_pos[slot].w > 0.0 && dist > u_local_pos[slot].w) return 1.0;
    float bias = local_bias(slot, dist, normal, to_light / max(dist, 1e-6));
    vec2 texel = 1.0 / vec2(textureSize(u_local_shadow, 0).xy);
    vec2 margin = (float(LOCAL_PCF_RADIUS) + 0.5) * texel;
    vec2 uv = mix(margin, vec2(1.0) - margin, p.xy);
    float lit = 0.0;
    float taps = 0.0;
    for (int y = -LOCAL_PCF_RADIUS; y <= LOCAL_PCF_RADIUS; ++y) {
        for (int x = -LOCAL_PCF_RADIUS; x <= LOCAL_PCF_RADIUS; ++x) {
            vec2 sample_uv = clamp(uv + vec2(float(x), float(y)) * texel,
                                   margin, vec2(1.0) - margin);
            float weight = pcf_tent_weight(x, LOCAL_PCF_RADIUS) *
                           pcf_tent_weight(y, LOCAL_PCF_RADIUS);
            lit += step(dist - bias,
                        texture(u_local_shadow, vec3(sample_uv, float(u_local_layer[slot]))).r) * weight;
            taps += weight;
        }
    }
    return lit / taps;
}

vec3 point_layer_uv(int slot, vec3 d) {
    vec3 a = abs(d);
    float face;
    vec2 sc;
    float ma;
    if (a.x >= a.y && a.x >= a.z) {
        ma = a.x;
        if (d.x > 0.0) { face = 0.0; sc = vec2(-d.z, -d.y); }
        else           { face = 1.0; sc = vec2( d.z, -d.y); }
    } else if (a.y >= a.z) {
        ma = a.y;
        if (d.y > 0.0) { face = 2.0; sc = vec2( d.x,  d.z); }
        else           { face = 3.0; sc = vec2( d.x, -d.z); }
    } else {
        ma = a.z;
        if (d.z > 0.0) { face = 4.0; sc = vec2( d.x, -d.y); }
        else           { face = 5.0; sc = vec2(-d.x, -d.y); }
    }
    return vec3(sc / max(ma, 1e-6) * 0.5 + 0.5, float(u_local_layer[slot]) + face);
}

float local_point_shadow(int slot, vec3 world_pos, vec3 normal) {
    vec3 ray = world_pos - u_local_pos[slot].xyz;
    float dist = length(ray);
    if (u_local_pos[slot].w > 0.0 && dist > u_local_pos[slot].w) return 1.0;
    vec3 direction = ray / max(dist, 1e-6);
    float bias = local_bias(slot, dist, normal, -direction);

    vec3 seed = (abs(direction.z) < 0.9) ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
    vec3 right = normalize(cross(direction, seed));
    vec3 up = cross(right, direction);
    float lit = 0.0;
    float taps = 0.0;
    int radius = (u_local_radius[slot] > 0.0) ? AREA_PCF_RADIUS : LOCAL_PCF_RADIUS;
    float step_angle = max(
        u_local_texel[slot],
        u_local_radius[slot] / max(dist * float(AREA_PCF_RADIUS), 1e-6)
    );
    for (int y = -AREA_PCF_RADIUS; y <= AREA_PCF_RADIUS; ++y) {
        for (int x = -AREA_PCF_RADIUS; x <= AREA_PCF_RADIUS; ++x) {
            if (abs(x) > radius || abs(y) > radius) continue;
            vec3 sample_dir = direction +
                (right * float(x) + up * float(y)) * step_angle;
            lit += step(dist - bias,
                        texture(u_local_shadow, point_layer_uv(slot, sample_dir)).r);
            taps += 1.0;
        }
    }
    return lit / taps;
}

float local_shadow(int light_type, int slot, vec3 world_pos, vec3 normal) {
    if (slot < 0 || slot >= u_local_count) return 1.0;
    return (light_type == 2)
        ? local_spot_shadow(slot, world_pos, normal)
        : local_point_shadow(slot, world_pos, normal);
}

#endif
