// Shadow sampling for the webgpu backend.
//
// Full port of opengl's shadow_sample.glsl (concatenated ahead of scene.wgsl
// by programs.load_wgsl; WGSL has no include mechanism and resolves module
// scope order-independently, so this chunk references the `lights` storage
// block declared in scene.wgsl).  Deliberate deltas from the GLSL:
//
// - Shadow matrices are built for WebGPU clip z in [0,1], so the NDC z remap
//   (`* 0.5 + 0.5`) drops out and `depth_per_world` is `length(axis)` without
//   the 0.5 factor (the WebGPU ortho z-scale is 1/(far-near), GL's is twice
//   that).
// - WebGPU framebuffer rows run y-down where GL's run y-up, so every
//   NDC-derived uv flips y (atlas tile, spot layer, point cube face).
// - Directional maps use a comparison sampler so filtering blends visibility
//   rather than stored depth and the GPU supplies each 2x2 PCF sample.
//
// Count fields gate everything: with `shadow_counts` zero the functions never
// touch the (1x1 fallback) textures.

const SHADOW_MAX_CASCADES: i32 = 3;
const SHADOW_PCF_RADIUS: i32 = 1;
const MAX_POINT_PCF_RADIUS: i32 = 3;
const OPENGL_SHADOW_BIAS: vec2f = vec2f(1.0, 2.5);
const OPENGL_SHADOW_MIN_NDL: f32 = 0.15;
// Local distance maps are R16F. One relative half-float ULP keeps a receiver
// on the lit side of its quantized floor distance instead of producing rings.
const LOCAL_DISTANCE_QUANTIZATION_BIAS: f32 = 1.0 / 1024.0;

@group(3) @binding(0) var shadow_atlas: texture_depth_2d;
@group(3) @binding(1) var local_shadow: texture_2d_array<f32>;
@group(3) @binding(2) var shadow_sampler: sampler_comparison;

fn shadow_bias_factors() -> vec2f {
    return select(OPENGL_SHADOW_BIAS, lights.shadow_bias.xy, lights.shadow_bias.x > 0.0);
}

// shadow_sample.glsl shadow_factor(): cascade select with fallback, slope-
// scaled bias, continuous PCF, atlas tile clamp.
fn hardware_shadow_compare(
    uv: vec2f,
    reference: f32,
    min_texel: vec2i,
    max_texel: vec2i,
) -> f32 {
    let dims = vec2f(textureDimensions(shadow_atlas, 0u));
    let lo = (vec2f(min_texel) + vec2f(0.5)) / dims;
    let hi = (vec2f(max_texel) + vec2f(0.5)) / dims;
    return textureSampleCompareLevel(
        shadow_atlas, shadow_sampler, clamp(uv, lo, hi), reference,
    );
}

fn bilinear_local_shadow_compare(uv: vec2f, layer: i32, reference: f32) -> f32 {
    let dims_i = vec2i(textureDimensions(local_shadow, 0u).xy);
    let texel_pos = uv * vec2f(dims_i) - vec2f(0.5);
    let base = vec2i(floor(texel_pos));
    let blend = fract(texel_pos);
    let lo = vec2i(0);
    let hi = dims_i - vec2i(1);
    let p00 = clamp(base, lo, hi);
    let p10 = clamp(base + vec2i(1, 0), lo, hi);
    let p01 = clamp(base + vec2i(0, 1), lo, hi);
    let p11 = clamp(base + vec2i(1, 1), lo, hi);
    let row0 = mix(
        step(reference, textureLoad(local_shadow, p00, layer, 0).r),
        step(reference, textureLoad(local_shadow, p10, layer, 0).r),
        blend.x,
    );
    let row1 = mix(
        step(reference, textureLoad(local_shadow, p01, layer, 0).r),
        step(reference, textureLoad(local_shadow, p11, layer, 0).r),
        blend.x,
    );
    return mix(row0, row1, blend.y);
}

fn filtered_shadow_compare(
    uv: vec2f,
    texel: vec2f,
    reference: f32,
    min_texel: vec2i,
    max_texel: vec2i,
) -> f32 {
    let quality = clamp(i32(lights.shadow_counts.w), 0, 2);
    if quality == 0 {
        return hardware_shadow_compare(uv, reference, min_texel, max_texel);
    }
    let offset = texel * 0.75;
    var lit = 0.0;
    if quality == 1 {
        for (var y = -1; y <= 1; y = y + 2) {
            for (var x = -1; x <= 1; x = x + 2) {
                lit += hardware_shadow_compare(
                    uv + vec2f(f32(x), f32(y)) * offset,
                    reference,
                    min_texel,
                    max_texel,
                );
            }
        }
        return lit * 0.25;
    }
    // Each comparison sample already provides a filtered 2x2 footprint. A
    // center plus four diagonals retains a wider high-quality kernel without
    // the redundant work of a full 3x3 comparison grid.
    lit = hardware_shadow_compare(uv, reference, min_texel, max_texel);
    let high_offset = texel * 1.25;
    for (var y = -1; y <= 1; y = y + 2) {
        for (var x = -1; x <= 1; x = x + 2) {
            lit += hardware_shadow_compare(
                uv + vec2f(f32(x), f32(y)) * high_offset,
                reference,
                min_texel,
                max_texel,
            );
        }
    }
    return lit * 0.2;
}

fn filtered_local_shadow_compare(uv: vec2f, layer: i32, reference: f32) -> f32 {
    let quality = clamp(i32(lights.shadow_counts.w), 0, 2);
    if quality == 0 {
        return bilinear_local_shadow_compare(uv, layer, reference);
    }
    let texel = 1.0 / vec2f(textureDimensions(local_shadow, 0u).xy);
    let offset = texel * 0.75;
    var lit = 0.0;
    if quality == 1 {
        for (var y = -1; y <= 1; y = y + 2) {
            for (var x = -1; x <= 1; x = x + 2) {
                lit += bilinear_local_shadow_compare(
                    uv + vec2f(f32(x), f32(y)) * offset, layer, reference
                );
            }
        }
        return lit * 0.25;
    }
    lit = bilinear_local_shadow_compare(uv, layer, reference);
    let high_offset = texel * 1.25;
    for (var y = -1; y <= 1; y = y + 2) {
        for (var x = -1; x <= 1; x = x + 2) {
            lit += bilinear_local_shadow_compare(
                uv + vec2f(f32(x), f32(y)) * high_offset, layer, reference
            );
        }
    }
    return lit * 0.2;
}

fn shadow_factor(world_pos: vec3f, normal: vec3f, view_depth: f32) -> f32 {
    let count = min(i32(lights.shadow_counts.x), SHADOW_MAX_CASCADES);
    if count <= 0 {
        return 1.0;
    }
    var c = count - 1;
    for (var i = 0; i < SHADOW_MAX_CASCADES; i = i + 1) {
        if i >= count {
            break;
        }
        if view_depth < lights.shadow_splits[i] {
            c = i;
            break;
        }
    }

    var p = vec3f(0.0);
    var cascade = -1;
    for (var k = c; k < SHADOW_MAX_CASCADES; k = k + 1) {
        if k >= count {
            break;
        }
        let clip = lights.shadow_matrix[k] * vec4f(world_pos, 1.0);
        let ndc = clip.xyz / clip.w;
        let q = vec3f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5, ndc.z);
        if all(q >= vec3f(0.0)) && all(q <= vec3f(1.0)) {
            p = q;
            cascade = k;
            break;
        }
    }
    if cascade < 0 {
        return 1.0;
    }

    let axis = vec3f(
        lights.shadow_matrix[cascade][0][2],
        lights.shadow_matrix[cascade][1][2],
        lights.shadow_matrix[cascade][2][2],
    );
    let depth_per_world = length(axis);  // = 1 / (far - near)
    let n = normalize(normal);
    let ndl = dot(n, -normalize(axis));
    if ndl <= 0.0 {
        return 0.0;
    }

    let k2 = shadow_bias_factors();
    let tan_theta = sqrt(max(1.0 - ndl * ndl, 0.0)) / max(ndl, OPENGL_SHADOW_MIN_NDL);
    let bias = lights.shadow_texel[cascade] * (k2.x + k2.y * tan_theta) * depth_per_world;

    let dims = vec2f(textureDimensions(shadow_atlas, 0u));
    let texel_uv = 1.0 / dims;
    let tile = lights.shadow_tile[cascade];
    let margin = (f32(SHADOW_PCF_RADIUS) + 0.5) * texel_uv;
    let uv = mix(tile.xy - margin, tile.zw + margin, p.xy);

    let ref_depth = p.z - bias;
    let dims_i = vec2i(textureDimensions(shadow_atlas, 0u));
    let min_texel = max(vec2i(floor((tile.xy - margin) * dims)), vec2i(0));
    let max_texel = min(
        vec2i(ceil((tile.zw + margin) * dims)) - vec2i(1), dims_i - vec2i(1)
    );
    return filtered_shadow_compare(uv, texel_uv, ref_depth, min_texel, max_texel);
}

fn local_bias(slot: i32, dist: f32, normal: vec3f, l: vec3f) -> f32 {
    let ndl = max(dot(normalize(normal), l), OPENGL_SHADOW_MIN_NDL);
    let tan_theta = sqrt(max(1.0 - ndl * ndl, 0.0)) / ndl;
    let k = shadow_bias_factors();
    return dist * (
        lights.local_texel[slot] * (k.x + k.y * tan_theta)
        + LOCAL_DISTANCE_QUANTIZATION_BIAS
    );
}

// shadow_sample.glsl local_spot_shadow(): perspective-projected distance map.
fn local_spot_shadow(slot: i32, world_pos: vec3f, normal: vec3f) -> f32 {
    let clip = lights.local_matrix[slot] * vec4f(world_pos, 1.0);
    if clip.w <= 0.0 {
        return 1.0;
    }
    let ndc = clip.xyz / clip.w;
    let p = vec3f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5, ndc.z);
    if any(p < vec3f(0.0)) || any(p > vec3f(1.0)) {
        return 1.0;
    }
    let to_light = lights.local_pos[slot].xyz - world_pos;
    let dist = length(to_light);
    if lights.local_pos[slot].w > 0.0 && dist > lights.local_pos[slot].w {
        return 1.0;
    }
    let bias = local_bias(slot, dist, normal, to_light / max(dist, 1e-6));
    return filtered_local_shadow_compare(p.xy, lights.local_layer[slot], dist - bias);
}

// shadow_sample.glsl point_layer_uv(): direction to cube-face layer uv; the
// GL face formulas stand, the v axis flips for WebGPU's y-down rows.
fn point_layer_uv(slot: i32, d: vec3f) -> vec3f {
    let a = abs(d);
    var face: f32;
    var sc: vec2f;
    var ma: f32;
    if a.x >= a.y && a.x >= a.z {
        ma = a.x;
        if d.x > 0.0 {
            face = 0.0;
            sc = vec2f(-d.z, -d.y);
        } else {
            face = 1.0;
            sc = vec2f(d.z, -d.y);
        }
    } else if a.y >= a.z {
        ma = a.y;
        if d.y > 0.0 {
            face = 2.0;
            sc = vec2f(d.x, d.z);
        } else {
            face = 3.0;
            sc = vec2f(d.x, -d.z);
        }
    } else {
        ma = a.z;
        if d.z > 0.0 {
            face = 4.0;
            sc = vec2f(d.x, -d.y);
        } else {
            face = 5.0;
            sc = vec2f(-d.x, -d.y);
        }
    }
    return vec3f(
        sc.x / max(ma, 1e-6) * 0.5 + 0.5,
        0.5 - sc.y / max(ma, 1e-6) * 0.5,
        f32(lights.local_layer[slot]) + face,
    );
}

// shadow_sample.glsl local_point_shadow(): cube-face distance map; area
// lights (local_radius > 0) widen the kernel to 7x7.
fn local_point_shadow(slot: i32, world_pos: vec3f, normal: vec3f) -> f32 {
    let ray = world_pos - lights.local_pos[slot].xyz;
    let dist = length(ray);
    if lights.local_pos[slot].w > 0.0 && dist > lights.local_pos[slot].w {
        return 1.0;
    }
    let direction = ray / max(dist, 1e-6);
    let bias = local_bias(slot, dist, normal, -direction);

    let seed = select(vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(direction.z) < 0.9);
    let right = normalize(cross(direction, seed));
    let up = cross(right, direction);
    var lit = 0.0;
    var taps = 0.0;
    let quality = clamp(i32(lights.shadow_counts.w), 0, 2);
    let area = lights.local_radius[slot] > 0.0;
    var radius = select(0, 1, area);
    if quality == 1 {
        radius = select(1, 2, area);
    } else if quality >= 2 {
        radius = select(2, 3, area);
    }
    let step_angle = max(
        lights.local_texel[slot],
        lights.local_radius[slot] / max(dist * f32(max(radius, 1)), 1e-6),
    );
    let dims = vec2f(textureDimensions(local_shadow, 0u).xy);
    let dims_i = vec2i(dims);
    for (var y = -MAX_POINT_PCF_RADIUS; y <= MAX_POINT_PCF_RADIUS; y = y + 1) {
        for (var x = -MAX_POINT_PCF_RADIUS; x <= MAX_POINT_PCF_RADIUS; x = x + 1) {
            if abs(x) > radius || abs(y) > radius {
                continue;
            }
            let sample_dir = direction + (right * f32(x) + up * f32(y)) * step_angle;
            let luv = point_layer_uv(slot, sample_dir);
            let texel = clamp(vec2i(luv.xy * dims), vec2i(0), dims_i - vec2i(1));
            lit += step(dist - bias, textureLoad(local_shadow, texel, i32(luv.z), 0).r);
            taps += 1.0;
        }
    }
    return lit / taps;
}

fn local_shadow_factor(light_type: i32, slot: i32, world_pos: vec3f, normal: vec3f) -> f32 {
    if slot < 0 || slot >= i32(lights.shadow_counts.y) {
        return 1.0;
    }
    if light_type == 2 {
        return local_spot_shadow(slot, world_pos, normal);
    }
    return local_point_shadow(slot, world_pos, normal);
}
