@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> packed: array<u32>;

fn rgb8(index: u32, dimensions: vec2u) -> vec3u {
    let pixel_count = dimensions.x * dimensions.y;
    if index >= pixel_count {
        return vec3u(0u);
    }
    let coordinate = vec2u(index % dimensions.x, index / dimensions.x);
    let value = textureLoad(source, vec2i(coordinate), 0).rgb;
    return vec3u(round(clamp(value, vec3f(0.0), vec3f(1.0)) * 255.0));
}

@compute @workgroup_size(64)
fn cs_rgb_pack(@builtin(global_invocation_id) global_id: vec3u) {
    let dimensions = textureDimensions(source);
    // WebGPU guarantees at least 65,535 workgroups per dimension. Linearize
    // a two-dimensional dispatch so targets above roughly 4K remain valid.
    let invocation = global_id.x + global_id.y * 4194240u;
    let base = invocation * 4u;
    if base >= dimensions.x * dimensions.y {
        return;
    }

    let p0 = rgb8(base, dimensions);
    let p1 = rgb8(base + 1u, dimensions);
    let p2 = rgb8(base + 2u, dimensions);
    let p3 = rgb8(base + 3u, dimensions);

    packed[invocation * 3u] =
        p0.x | (p0.y << 8u) | (p0.z << 16u) | (p1.x << 24u);
    packed[invocation * 3u + 1u] =
        p1.y | (p1.z << 8u) | (p2.x << 16u) | (p2.y << 24u);
    packed[invocation * 3u + 2u] =
        p2.z | (p3.x << 8u) | (p3.y << 16u) | (p3.z << 24u);
}
