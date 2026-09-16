@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;

@vertex
fn vs_mipmap(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
    let p = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_mipmap(@builtin(position) pixel: vec4f) -> @location(0) vec4f {
    // Exact 2x box footprint; a unit axis remains clamped to its single texel.
    // sRGB views decode before interpolation and encode the render target.
    let extent = max(textureDimensions(source) / vec2u(2u), vec2u(1u));
    return textureSampleLevel(source, source_sampler, pixel.xy / vec2f(extent), 0.0);
}
