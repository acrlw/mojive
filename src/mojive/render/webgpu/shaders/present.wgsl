// Present pass for the webgpu backend: rebuilds the SEGMENT/IDCOLOR debug
// views from the export id texture.  Direct port of opengl's present.frag
// pseudocolor path with two deliberate deltas:
//
// - opengl reads the multisampled id texture per sample (texelFetch); this
//   backend re-rasterizes ids into the single-sampled export MRT instead
//   (WebGPU cannot resolve integer MSAA), so the id fetch is a textureLoad.
// - Mode 0 (plain color) needs no shader here: the main pass already resolves
//   MSAA into the color target, so the pass only runs for SEGMENT/IDCOLOR.

struct PresentUniforms {
    params: vec4u,  // x mode (1 segment, 2 object id), y selected id
};

@group(0) @binding(0) var present_ids: texture_2d<u32>;
@group(0) @binding(1) var<uniform> present: PresentUniforms;

fn id_color(id: u32) -> vec3f {
    if (id == 0u) {
        return vec3f(0.0);
    }
    let h = id * 2654435761u;
    return vec3f(
        f32((h >> 16u) & 255u),
        f32((h >> 8u) & 255u),
        f32(h & 255u),
    ) / 255.0;
}

@vertex
fn vs_present(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4f {
    let p = vec2f(f32((vertex_index << 1u) & 2u), f32(vertex_index & 2u));
    return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_present(@builtin(position) frag_coord: vec4f) -> @location(0) vec4f {
    let id = textureLoad(present_ids, vec2i(frag_coord.xy), 0).r;
    if (present.params.x == 1u) {
        // SEGMENT: the selected object renders white, everything else hashed.
        let selected = present.params.y;
        return vec4f(select(id_color(id), vec3f(1.0), id == selected && id != 0u), 1.0);
    }
    return vec4f(id_color(id), 1.0);
}
