// World-anchored glyph quads sampled from a single-channel coverage atlas;
// port of forge's debug_text.vert/debug_text.frag.  One instance is one glyph
// record (see render.text): anchor(3) offset_px(2) rect(4) uv_rect(4) rgba(4).
// The anchor projects to clip space; the pixel offset is applied afterwards,
// so text is glued to the world point but laid out in screen pixels.

struct DebugTextIn {
    @location(0) anchor: vec3f,
    @location(1) offset: vec2f,
    @location(2) rect: vec4f,
    @location(3) uv_rect: vec4f,
    @location(4) color: vec4f,
};

struct DebugTextOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
    @location(1) color: vec4f,
};

@group(1) @binding(0) var dbg_atlas: texture_2d<f32>;
@group(1) @binding(1) var dbg_atlas_sampler: sampler;

@vertex
fn vs_debug_text(in: DebugTextIn, @builtin(vertex_index) v: u32) -> DebugTextOut {
    var C = array<vec2f, 6>(
        vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0),
        vec2f(1.0, 0.0), vec2f(1.0, 1.0), vec2f(0.0, 1.0),
    );
    let corner = C[v];
    let pixel = in.offset + mix(in.rect.xy, in.rect.zw, corner);
    var clip = dbg.view_proj * vec4f(in.anchor, 1.0);
    let viewport = dbg_viewport();
    clip = vec4f(
        clip.xy + vec2f(2.0 * pixel.x / viewport.x, -2.0 * pixel.y / viewport.y) * clip.w,
        clip.z,
        clip.w,
    );
    var out: DebugTextOut;
    out.pos = clip;
    out.uv = mix(in.uv_rect.xy, in.uv_rect.zw, corner);
    out.color = vec4f(in.color.rgb, in.color.a * dbg_alpha());
    return out;
}

@fragment
fn fs_debug_text(in: DebugTextOut) -> @location(0) vec4f {
    let coverage = textureSample(dbg_atlas, dbg_atlas_sampler, in.uv).r;
    return vec4f(in.color.rgb, in.color.a * coverage);
}
