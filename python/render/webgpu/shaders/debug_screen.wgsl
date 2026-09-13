struct DebugScreenIn {
    @location(0) a: vec3f,
    @location(1) b: vec3f,
    @location(2) c: vec3f,
    @location(3) color: vec4f,
};

@vertex
fn vs_debug_screen(in: DebugScreenIn, @builtin(vertex_index) index: u32) -> DebugLineOut {
    let p = select(select(in.c, in.b, index == 1u), in.a, index == 0u);
    let ndc = p.xy / dbg_viewport() * 2.0 - vec2f(1.0);
    var out: DebugLineOut;
    out.pos = vec4f(ndc.x, -ndc.y, 0.0, 1.0);
    out.color = vec4f(in.color.rgb, in.color.a * p.z * dbg_alpha());
    return out;
}

@vertex
fn vs_debug_triangle(in: DebugScreenIn, @builtin(vertex_index) index: u32) -> DebugLineOut {
    let p = select(select(in.c, in.b, index == 1u), in.a, index == 0u);
    var out: DebugLineOut;
    out.pos = dbg.view_proj * vec4f(p, 1.0);
    out.color = vec4f(in.color.rgb, in.color.a * dbg_alpha());
    return out;
}
