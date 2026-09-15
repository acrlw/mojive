struct DebugLitTriangleIn {
    @location(0) a: vec3f, @location(1) b: vec3f, @location(2) c: vec3f,
    @location(3) na: vec3f, @location(4) nb: vec3f, @location(5) nc: vec3f,
    @location(6) color: vec4f,
};
fn lit_triangle(in: DebugLitTriangleIn, index: u32) -> DebugSolidOut {
    let p = select(select(in.a, in.b, index == 1u), in.c, index == 2u);
    let n = select(select(in.na, in.nb, index == 1u), in.nc, index == 2u);
    var out: DebugSolidOut;
    out.pos = dbg.view_proj * vec4f(p, 1.0);
    out.normal = (dbg.view * vec4f(n, 0.0)).xyz;
    out.view_pos = (dbg.view * vec4f(p, 1.0)).xyz;
    out.color = in.color;
    return out;
}
@vertex
fn vs_debug_lit_triangle(in: DebugLitTriangleIn, @builtin(vertex_index) index: u32) -> DebugSolidOut {
    return lit_triangle(in, index);
}
@vertex
fn vs_debug_lit_foreground(in: DebugLitTriangleIn, @builtin(vertex_index) index: u32) -> DebugSolidOut {
    var out = lit_triangle(in, index);
    out.pos.z = 0.01 * out.pos.z;
    return out;
}
