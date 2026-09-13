// Instanced built-in meshes (box/sphere/arrow/...) with cheap facing shading;
// port of opengl's debug_solid.vert/debug_solid.frag.  Slot 0 is the mesh
// (position, normal — the first two scene vertex attributes), slot 1 the
// per-instance record: model matrix columns 0..3 plus rgba.

struct DebugSolidVertex {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
};

struct DebugSolidInstance {
    @location(2) model0: vec4f,
    @location(3) model1: vec4f,
    @location(4) model2: vec4f,
    @location(5) model3: vec4f,
    @location(6) color: vec4f,
};

struct DebugSolidOut {
    @builtin(position) pos: vec4f,
    @location(0) color: vec4f,
    @location(1) normal: vec3f,
    @location(2) view_pos: vec3f,
};

@vertex
fn vs_debug_solid(v: DebugSolidVertex, in: DebugSolidInstance) -> DebugSolidOut {
    let model = mat4x4f(in.model0, in.model1, in.model2, in.model3);
    let world = model * vec4f(v.position, 1.0);
    var out: DebugSolidOut;
    out.color = vec4f(in.color.rgb, in.color.a * dbg_alpha());
    out.normal = mat3x3f(model[0].xyz, model[1].xyz, model[2].xyz) * v.normal;
    out.view_pos = (dbg.view * world).xyz;
    out.pos = dbg.view_proj * world;
    return out;
}

@fragment
fn fs_debug_solid(in: DebugSolidOut) -> @location(0) vec4f {
    let n = normalize(in.normal);
    let to_eye = normalize(-in.view_pos);
    let facing = abs(dot(n, to_eye));
    return vec4f(in.color.rgb * (0.55 + 0.45 * facing), in.color.a);
}
