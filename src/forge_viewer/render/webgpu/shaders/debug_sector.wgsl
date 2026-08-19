// Rodrigues-rotation sector fan (32 triangles per instance); port of forge's
// debug_sector.vert/debug_sector.frag.  One record: center(3) rotvec_end(3)
// ref_end(3) rgba(4) radius_px(1); radius_px zero uses the reference length.

const SECTOR_SEGMENTS: i32 = 32;

struct DebugSectorIn {
    @location(0) center: vec3f,
    @location(1) rot_end: vec3f,  // center plus rotation vector
    @location(2) ref_end: vec3f,  // center plus reference direction
    @location(3) color: vec4f,
    @location(4) radius: f32,     // pixel radius; zero uses the reference length
};

@vertex
fn vs_debug_sector(in: DebugSectorIn, @builtin(vertex_index) v: u32) -> DebugLineOut {
    var out: DebugLineOut;
    out.color = vec4f(in.color.rgb, in.color.a * dbg_alpha());

    let c = in.center;
    let rotvec = in.rot_end - c;
    let refv = in.ref_end - c;
    let angle = length(rotvec);
    let ref_len = length(refv);
    if ref_len < 1e-9 {
        out.pos = vec4f(0.0, 0.0, 2.0, 1.0);
        return out;
    }
    let axis = select(vec3f(0.0, 0.0, 1.0), rotvec / angle, angle > 1e-9);
    let dir = refv / ref_len;

    var radius = ref_len;
    if in.radius > 0.0 {
        let w = (dbg.proj * (dbg.view * vec4f(c, 1.0))).w;
        radius = in.radius * dbg_px_scale() * w;
    }

    let tri = i32(v) / 3;
    let corner = i32(v) % 3;
    var p = c;
    if corner > 0 {
        let t = angle * f32(tri + corner - 1) / f32(SECTOR_SEGMENTS);
        let rv = dir * cos(t) + cross(axis, dir) * sin(t) + axis * dot(axis, dir) * (1.0 - cos(t));
        p = c + rv * radius;
    }
    out.pos = dbg.view_proj * vec4f(p, 1.0);
    return out;
}
