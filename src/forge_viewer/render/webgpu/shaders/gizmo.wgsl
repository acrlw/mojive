// Native 3D gizmo handles; port of forge's gizmo.vert/gizmo.frag.
//
// One uniform block per handle draw, selected by dynamic offset: model
// carries the screen-constant scale computed CPU-side, params.x is the
// center-hole mask radius in mesh-local units (0 disables the hole).
struct GizmoUniforms {
    view_proj : mat4x4<f32>,
    view : mat4x4<f32>,
    model : mat4x4<f32>,
    color : vec4<f32>,
    params : vec4<f32>,
};

@group(0) @binding(0) var<uniform> u : GizmoUniforms;

struct VertexOut {
    @builtin(position) clip : vec4<f32>,
    @location(0) color : vec4<f32>,
    @location(1) normal : vec3<f32>,
    @location(2) view_pos : vec3<f32>,
    @location(3) local : vec3<f32>,
};

@vertex
fn vs_gizmo(
    @location(0) in_position : vec3<f32>,
    @location(1) in_normal : vec3<f32>,
) -> VertexOut {
    var out : VertexOut;
    let world = u.model * vec4<f32>(in_position, 1.0);
    out.color = u.color;
    out.local = in_position;
    out.normal = (u.model * vec4<f32>(in_normal, 0.0)).xyz;
    out.view_pos = (u.view * world).xyz;
    var clip = u.view_proj * world;
    // forge squashes the handle against the near plane with clip.z =
    // -0.999*w + 0.001*z (GL z in [-1, 1]); with WebGPU z in [0, 1] the near
    // plane is 0, keeping the same tiny relative-order weight on z.
    clip.z = 0.001 * clip.w + 0.001 * clip.z;
    out.clip = clip;
    return out;
}

@fragment
fn fs_gizmo(in : VertexOut) -> @location(0) vec4<f32> {
    let mask_radius = u.params.x;
    if (mask_radius > 0.0 && length(in.local) < mask_radius) {
        discard;
    }
    let n = normalize(in.normal);
    let to_eye = normalize(-in.view_pos);
    let facing = abs(dot(n, to_eye));
    let shade = 0.72 + 0.28 * facing;
    return vec4<f32>(in.color.rgb * shade, in.color.a);
}
