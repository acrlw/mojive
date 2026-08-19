"""WGSL shaders for the webgpu backend.

These are direct ports of the forge GLSL pipeline (scene.vert, scene_body.glsl,
lighting.glsl, id.vert/id.frag) with three deliberate deltas:

- Clip space follows WebGPU conventions (z in [0,1]); the perspective and
  orthographic projections are built on the CPU side in ``targets.py``.
- ``gl_ClipDistance`` (planar reflection clipping) is dropped; the reflection
  pass is not part of this backend.
- Depth and object-id export runs as a separate single-sampled MRT pass that
  re-rasterizes the scene and writes GL-compatible nonlinear depth.  WebGPU
  cannot resolve multisampled integer or depth attachments, so this replaces
  forge's ``blit_depth``/``blit_color`` resolve of the shared MSAA buffers.
"""

from __future__ import annotations

# Keep in sync with instances.py (128-byte stride) and targets.py (frame block).
SCENE_WGSL = """
struct Frame {
    view_proj: mat4x4f,         // WebGPU-clip view-projection
    view: mat4x4f,
    camera_pos: vec4f,
    camera_dir: vec4f,
    ambient: vec4f,             // raw sRGB ambient; converted in shade()
    headlight_diffuse: vec4f,   // rgb linear, w enabled
    headlight_specular: vec4f,  // rgb linear
    fog: vec4f,                 // start, end, enabled, haze density
    fog_color: vec4f,           // raw sRGB
    haze_color: vec4f,          // raw sRGB
    highlight_color: vec4f,
    highlight: vec4f,           // x blend, y emission
    shading: vec4f,             // exposure, tonemap on, near, far
    flags: vec4f,               // x orthographic
    ids: vec4u,                 // x selected id, y light count
};

struct Instance {
    model: mat4x4f,
    color: vec4f,               // linear rgba
    material: vec4f,            // emission, specular, shininess, reflectance
    texcoef: vec4f,             // scale/offset; z=1 selects box face-axis mapping
    object_id: u32,
    // No trailing pad field: a vec3f member would push the array stride to
    // 144 (vec3 alignment is 16 in WGSL).  The struct ends at 116 and rounds
    // up to a 128-byte array stride, matching instances.py's padded dtype.
};

struct Lights {
    pos: array<vec4f, 100>,       // xyz position, w kind
    dir: array<vec4f, 100>,       // xyz direction, w cutoff cosine
    diffuse: array<vec4f, 100>,   // rgb linear, w spot exponent
    specular: array<vec4f, 100>,
    atten: array<vec4f, 100>,     // constant, linear, quadratic, range
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;
@group(0) @binding(2) var<storage, read> lights: Lights;
@group(1) @binding(0) var albedo_tex: texture_2d<f32>;
@group(1) @binding(1) var albedo_sampler: sampler;

const FORGE_GAMMA: f32 = 2.2;
const FORGE_KNEE: f32 = 0.8;

fn srgb_to_linear3(c: vec3f) -> vec3f {
    let lo = c / 12.92;
    let hi = pow((max(c, vec3f(0.0)) + 0.055) / 1.055, vec3f(2.4));
    return select(hi, lo, c <= vec3f(0.04045));
}

fn gamma_encode(c: vec3f) -> vec3f {
    return pow(max(c, vec3f(0.0)), vec3f(1.0 / FORGE_GAMMA));
}

fn ambient_linear(a: vec3f) -> vec3f {
    return srgb_to_linear3(clamp(a, vec3f(0.0), vec3f(1.0)));
}

fn softroll(excess: f32, headroom: f32) -> f32 {
    return headroom * excess / (excess + headroom);
}

fn tonemap(c_in: vec3f) -> vec3f {
    let c = c_in;
    let peak = max(c.r, max(c.g, c.b));
    if peak <= FORGE_KNEE {
        return c;
    }
    let headroom = 1.0 - FORGE_KNEE;
    let mapped = FORGE_KNEE + softroll(peak - FORGE_KNEE, headroom);
    return c * (mapped / peak);
}

fn finish_color(c_in: vec3f, exposure: f32, tonemap_on: bool) -> vec3f {
    var c = c_in * exposure;
    if tonemap_on {
        c = tonemap(c);
    } else {
        c = clamp(c, vec3f(0.0), vec3f(1.0));
    }
    return gamma_encode(clamp(c, vec3f(0.0), vec3f(1.0)));
}

fn light_term(
    albedo: vec3f, n: vec3f, l: vec3f, view_dir: vec3f,
    diffuse_rgb: vec3f, specular_rgb: vec3f,
    specular: f32, shininess: f32, atten: f32,
) -> vec3f {
    let ndl = max(dot(n, l), 0.0);
    if ndl <= 0.0 || atten <= 0.0 {
        return vec3f(0.0);
    }
    let h = normalize(l + view_dir);
    let spec = specular * pow(max(dot(n, h), 0.0), max(shininess * 128.0, 1e-3));
    return atten * ndl * (diffuse_rgb * albedo + specular_rgb * spec);
}

fn shade(
    albedo: vec3f, normal: vec3f, world_pos: vec3f,
    emission: f32, specular: f32, shininess: f32,
) -> vec3f {
    let n = normalize(normal);
    let view_dir = normalize(frame.camera_pos.xyz - world_pos);
    var color = ambient_linear(frame.ambient.xyz) * albedo;

    let light_count = i32(frame.ids.y);
    for (var i = 0; i < light_count; i = i + 1) {
        let kind = i32(lights.pos[i].w + 0.5);
        var l: vec3f;
        var atten = 1.0;
        if kind == 0 {
            l = -normalize(lights.dir[i].xyz);
        } else {
            let to_light = lights.pos[i].xyz - world_pos;
            let dist = length(to_light);
            l = to_light / max(dist, 1e-6);
            let k = lights.atten[i].xyz;
            atten = 1.0 / max(k.x + k.y * dist + k.z * dist * dist, 1e-6);
            if lights.atten[i].w > 0.0 && dist > lights.atten[i].w {
                atten = 0.0;
            }
            if kind == 2 {
                let cd = dot(-l, normalize(lights.dir[i].xyz));
                if cd < lights.dir[i].w {
                    atten = 0.0;
                } else {
                    atten = atten * pow(max(cd, 0.0), lights.diffuse[i].w);
                }
            }
        }
        color += light_term(
            albedo, n, l, view_dir,
            lights.diffuse[i].rgb, lights.specular[i].rgb,
            specular, shininess, atten,
        );
    }

    if frame.headlight_diffuse.w > 0.5 {
        color += light_term(
            albedo, n, -normalize(frame.camera_dir.xyz), view_dir,
            frame.headlight_diffuse.rgb, frame.headlight_specular.rgb,
            specular, shininess, 1.0,
        );
    }

    return color + emission * albedo;
}

// transpose(inverse(mat3(m))) without a matrix inverse builtin: the adjugate
// columns divided by the determinant, matching the GLSL computation bit-for-bit
// in fp32 up to rounding.
fn normal_transform(m: mat4x4f, n: vec3f) -> vec3f {
    let c0 = m[0].xyz;
    let c1 = m[1].xyz;
    let c2 = m[2].xyz;
    let r0 = cross(c1, c2);
    let r1 = cross(c2, c0);
    let r2 = cross(c0, c1);
    let det = dot(c0, r0);
    return (r0 * n.x + r1 * n.y + r2 * n.z) / det;
}

struct SceneOut {
    @builtin(position) clip: vec4f,
    @location(0) world: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f,
    @location(3) color: vec4f,
    @location(4) material: vec3f,    // emission, specular, shininess
    @location(5) view_depth: f32,
    @location(6) selected: f32,
};

@vertex
fn vs_scene(
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f,
    @builtin(instance_index) instance_index: u32,
) -> SceneOut {
    let inst = instances[instance_index];
    let model = inst.model;
    let world = model * vec4f(position, 1.0);

    var out: SceneOut;
    out.clip = frame.view_proj * world;
    out.world = world.xyz;
    out.normal = normal_transform(model, normal);

    var texcoord = uv;
    if inst.texcoef.z > 0.5 {
        let extent = vec3f(length(model[0].xyz), length(model[1].xyz), length(model[2].xyz));
        let repeat = inst.texcoef.xy / max(extent.xy, vec2f(1e-7));
        let axis = abs(normal);
        var scale: vec2f;
        if axis.x >= axis.y && axis.x >= axis.z {
            scale = vec2f(extent.y * repeat.x, extent.z * repeat.y);
        } else if axis.y >= axis.z {
            scale = vec2f(extent.x * repeat.x, extent.z * repeat.y);
        } else {
            scale = inst.texcoef.xy;
        }
        texcoord = uv * scale;
    } else {
        texcoord = uv * inst.texcoef.xy + inst.texcoef.zw;
    }
    out.uv = texcoord;
    out.color = inst.color;
    out.material = inst.material.xyz;
    out.view_depth = -(frame.view * world).z;
    out.selected = select(0.0, 1.0, frame.ids.x != 0u && inst.object_id == frame.ids.x);
    return out;
}

fn scene_albedo(in: SceneOut) -> vec4f {
    var base = in.color * textureSample(albedo_tex, albedo_sampler, in.uv);
    if in.selected > 0.5 {
        let albedo = mix(base.rgb, frame.highlight_color.xyz, frame.highlight.x);
        base = vec4f(albedo, base.a);
    }
    return base;
}

fn apply_atmosphere(lit_in: vec3f, view_depth: f32) -> vec3f {
    var lit = lit_in;
    let fog = frame.fog.z * smoothstep(frame.fog.x, max(frame.fog.y, frame.fog.x + 1e-6), view_depth);
    let haze = 1.0 - exp(-max(frame.fog.w, 0.0) * max(view_depth, 0.0));
    lit = mix(lit, srgb_to_linear3(frame.fog_color.xyz), fog);
    lit = mix(lit, srgb_to_linear3(frame.haze_color.xyz), haze);
    return lit;
}

@fragment
fn fs_scene(in: SceneOut) -> @location(0) vec4f {
    let base = scene_albedo(in);
    var emission = in.material.x;
    if in.selected > 0.5 {
        emission += frame.highlight.y;
    }
    let lit = shade(base.rgb, in.normal, in.world, emission, in.material.y, in.material.z);
    let rgb = finish_color(apply_atmosphere(lit, in.view_depth), frame.shading.x, frame.shading.y > 0.5);
    return vec4f(rgb, base.a);
}

@fragment
fn fs_albedo(in: SceneOut) -> @location(0) vec4f {
    let base = scene_albedo(in);
    return vec4f(gamma_encode(base.rgb), base.a);
}

@fragment
fn fs_normal(in: SceneOut) -> @location(0) vec4f {
    return vec4f(normalize(in.normal) * 0.5 + 0.5, in.color.a);
}

@fragment
fn fs_depth(in: SceneOut) -> @location(0) vec4f {
    let range = max(frame.shading.w - frame.shading.z, 1e-6);
    let d = clamp((in.view_depth - frame.shading.z) / range, 0.0, 1.0);
    return vec4f(vec3f(1.0 - d), in.color.a);
}

// ---- depth/id export pass ---------------------------------------------------

struct ExportOut {
    @builtin(position) clip: vec4f,
    @location(0) view_depth: f32,
    @location(1) @interpolate(flat) object_id: u32,
};

@vertex
fn vs_export(
    @location(0) position: vec3f,
    @builtin(instance_index) instance_index: u32,
) -> ExportOut {
    let inst = instances[instance_index];
    let world = inst.model * vec4f(position, 1.0);
    var out: ExportOut;
    out.clip = frame.view_proj * world;
    out.view_depth = -(frame.view * world).z;
    out.object_id = inst.object_id;
    return out;
}

struct ExportFrag {
    @location(0) depth: f32,        // GL-compatible nonlinear depth in [0,1]
    @location(1) object_id: u32,
};

@fragment
fn fs_export(in: ExportOut) -> ExportFrag {
    let near = frame.shading.z;
    let far = frame.shading.w;
    let d = max(in.view_depth, 1e-6);
    var depth: f32;
    if frame.flags.x > 0.5 {
        depth = clamp((d - near) / max(far - near, 1e-6), 0.0, 1.0);
    } else {
        let ndc = (far + near) / (far - near) - (2.0 * far * near) / ((far - near) * d);
        depth = clamp(0.5 + 0.5 * ndc, 0.0, 1.0);
    }
    var out: ExportFrag;
    out.depth = depth;
    out.object_id = in.object_id;
    return out;
}
"""
