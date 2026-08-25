// Scene shaders for the webgpu backend.
//
// Direct ports of the forge GLSL pipeline (scene.vert, scene_body.glsl,
// lighting.glsl, id.vert/id.frag) with three deliberate deltas:
//
// - Clip space follows WebGPU conventions (z in [0,1]); the perspective and
//   orthographic projections are built on the CPU side in ``targets.py``.
// - ``gl_ClipDistance`` becomes a fragment discard on a plane equation in the
//   frame uniforms (clip_plane); the main pass binds the (0,0,0,1) no-op.
// - Depth and object-id export runs as a separate single-sampled MRT pass that
//   re-rasterizes the scene and writes GL-compatible nonlinear depth.  WebGPU
//   cannot resolve multisampled integer or depth attachments, so this replaces
//   forge's ``blit_depth``/``blit_color`` resolve of the shared MSAA buffers.
//
// Shadow sampling lives in shadow_sample.wgsl (prepended by load_wgsl); the
// GLSL USE_SHADOW define becomes runtime gating on the shadow_counts fields.

// Keep in sync with instances.py (144-byte stride) and targets.py (frame block).
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
    flags: vec4f,               // x orthographic, y reflection output, z MuJoCo classic
    ids: vec4u,                 // x selected id, y light count
    image_light: vec4f,         // x gain (intensity / 5000), y max mip level
    clip_plane: vec4f,          // reflection clip plane; (0,0,0,1) keeps everything
    reflection: vec4f,          // xy reflection target size; x=0 disables sampling
};

struct Instance {
    model: mat4x4f,
    color: vec4f,               // linear rgba
    material: vec4f,            // emission, specular, shininess, reflectance
    texcoef: vec4f,             // xy scale; z=1 box mapping, zw<0 infinite-plane light grid
    cubecoef: vec4f,            // xyz object-linear scale, w capsule-axis offset
    object_id: u32,
    // The struct ends at 132 and rounds up to a 144-byte array stride,
    // matching instances.py's padded dtype.
};

struct Lights {
    pos: array<vec4f, 100>,       // xyz position, w kind
    dir: array<vec4f, 100>,       // xyz direction, w cutoff cosine
    diffuse: array<vec4f, 100>,   // rgb linear, w spot exponent
    specular: array<vec4f, 100>,
    atten: array<vec4f, 100>,     // constant, linear, quadratic, range
    // Shadow block, mirroring the shadow_sample.glsl uniform set (keep in
    // sync with lighting.py LIGHTS_DTYPE).
    shadow_matrix: array<mat4x4f, 3>,  // u_shadow_matrix
    shadow_tile: array<vec4f, 3>,      // u_shadow_tile
    shadow_splits: vec4f,              // xyz u_shadow_splits
    shadow_texel: vec4f,               // xyz u_shadow_texel
    shadow_bias: vec4f,                // xy u_shadow_bias
    shadow_counts: vec4f,              // x u_shadow_count, y u_local_count, z u_shadow_light
    local_matrix: array<mat4x4f, 8>,   // u_local_matrix (spot view-projections)
    local_pos: array<vec4f, 8>,        // u_local_pos: xyz position, w range
    local_texel: array<f32, 8>,        // u_local_texel
    local_radius: array<f32, 8>,       // u_local_radius
    local_slot: array<i32, 100>,       // u_local_slot: shadow slot per light, -1 none
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;
@group(0) @binding(2) var<storage, read> lights: Lights;
@group(1) @binding(0) var albedo_tex: texture_2d<f32>;
@group(1) @binding(1) var albedo_sampler: sampler;
@group(1) @binding(2) var cube_albedo_tex: texture_cube<f32>;
@group(1) @binding(3) var cube_albedo_sampler: sampler;
@group(2) @binding(0) var image_light_tex: texture_cube<f32>;
@group(2) @binding(1) var image_light_sampler: sampler;
// Planar reflection color targets (mirrors u_reflection0-3 in scene_body.glsl);
// the reflect pass binds 1x1 fallbacks so no reflection feeds back into itself.
@group(4) @binding(0) var reflection0: texture_2d<f32>;
@group(4) @binding(1) var reflection1: texture_2d<f32>;
@group(4) @binding(2) var reflection2: texture_2d<f32>;
@group(4) @binding(3) var reflection3: texture_2d<f32>;
@group(4) @binding(4) var reflection_sampler: sampler;

const FORGE_GAMMA: f32 = 2.2;
const FORGE_KNEE: f32 = 0.8;

fn srgb_to_linear3(c: vec3f) -> vec3f {
    let lo = c / 12.92;
    let hi = pow((max(c, vec3f(0.0)) + 0.055) / 1.055, vec3f(2.4));
    return select(hi, lo, c <= vec3f(0.04045));
}

fn linear_to_srgb3(c_in: vec3f) -> vec3f {
    let c = max(c_in, vec3f(0.0));
    let lo = c * 12.92;
    let hi = 1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3f(0.0031308));
}

fn lighting_color(c: vec3f) -> vec3f {
    return select(c, linear_to_srgb3(c), frame.flags.z > 0.5);
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
    specular_mod: vec3f,
    specular: f32, shininess: f32, atten: f32, shadow: f32,
) -> vec3f {
    let ndl = max(dot(n, l), 0.0);
    if ndl <= 0.0 || atten <= 0.0 {
        return vec3f(0.0);
    }
    let h = normalize(l + view_dir);
    let spec = specular * pow(max(dot(n, h), 0.0), max(shininess * 128.0, 1e-3));
    if frame.flags.z > 0.5 {
        return atten * shadow * (
            ndl * lighting_color(diffuse_rgb) * albedo
                + lighting_color(specular_rgb) * spec * specular_mod
        );
    }
    return atten * shadow * ndl * (
        lighting_color(diffuse_rgb) * albedo
            + lighting_color(specular_rgb) * spec * specular_mod
    );
}

fn shade(
    albedo: vec3f, normal: vec3f, world_pos: vec3f,
    emission: f32, specular: f32, shininess: f32, view_depth: f32,
    texture_color: vec3f,
) -> vec3f {
    let n = normalize(normal);
    let view_dir = normalize(frame.camera_pos.xyz - world_pos);
    let ambient = select(
        ambient_linear(frame.ambient.xyz),
        clamp(frame.ambient.xyz, vec3f(0.0), vec3f(1.0)),
        frame.flags.z > 0.5,
    );
    var color = ambient * albedo;
    let specular_mod = select(vec3f(1.0), texture_color, frame.flags.z > 0.5);

    // Image-based lighting, mirroring lighting.glsl: diffuse irradiance comes
    // from the most blurred mip, specular from the roughness-selected LOD.
    if frame.image_light.x > 0.0 {
        let cube_n = vec3f(n.x, n.z, -n.y);
        let reflected = reflect(-view_dir, n);
        let cube_r = vec3f(reflected.x, reflected.z, -reflected.y);
        let diffuse_ibl = textureSampleLevel(
            image_light_tex, image_light_sampler, cube_n, frame.image_light.y
        ).rgb;
        let roughness = 1.0 - clamp(shininess, 0.0, 1.0);
        let specular_ibl = textureSampleLevel(
            image_light_tex, image_light_sampler, cube_r, roughness * frame.image_light.y
        ).rgb;
        color += frame.image_light.x * (
            lighting_color(diffuse_ibl) * albedo
                + specular * lighting_color(specular_ibl) * specular_mod
        );
    }

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
        // lighting.glsl:82-86 under USE_SHADOW; here the shadow_counts fields
        // gate at runtime (-1 / 0 when the shadow pass did not run).
        var shadow = 1.0;
        if i == i32(lights.shadow_counts.z) {
            shadow = shadow_factor(world_pos, n, view_depth);
        }
        if kind != 0 {
            shadow *= local_shadow_factor(kind, lights.local_slot[i], world_pos, n);
        }
        color += light_term(
            albedo, n, l, view_dir,
            lights.diffuse[i].rgb, lights.specular[i].rgb,
            specular_mod,
            specular, shininess, atten, shadow,
        );
    }

    if frame.headlight_diffuse.w > 0.5 {
        color += light_term(
            albedo, n, -normalize(frame.camera_dir.xyz), view_dir,
            frame.headlight_diffuse.rgb, frame.headlight_specular.rgb,
            specular_mod,
            specular, shininess, 1.0, 1.0,
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
    @location(7) reflect: f32,       // planar reflection coefficient (encoded < 0)
    @location(8) cube: vec3f,
    @location(9) cube_on: f32,
};

@vertex
fn vs_scene(
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f,
    @builtin(instance_index) instance_index: u32,
) -> SceneOut {
    return scene_vertex(position, normal, uv, instance_index);
}

fn scene_vertex(position: vec3f, normal: vec3f, uv: vec2f, instance_index: u32) -> SceneOut {
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
    out.cube = position * inst.cubecoef.xyz + vec3f(0.0, 0.0, inst.cubecoef.w);
    out.cube_on = select(0.0, 1.0, dot(abs(inst.cubecoef.xyz), vec3f(1.0)) > 0.0);
    out.color = inst.color;
    out.material = inst.material.xyz;
    // scene.vert:60-62: negative reflectance encodes (layer, top-face); the
    // top-face code keeps only the local +Z face of a box reflective.
    out.reflect = inst.material.w;
    if out.reflect < 0.0 && (-out.reflect % 4.0) >= 2.0 && normal.z < 0.5 {
        out.reflect = 0.0;
    }
    out.view_depth = -(frame.view * world).z;
    out.selected = select(0.0, 1.0, frame.ids.x != 0u && inst.object_id == frame.ids.x);
    return out;
}

struct SurfaceSample {
    base: vec4f,
    texture_color: vec3f,
};

fn scene_surface(in: SceneOut) -> SurfaceSample {
    var texel: vec4f;
    if in.cube_on > 0.5 {
        texel = textureSample(cube_albedo_tex, cube_albedo_sampler, in.cube);
    } else {
        texel = textureSample(albedo_tex, albedo_sampler, in.uv);
    }
    var surface = in.color.rgb;
    if frame.flags.z > 0.5 {
        surface = gamma_encode(surface);
        texel = vec4f(linear_to_srgb3(texel.rgb), texel.a);
    }
    var base = vec4f(surface * texel.rgb, in.color.a * texel.a);
    if in.selected > 0.5 {
        let albedo = mix(base.rgb, frame.highlight_color.xyz, frame.highlight.x);
        base = vec4f(albedo, base.a);
    }
    var out: SurfaceSample;
    out.base = base;
    out.texture_color = texel.rgb;
    return out;
}

fn apply_atmosphere(lit_in: vec3f, view_depth: f32) -> vec3f {
    var lit = lit_in;
    let fog = frame.fog.z * smoothstep(frame.fog.x, max(frame.fog.y, frame.fog.x + 1e-6), view_depth);
    let haze = 1.0 - exp(-max(frame.fog.w, 0.0) * max(view_depth, 0.0));
    let fog_color = select(
        srgb_to_linear3(frame.fog_color.xyz), frame.fog_color.xyz, frame.flags.z > 0.5
    );
    let haze_color = select(
        srgb_to_linear3(frame.haze_color.xyz), frame.haze_color.xyz, frame.flags.z > 0.5
    );
    lit = mix(lit, fog_color, fog);
    lit = mix(lit, haze_color, haze);
    return lit;
}

@fragment
fn fs_scene(in: SceneOut) -> @location(0) vec4f {
    return scene_fragment(in);
}

fn scene_fragment(in: SceneOut) -> vec4f {
    let surface = scene_surface(in);
    let base = surface.base;
    // gl_ClipDistance[0] port: drop fragments behind the reflection plane.
    // The main pass binds (0,0,0,1), which never discards.
    if dot(in.world, frame.clip_plane.xyz) + frame.clip_plane.w < 0.0 {
        discard;
    }
    var emission = in.material.x;
    if in.selected > 0.5 {
        emission += frame.highlight.y;
    }
    var lit = shade(
        base.rgb, in.normal, in.world, emission, in.material.y, in.material.z, in.view_depth,
        surface.texture_color,
    );
    // scene_body.glsl:72-89: negative reflectance carries (layer, top-face);
    // add the reflected color before atmosphere, in linear space.
    if in.reflect < 0.0 && frame.reflection.x > 0.0 {
        let code = -in.reflect;
        let layer = i32(floor(code * 0.25));
        let surface = code - f32(layer * 4);
        let reflectance = surface - select(0.0, 2.0, surface >= 2.0);
        let reflection_uv = in.clip.xy / frame.reflection.xy;
        // Explicit LOD: the reflection targets have a single mip level, and
        // textureSample would be rejected in this non-uniform branch.
        var reflected: vec3f;
        if layer == 0 {
            reflected = textureSampleLevel(reflection0, reflection_sampler, reflection_uv, 0.0).rgb;
        } else if layer == 1 {
            reflected = textureSampleLevel(reflection1, reflection_sampler, reflection_uv, 0.0).rgb;
        } else if layer == 2 {
            reflected = textureSampleLevel(reflection2, reflection_sampler, reflection_uv, 0.0).rgb;
        } else {
            reflected = textureSampleLevel(reflection3, reflection_sampler, reflection_uv, 0.0).rgb;
        }
        lit += reflectance * reflected;
    }
    let shaded = apply_atmosphere(lit, in.view_depth);
    // u_linear_out port: the reflection pass stores linear HDR color and the
    // main pass applies tonemap/gamma when compositing.
    if frame.flags.y > 0.5 {
        return vec4f(shaded, base.a);
    }
    var rgb: vec3f;
    if frame.flags.z > 0.5 {
        rgb = clamp(shaded * frame.shading.x, vec3f(0.0), vec3f(1.0));
    } else {
        rgb = finish_color(shaded, frame.shading.x, frame.shading.y > 0.5);
    }
    return vec4f(rgb, base.a);
}

@fragment
fn fs_albedo(in: SceneOut) -> @location(0) vec4f {
    let base = scene_surface(in).base;
    let albedo_rgb = select(gamma_encode(base.rgb), base.rgb, frame.flags.z > 0.5);
    return vec4f(albedo_rgb, base.a);
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

// ---- wireframe variant --------------------------------------------------------
// forge injects barycentrics in wireframe.geom (a geometry stage); WebGPU has
// none, so the mesh store supplies a non-indexed stream with a bary attribute
// (see meshes.py GpuMesh.wireframe).  Fragment math mirrors the WIREFRAME
// branch of scene_body.glsl: the mix runs after finish_color, in display space.

const WIRE_COLOR: vec3f = vec3f(0.10, 0.10, 0.12);  // forge opaque.WIRE_COLOR
const WIRE_WIDTH: f32 = 1.2;                          // forge opaque.WIRE_WIDTH

struct SceneWireOut {
    @builtin(position) clip: vec4f,
    @location(0) world: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f,
    @location(3) color: vec4f,
    @location(4) material: vec3f,    // emission, specular, shininess
    @location(5) view_depth: f32,
    @location(6) selected: f32,
    @location(7) reflect: f32,
    @location(8) cube: vec3f,
    @location(9) cube_on: f32,
    @location(10) bary: vec3f,
};

@vertex
fn vs_scene_wire(
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f,
    @location(3) bary: vec3f,
    @builtin(instance_index) instance_index: u32,
) -> SceneWireOut {
    let base = scene_vertex(position, normal, uv, instance_index);
    var out: SceneWireOut;
    out.clip = base.clip;
    out.world = base.world;
    out.normal = base.normal;
    out.uv = base.uv;
    out.color = base.color;
    out.material = base.material;
    out.view_depth = base.view_depth;
    out.selected = base.selected;
    out.reflect = base.reflect;
    out.cube = base.cube;
    out.cube_on = base.cube_on;
    out.bary = bary;
    return out;
}

fn wire_base(in: SceneWireOut) -> SceneOut {
    var out: SceneOut;
    out.clip = in.clip;
    out.world = in.world;
    out.normal = in.normal;
    out.uv = in.uv;
    out.color = in.color;
    out.material = in.material;
    out.view_depth = in.view_depth;
    out.selected = in.selected;
    out.reflect = in.reflect;
    out.cube = in.cube;
    out.cube_on = in.cube_on;
    return out;
}

@fragment
fn fs_scene_wire(in: SceneWireOut) -> @location(0) vec4f {
    let base = scene_fragment(wire_base(in));
    let d = min(in.bary.x, min(in.bary.y, in.bary.z));
    let w = max(fwidth(d), 1e-6) * max(WIRE_WIDTH, 0.1);
    return vec4f(mix(WIRE_COLOR, base.rgb, smoothstep(0.0, w, d)), base.a);
}

// ---- overdraw variant ---------------------------------------------------------
// scene_body.glsl DEBUG_VIEW == 4: a fixed additive increment per covered
// layer; the pipeline variant supplies the one/one blend and disables depth.

const OVERDRAW_STEP: f32 = 1.0 / 16.0;

@fragment
fn fs_overdraw(in: SceneOut) -> @location(0) vec4f {
    return vec4f(vec3f(OVERDRAW_STEP), 0.0);
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
