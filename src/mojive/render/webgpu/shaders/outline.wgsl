// Selection outline for the webgpu backend.
//
// Direct port of opengl's outline pipeline (passes/outline.py + outline.frag +
// id.vert/id.frag with ID_ONLY_SELECTED|ID_MASK_FLOAT):
//
// - vs_outline_mask/fs_outline_mask rasterize the selected object into a
//   single-sampled r8unorm mask with no depth test, so the silhouette covers
//   the object even where an occluder hides it.  Deliberate delta from opengl:
//   the opengl mask is 4x MSAA resolved into R8, which carries subpixel
//   coverage; WebGPU cannot resolve multisampled targets into a sampled
//   texture without an extra pass, and the dilation shader below already does
//   its own antialiasing, so a single-sampled binary mask is used instead.
// - vs_outline/fs_outline draw a fullscreen triangle that dilates the mask by
//   a circular kernel (radius OUTLINE_RADIUS + 1px of AA) and alpha-blends
//   the ring into the main color target, with the border-connectivity fix
//   that keeps the outline closed when the object is clipped by the viewport.

struct OutlineInstances {
    model: mat4x4f,
    color: vec4f,
    material: vec4f,
    texcoef: vec4f,
    cubecoef: vec4f,
    object_id: u32,
    // Ends at 132 bytes; the array stride rounds up to 144, matching
    // instances.py (same layout as scene.wgsl's Instance).
};

struct OutlineMaskUniforms {
    view_proj: mat4x4f,
    params: vec4u,              // x selected id
};

@group(0) @binding(0) var<uniform> outline_mask_uniforms: OutlineMaskUniforms;
@group(0) @binding(1) var<storage, read> outline_instances: array<OutlineInstances>;

struct OutlineMaskOut {
    @builtin(position) clip: vec4f,
    @location(0) @interpolate(flat) object_id: u32,
};

@vertex
fn vs_outline_mask(
    @location(0) position: vec3f,
    @builtin(instance_index) instance_index: u32,
) -> OutlineMaskOut {
    let inst = outline_instances[instance_index];
    var out: OutlineMaskOut;
    out.clip = outline_mask_uniforms.view_proj * (inst.model * vec4f(position, 1.0));
    out.object_id = inst.object_id;
    return out;
}

@fragment
fn fs_outline_mask(in: OutlineMaskOut) -> @location(0) f32 {
    if (in.object_id != outline_mask_uniforms.params.x) {
        discard;
    }
    return 1.0;
}

// ---- dilation composite -------------------------------------------------------

struct OutlineCompositeUniforms {
    color: vec4f,
    size: vec4u,                // xy target size
};

@group(0) @binding(2) var outline_mask_tex: texture_2d<f32>;
@group(0) @binding(3) var<uniform> outline_composite: OutlineCompositeUniforms;

const OUTLINE_RADIUS: i32 = 3;  // opengl OUTLINE_RADIUS
const AA_RADIUS: i32 = OUTLINE_RADIUS + 1;
const CORE_R2: f32 = f32(OUTLINE_RADIUS * OUTLINE_RADIUS);
const OUTER_R2: f32 = f32(AA_RADIUS * AA_RADIUS);

@vertex
fn vs_outline(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4f {
    let p = vec2f(f32((vertex_index << 1u) & 2u), f32(vertex_index & 2u));
    return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

fn coverage_at(c: vec2i) -> f32 {
    if (c.x < 0 || c.y < 0 || c.x >= i32(outline_composite.size.x) || c.y >= i32(outline_composite.size.y)) {
        return 0.0;
    }
    return textureLoad(outline_mask_tex, c, 0).r;
}

fn selected_at(c: vec2i) -> bool {
    return coverage_at(c) >= 0.5;
}

fn outline_alpha(p: vec2i) -> f32 {
    var expanded = 0.0;
    for (var dy = -AA_RADIUS; dy <= AA_RADIUS; dy = dy + 1) {
        for (var dx = -AA_RADIUS; dx <= AA_RADIUS; dx = dx + 1) {
            let r2 = f32(dx * dx + dy * dy);
            if (r2 > OUTER_R2) {
                continue;
            }
            let kernel = 1.0 - smoothstep(CORE_R2, OUTER_R2, r2);
            expanded = max(expanded, coverage_at(p + vec2i(dx, dy)) * kernel);
        }
    }
    return max(expanded - coverage_at(p), 0.0);
}

fn runs_to_border(p: vec2i, step: vec2i, to_border: i32) -> bool {
    if (to_border >= OUTLINE_RADIUS) {
        return false;
    }
    for (var i = 1; i <= OUTLINE_RADIUS; i = i + 1) {
        if (i > to_border) {
            break;
        }
        if (!selected_at(p + step * i)) {
            return false;
        }
    }
    return true;
}

fn clipped_by_border(p: vec2i) -> bool {
    return runs_to_border(p, vec2i(-1, 0), p.x)
        || runs_to_border(p, vec2i(1, 0), i32(outline_composite.size.x) - 1 - p.x)
        || runs_to_border(p, vec2i(0, -1), p.y)
        || runs_to_border(p, vec2i(0, 1), i32(outline_composite.size.y) - 1 - p.y);
}

@fragment
fn fs_outline(@builtin(position) frag_coord: vec4f) -> @location(0) vec4f {
    let p = vec2i(frag_coord.xy);
    let center = coverage_at(p);
    var alpha = outline_alpha(p);
    if (center > 0.0 && clipped_by_border(p)) {
        alpha = max(alpha, center);
    }
    if (alpha <= 0.0) {
        discard;
    }
    let color = outline_composite.color;
    return vec4f(color.rgb, color.a * alpha);
}
