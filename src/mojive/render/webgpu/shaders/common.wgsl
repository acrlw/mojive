// Shared color helpers for the webgpu backend shaders, mirroring opengl's
// common.glsl.  Prepended to other WGSL sources by programs.load_wgsl; do not
// compile this chunk standalone.  scene.wgsl predates this file and keeps its
// own inline copies.

const OPENGL_GAMMA: f32 = 2.2;
const OPENGL_KNEE: f32 = 0.8;

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

fn gamma_encode(c: vec3f) -> vec3f {
    return pow(max(c, vec3f(0.0)), vec3f(1.0 / OPENGL_GAMMA));
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
    if peak <= OPENGL_KNEE {
        return c;
    }
    let headroom = 1.0 - OPENGL_KNEE;
    let mapped = OPENGL_KNEE + softroll(peak - OPENGL_KNEE, headroom);
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
