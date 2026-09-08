vec3 srgbToLinear(vec3 c) {
    return mix(c / 12.92, pow((max(c, vec3(0)) + .055) / 1.055, vec3(2.4)), step(vec3(.04045), c));
}
vec3 linearToSrgb(vec3 c) {
    c = max(c, vec3(0));
    return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - .055, step(vec3(.0031308), c));
}
vec3 gammaEncode(vec3 c) { return pow(max(c, vec3(0)), vec3(1.0 / 2.2)); }
vec3 finishColor(vec3 c) {
    float peak = max(c.r, max(c.g, c.b));
    if (u_options.z > .5 && peak > .8) c *= (.8 + .2 * (peak - .8) / (peak - .6)) / peak;
    return gammaEncode(clamp(c, 0.0, 1.0));
}
