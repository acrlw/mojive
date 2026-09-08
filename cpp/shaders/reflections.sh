SAMPLER2D(s_reflection0, 5);
SAMPLER2D(s_reflection1, 6);
SAMPLER2D(s_reflection2, 7);
SAMPLER2D(s_reflection3, 8);
uniform vec4 u_reflectionConfig, u_reflectionPlane;
vec3 reflectedColor(float slot,vec2 position) {
    vec2 uv=position/u_reflectionConfig.yz;
    if(slot<1.5) return texture2D(s_reflection0,uv).rgb;
    if(slot<2.5) return texture2D(s_reflection1,uv).rgb;
    if(slot<3.5) return texture2D(s_reflection2,uv).rgb;
    return texture2D(s_reflection3,uv).rgb;
}
