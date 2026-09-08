#version 450
layout(location=1) in vec4 a;
layout(location=2) in vec4 b;
layout(location=3) in float depth;
layout(location=0) out vec4 out_id;
layout(location=1) out vec4 out_s0;
layout(location=2) out vec4 out_s1;
layout(location=3) out float out_depth;
vec4 pack_words(vec2 f) {
    uvec2 w=uvec2(floor(f+0.5)); return vec4(w.x&255u,w.x>>8u,w.y&255u,w.y>>8u)/255.0;
}
void main() { out_id=pack_words(a.xy);out_s0=pack_words(a.zw);out_s1=pack_words(b.xy);out_depth=depth; }
