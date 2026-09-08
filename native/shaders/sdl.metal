#include <metal_stdlib>
using namespace metal;
struct Camera { float4 v[4]; float4 p[4]; };
struct SceneInput {
    float3 position [[attribute(0)]]; float3 normal [[attribute(1)]];
    float4 r0 [[attribute(2)]]; float4 r1 [[attribute(3)]]; float4 r2 [[attribute(4)]];
    float4 a [[attribute(5)]]; float4 b [[attribute(6)]];
};
struct SceneOutput { float4 position [[position]]; float3 normal; float4 a; float4 b; float depth; };
vertex SceneOutput scene_vertex(SceneInput i [[stage_in]], constant Camera &c [[buffer(0)]]) {
    float4 local=float4(i.position,1), world=float4(dot(i.r0,local),dot(i.r1,local),dot(i.r2,local),1);
    float4 view=float4(dot(c.v[0],world),dot(c.v[1],world),dot(c.v[2],world),dot(c.v[3],world));
    float4 clip=float4(dot(c.p[0],view),dot(c.p[1],view),dot(c.p[2],view),dot(c.p[3],view));
    clip.z=(clip.z+clip.w)*0.5;
    SceneOutput o; o.position=clip; o.depth=-view.z; o.a=i.a; o.b=i.b;
    o.normal=float3(dot(cross(i.r1.xyz,i.r2.xyz),i.normal),dot(cross(i.r2.xyz,i.r0.xyz),i.normal),dot(cross(i.r0.xyz,i.r1.xyz),i.normal));
    return o;
}
fragment float4 color_fragment(SceneOutput i [[stage_in]]) {
    float shade=0.3+0.7*abs(dot(normalize(i.normal),normalize(float3(0.4,-0.5,0.8))));
    return float4(i.a.rgb*shade,i.a.a);
}
float4 pack_words(float2 f) {
    uint2 w=uint2(floor(f+0.5)); return float4(w.x&255,w.x>>8,w.y&255,w.y>>8)/255.0;
}
struct DataOutput { float4 id [[color(0)]]; float4 s0 [[color(1)]]; float4 s1 [[color(2)]]; float depth [[color(3)]]; };
fragment DataOutput data_fragment(SceneOutput i [[stage_in]]) { return {pack_words(i.a.xy),pack_words(i.a.zw),pack_words(i.b.xy),i.depth}; }
struct UiInput { float2 position [[attribute(0)]]; float2 uv [[attribute(1)]]; float4 color [[attribute(2)]]; };
struct UiOutput { float4 position [[position]]; float2 uv; float4 color; };
vertex UiOutput ui_vertex(UiInput i [[stage_in]], constant float4 &size [[buffer(0)]]) {
    return {float4(i.position.x/size.x*2-1,1-i.position.y/size.y*2,0,1),i.uv,i.color};
}
fragment float4 ui_fragment(UiOutput i [[stage_in]], texture2d<float> image [[texture(0)]], sampler sample_state [[sampler(0)]]) {
    return i.color*image.sample(sample_state,i.uv);
}
