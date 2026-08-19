#version 330 core
// Barycentric coordinates provide stable, pixel-width wireframe edges.

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

in VertexData {
    vec3 world;
    vec3 normal;
    vec2 uv;
    vec4 color;
    vec3 material;
    float reflect;
    float view_depth;
    float selected;
} v_in[];

out VertexData {
    vec3 world;
    vec3 normal;
    vec2 uv;
    vec4 color;
    vec3 material;
    float reflect;
    float view_depth;
    float selected;
} v;

out vec3 v_bary;

void main() {
    for (int i = 0; i < 3; ++i) {
        v.world = v_in[i].world;
        v.normal = v_in[i].normal;
        v.uv = v_in[i].uv;
        v.color = v_in[i].color;
        v.material = v_in[i].material;
        v.reflect = v_in[i].reflect;
        v.view_depth = v_in[i].view_depth;
        v.selected = v_in[i].selected;
        v_bary = vec3(i == 0 ? 1.0 : 0.0, i == 1 ? 1.0 : 0.0, i == 2 ? 1.0 : 0.0);
        gl_Position = gl_in[i].gl_Position;
        gl_ClipDistance[0] = gl_in[i].gl_ClipDistance[0];
        EmitVertex();
    }
    EndPrimitive();
}
