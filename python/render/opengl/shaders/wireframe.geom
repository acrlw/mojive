#version 330 core
// Barycentric coordinates provide stable, pixel-width wireframe edges.

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

in WireData { vec4 color; } wire_in[];
out WireData { vec4 color; } wire;

out vec3 v_bary;

void main() {
    for (int i = 0; i < 3; ++i) {
        wire.color = wire_in[i].color;
        v_bary = vec3(i == 0 ? 1.0 : 0.0, i == 1 ? 1.0 : 0.0, i == 2 ? 1.0 : 0.0);
        gl_Position = gl_in[i].gl_Position;
        gl_ClipDistance[0] = gl_in[i].gl_ClipDistance[0];
        EmitVertex();
    }
    EndPrimitive();
}
