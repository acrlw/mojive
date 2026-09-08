#version 450
layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
layout(location=2) in vec4 r0;
layout(location=3) in vec4 r1;
layout(location=4) in vec4 r2;
layout(location=5) in vec4 a;
layout(location=6) in vec4 b;
layout(set=1,binding=0,std140) uniform Camera { vec4 v[4]; vec4 p[4]; } camera;
layout(location=0) out vec3 out_normal;
layout(location=1) out vec4 out_a;
layout(location=2) out vec4 out_b;
layout(location=3) out float out_depth;
void main() {
    vec4 local=vec4(position,1), world=vec4(dot(r0,local),dot(r1,local),dot(r2,local),1);
    vec4 view=vec4(dot(camera.v[0],world),dot(camera.v[1],world),dot(camera.v[2],world),dot(camera.v[3],world));
    vec4 clip=vec4(dot(camera.p[0],view),dot(camera.p[1],view),dot(camera.p[2],view),dot(camera.p[3],view));
    clip.z=(clip.z+clip.w)*0.5;
    gl_Position=clip; out_depth=-view.z; out_a=a; out_b=b;
    out_normal=vec3(dot(cross(r1.xyz,r2.xyz),normal),dot(cross(r2.xyz,r0.xyz),normal),dot(cross(r0.xyz,r1.xyz),normal));
}
