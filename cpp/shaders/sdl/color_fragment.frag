#version 450
layout(location=0) in vec3 normal;
layout(location=1) in vec4 a;
layout(location=0) out vec4 color;
void main() {
    float shade=0.3+0.7*abs(dot(normalize(normal),normalize(vec3(0.4,-0.5,0.8))));
    color=vec4(a.rgb*shade,a.a);
}
