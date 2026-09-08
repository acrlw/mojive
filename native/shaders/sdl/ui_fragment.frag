#version 450
layout(location=0) in vec2 uv;
layout(location=1) in vec4 color;
layout(set=2,binding=0) uniform sampler2D image;
layout(location=0) out vec4 out_color;
void main() { out_color=color*texture(image,uv); }
