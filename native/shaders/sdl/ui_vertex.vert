#version 450
layout(location=0) in vec2 position;
layout(location=1) in vec2 uv;
layout(location=2) in vec4 color;
layout(set=1,binding=0,std140) uniform Size { vec4 size; } viewport;
layout(location=0) out vec2 out_uv;
layout(location=1) out vec4 out_color;
void main() {
    gl_Position=vec4(position.x/viewport.size.x*2-1,1-position.y/viewport.size.y*2,0,1);
    out_uv=uv;out_color=color;
}
