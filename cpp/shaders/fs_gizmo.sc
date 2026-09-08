$input v_world, v_normal, v_litCube
#include <bgfx_shader.sh>
uniform vec4 u_gizmoParams, u_gizmoColor;
void main() {
    if(u_gizmoParams.x>0 && length(v_world)<u_gizmoParams.x)discard;
    float facing=abs(dot(normalize(v_normal),normalize(-v_litCube.xyz)));
    gl_FragColor=vec4(u_gizmoColor.rgb*(.72+.28*facing),u_gizmoColor.a);
}
