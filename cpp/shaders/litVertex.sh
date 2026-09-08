void main()
{
    vec4 position = vec4(a_position, 1.0);
    vec4 world = vec4(dot(i_data0, position), dot(i_data1, position), dot(i_data2, position), 1.0);
    gl_Position = mul(u_viewProj, world);
    v_world = world.xyz;
#ifdef WIREFRAME
    v_bary = a_texcoord1;
#endif
    v_depth = -mul(u_view, world).z;
    v_color0 = i_data3;
    v_litMaterial = i_data5;
    v_litIdentity = i_data7;
    if (i_data7.z > .5 && a_normal.z < .5) v_litIdentity.y = 0;
    float determinant = dot(i_data0.xyz, cross(i_data1.xyz, i_data2.xyz));
    determinant = abs(determinant)<1e-20 ? (determinant<0 ? -1e-20 : 1e-20) : determinant;
    v_normal = vec3(dot(cross(i_data1.xyz, i_data2.xyz), a_normal),
        dot(cross(i_data2.xyz, i_data0.xyz), a_normal),
        dot(cross(i_data0.xyz, i_data1.xyz), a_normal)) / determinant;
    if(dot(v_normal,v_normal)<1e-20)v_normal=vec3(0,0,1);
    if (i_data4.z > 1.5) {
        v_texcoord0 = a_position.xy * i_data4.xy - vec2(0.5);
    } else if (i_data4.z > .5) {
        vec3 extent = vec3(length(vec3(i_data0.x, i_data1.x, i_data2.x)),
            length(vec3(i_data0.y, i_data1.y, i_data2.y)), length(vec3(i_data0.z, i_data1.z, i_data2.z)));
        vec2 repeat = i_data4.xy / max(extent.xy, vec2(1e-7));
        vec3 axis = abs(a_normal);
        vec2 scale = i_data4.xy;
        if (axis.x >= axis.y && axis.x >= axis.z) scale = vec2(extent.y * repeat.x, extent.z * repeat.y);
        else if (axis.y >= axis.z) scale = vec2(extent.x * repeat.x, extent.z * repeat.y);
        v_texcoord0 = a_texcoord0 * scale;
    } else v_texcoord0 = a_texcoord0 * i_data4.xy + i_data4.zw;
    v_litCube = vec4(a_position * i_data6.xyz + vec3(0, 0, i_data6.w),
        dot(abs(i_data6.xyz), vec3(1)) > 0 ? 1.0 : 0.0);
}
