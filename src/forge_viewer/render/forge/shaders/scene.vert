#version 330 core
in vec3 in_position;
in vec3 in_normal;
in vec2 in_uv;

in vec4 in_model0;
in vec4 in_model1;
in vec4 in_model2;
in vec4 in_model3;
in vec4 in_color;      // Linear RGBA
in vec4 in_material;   // (emission, specular, shininess, reflectance)
in vec4 in_texcoef;    // (u_scale, v_scale, u_offset, v_offset)
in uint in_object_id;

uniform mat4 u_view_proj;
uniform mat4 u_view;
uniform uint u_selected_id;

uniform vec4 u_clip_plane;

out VertexData {
    vec3 world;
    vec3 normal;
    vec2 uv;
    vec4 color;
    vec3 material;    // emission, specular, shininess
    float reflect;    // Planar reflection coefficient
    float view_depth; // View-space depth
    float selected;
} v;

void main() {
    mat4 model = mat4(in_model0, in_model1, in_model2, in_model3);
    vec4 world = model * vec4(in_position, 1.0);

    v.world = world.xyz;
    v.normal = transpose(inverse(mat3(model))) * in_normal;
    v.uv = in_uv * in_texcoef.xy + in_texcoef.zw;
    v.color = in_color;
    v.material = in_material.xyz;
    v.reflect = in_material.w;
    v.view_depth = -(u_view * world).z;

    v.selected = (u_selected_id != 0u && in_object_id == u_selected_id) ? 1.0 : 0.0;

    gl_ClipDistance[0] = dot(u_clip_plane, vec4(world.xyz, 1.0));
    gl_Position = u_view_proj * world;
}
