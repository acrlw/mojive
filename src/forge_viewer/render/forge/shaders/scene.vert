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
in vec4 in_texcoef;    // scale/offset; z=1 selects box face-axis mapping
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
    if (in_texcoef.z > 0.5) {
        vec3 extent = vec3(
            length(model[0].xyz),
            length(model[1].xyz),
            length(model[2].xyz)
        );
        vec2 repeat = in_texcoef.xy / max(extent.xy, vec2(1e-7));
        vec3 axis = abs(in_normal);
        vec2 scale;
        if (axis.x >= axis.y && axis.x >= axis.z) {
            scale = vec2(extent.y * repeat.x, extent.z * repeat.y);
        } else if (axis.y >= axis.z) {
            scale = vec2(extent.x * repeat.x, extent.z * repeat.y);
        } else {
            scale = in_texcoef.xy;
        }
        v.uv = in_uv * scale;
    } else {
        v.uv = in_uv * in_texcoef.xy + in_texcoef.zw;
    }
    v.color = in_color;
    v.material = in_material.xyz;
    v.reflect = in_material.w;
    if (v.reflect < 0.0 && mod(-v.reflect, 4.0) >= 2.0 && in_normal.z < 0.5) {
        v.reflect = 0.0;
    }
    v.view_depth = -(u_view * world).z;

    v.selected = (u_selected_id != 0u && in_object_id == u_selected_id) ? 1.0 : 0.0;

    gl_ClipDistance[0] = dot(u_clip_plane, vec4(world.xyz, 1.0));
    gl_Position = u_view_proj * world;
}
