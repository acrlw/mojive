$input v_color0, v_meta0, v_meta1, v_normal, v_depth, v_texcoord0, v_world
#include <bgfx_shader.sh>
SAMPLER2D(s_image, 0);
uniform vec4 u_material;
void main()
{
    vec3 normal = normalize(v_normal);
    float diffuse = abs(dot(normal, normalize(vec3(0.4, -0.5, 0.8))));
    vec4 texel = texture2D(s_image, v_texcoord0);
    vec3 color = v_color0.rgb * texel.rgb;
    float light = 0.3 + 0.7 * diffuse;
    vec3 eye = mul(u_invView, vec4(0.0, 0.0, 0.0, 1.0)).xyz;
    vec3 halfway = normalize(normalize(eye - v_world) + normalize(vec3(0.4, -0.5, 0.8)));
    float specular = u_material.y * pow(max(dot(normal, halfway), 0.0), max(1.0, u_material.z * 128.0));
    vec3 shaded = color * (light + u_material.x) + vec3(specular);
    if (u_material.w > 0.5)
        shaded = pow(max(shaded, vec3(0.0)), vec3(1.0 / 2.2));
    gl_FragColor = vec4(shaded, v_color0.a * texel.a);
}
