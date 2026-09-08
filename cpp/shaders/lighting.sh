uniform vec4 u_options;
uniform vec4 u_cameraPosition, u_cameraDirection, u_ambient;
uniform vec4 u_headlightDiffuse, u_headlightSpecular;
uniform vec4 u_fog, u_fogColor, u_hazeColor, u_depthRange, u_lightCount, u_imageLight;
uniform vec4 u_lightPosition[100], u_lightDirection[100], u_lightDiffuse[100], u_lightSpecular[100], u_lightAttenuation[100];
SAMPLERCUBE(s_imageLight, 2);
#include "shadows.sh"
#include "color.sh"
vec3 lightColor(vec3 c) { return u_options.y > .5 ? linearToSrgb(c) : c; }
vec3 lightTerm(vec3 albedo, vec3 n, vec3 l, vec3 viewDir, vec3 diffuse, vec3 specularColor,
    vec3 specularMod, float specular, float shininess, float atten, float shadow) {
    float ndl = max(dot(n, l), 0.0);
    if (ndl <= 0 || atten <= 0) return vec3(0);
    vec3 h = normalize(l + viewDir);
    float spec = specular * pow(max(dot(n, h), 0.0), max(shininess * 128.0, 1e-3));
    diffuse = lightColor(diffuse);
    specularColor = lightColor(specularColor);
    if (u_options.y > .5) return atten * shadow * (ndl * diffuse * albedo + specularColor * spec * specularMod);
    return atten * shadow * ndl * (diffuse * albedo + specularColor * spec * specularMod);
}
vec3 shade(vec3 albedo, vec3 normal, vec3 world, vec3 material, vec3 texel, float viewDepth) {
    vec3 n = normalize(normal), viewDir = normalize(u_cameraPosition.xyz - world);
    vec3 ambient = u_options.y > .5 ? clamp(u_ambient.rgb, 0, 1) : srgbToLinear(clamp(u_ambient.rgb, 0, 1));
    vec3 color = ambient * albedo;
    vec3 specularMod = u_options.y > .5 ? texel : vec3(1);
    if (u_imageLight.x > 0) {
        vec3 r = reflect(-viewDir, n);
        vec3 diffuse = textureCubeLod(s_imageLight, vec3(n.x, n.z, -n.y), u_imageLight.y).rgb;
        vec3 specular = textureCubeLod(s_imageLight, vec3(r.x, r.z, -r.y), (1.0-clamp(material.z, 0, 1))*u_imageLight.y).rgb;
        color += u_imageLight.x * (lightColor(diffuse) * albedo + material.y * lightColor(specular) * specularMod);
    }
    for (int i = 0; i < 100; ++i) {
        if (float(i) >= u_lightCount.x) break;
        int type = int(u_lightPosition[i].w + .5);
        vec3 l;
        float atten = 1.0;
        if (type == 0) l = -normalize(u_lightDirection[i].xyz);
        else {
            vec3 toLight = u_lightPosition[i].xyz - world;
            float distance = length(toLight);
            l = toLight / max(distance, 1e-6);
            vec3 k = u_lightAttenuation[i].xyz;
            atten = 1.0 / max(k.x + k.y * distance + k.z * distance * distance, 1e-6);
            if (u_options.y < .5 && u_lightAttenuation[i].w > 0 && distance > u_lightAttenuation[i].w) atten = 0.0;
            if (type == 2) {
                float cd = dot(-l, normalize(u_lightDirection[i].xyz));
                atten *= cd < u_lightDirection[i].w ? 0.0 : pow(max(cd, 0.0), u_lightDiffuse[i].w);
            }
        }
        float shadow = 1.0;
        if (float(i) == u_shadowConfig.x) shadow = directionalShadow(world, n, viewDepth);
        if (type != 0) shadow *= localShadow(type, int(u_localSlots[i].x), world, n);
        color += lightTerm(albedo, n, l, viewDir, u_lightDiffuse[i].rgb, u_lightSpecular[i].rgb,
            specularMod, material.y, material.z, atten, shadow);
    }
    if (u_headlightDiffuse.w > .5) color += lightTerm(albedo, n, -normalize(u_cameraDirection.xyz), viewDir,
        u_headlightDiffuse.rgb, u_headlightSpecular.rgb, specularMod, material.y, material.z, 1.0, 1.0);
    return color + material.x * albedo;
}
