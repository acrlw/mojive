void main()
{
    if (u_reflectionConfig.x > .5 && dot(vec4(v_world, 1), u_reflectionPlane) < 0) discard;
    vec4 texel = v_litCube.w > .5 ? textureCube(s_cube, v_litCube.xyz) : texture2DBias(s_image, v_texcoord0, 1.0);
    vec3 surface = v_color0.rgb;
    if (u_options.y > .5) { surface = gammaEncode(surface); texel.rgb = linearToSrgb(texel.rgb); }
    vec3 albedo = surface * texel.rgb;
    float alpha = v_color0.a * texel.a;
    vec3 material = v_litMaterial.xyz;
    if (v_litIdentity.x > .5) { albedo = mix(albedo, vec3(1, .82, .45), .35); material.x += .16; }
    int mode = int(u_options.w + .5);
    if (mode == 1) { gl_FragColor = vec4(u_options.y > .5 ? albedo : gammaEncode(albedo), alpha); return; }
    if (mode == 2) { gl_FragColor = vec4(normalize(v_normal) * .5 + .5, alpha); return; }
    if (mode == 3) { gl_FragColor = vec4(vec3(1.0 - clamp((v_depth-u_depthRange.x)/max(u_depthRange.y-u_depthRange.x, 1e-6), 0, 1)), alpha); return; }
    if (mode == 4) { gl_FragColor = vec4(vec3(1.0/16.0), 0); return; }
    vec3 lit = shade(albedo, v_normal, v_world, material, texel.rgb, v_depth);
    if (u_options.y > .5) lit = clamp(lit, vec3(0), texel.rgb);
    if (u_reflectionConfig.x < .5 && v_litIdentity.y > .5) lit += v_litMaterial.w * reflectedColor(v_litIdentity.y, gl_FragCoord.xy);
    float fog = u_fog.z * smoothstep(u_fog.x, max(u_fog.y, u_fog.x+1e-6), v_depth);
    float haze = 1.0 - exp(-max(u_fog.w, 0.0) * max(v_depth, 0.0));
    lit = mix(lit, u_options.y > .5 ? u_fogColor.rgb : srgbToLinear(u_fogColor.rgb), fog);
    lit = mix(lit, u_options.y > .5 ? u_hazeColor.rgb : srgbToLinear(u_hazeColor.rgb), haze);
    vec3 rgb=u_reflectionConfig.x>.5?lit:u_options.y>.5?clamp(lit,0,1):finishColor(lit);
#ifdef WIREFRAME
    float distanceToEdge=min(v_bary.x,min(v_bary.y,v_bary.z));
    float width=max(fwidth(distanceToEdge),1e-6)*1.2;
    rgb=mix(vec3(.1,.1,.12),rgb,smoothstep(0.0,width,distanceToEdge));
#endif
    gl_FragColor=vec4(rgb,alpha);
}
