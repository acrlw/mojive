#include <bgfx_shader.sh>
SAMPLER2D(s_selectionMask,0);
uniform vec4 u_outlineSize, u_outlineColor;
float coverage(vec2 p) {
    if(any(lessThan(p,vec2(0)))||any(greaterThanEqual(p,u_outlineSize.xy)))return 0.0;
    return texture2DLod(s_selectionMask,(p+.5)/u_outlineSize.xy,0).r;
}
bool borderRun(vec2 p,vec2 stepDirection,float toBorder) {
    if(toBorder>=3)return false;
    for(int i=1;i<=3;++i) {
        if(float(i)>toBorder)break;
        if(coverage(p+stepDirection*float(i))<.5)return false;
    }
    return true;
}
void main() {
    vec2 p=floor(gl_FragCoord.xy);
    float expanded=0.0,center=coverage(p);
    for(int y=-4;y<=4;++y)for(int x=-4;x<=4;++x) {
        float r2=float(x*x+y*y);
        if(r2>16)continue;
        float kernel=1.0-smoothstep(9.0,16.0,r2);
        expanded=max(expanded,coverage(p+vec2(float(x),float(y)))*kernel);
    }
    float alpha=max(max(expanded-center,0.0),center*u_outlineSize.z);
    if(center>0 && (borderRun(p,vec2(-1,0),p.x)||borderRun(p,vec2(1,0),u_outlineSize.x-1-p.x)||borderRun(p,vec2(0,-1),p.y)||borderRun(p,vec2(0,1),u_outlineSize.y-1-p.y)))alpha=max(alpha,center);
    if(alpha<=0)discard;
    gl_FragColor=vec4(u_outlineColor.rgb,u_outlineColor.a*alpha);
}
