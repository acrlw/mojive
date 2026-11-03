#include <bgfx_shader.sh>
SAMPLER2D(s_selectionMask,0);
uniform vec4 u_outlineSize;
uniform vec4 u_outlineColor;
float coverage(vec2 p) {
    if(any(lessThan(p,vec2_splat(0)))||any(greaterThanEqual(p,u_outlineSize.xy)))return 0.0;
    return texture2DLod(s_selectionMask,(p+.5)/u_outlineSize.xy,0).r;
}
bool borderRun(vec2 p,vec2 stepDirection,float toBorder) {
    if(toBorder>=3)return false;
    vec2 tangent=vec2(-stepDirection.y,stepDirection.x);
    for(int offset=-4;offset<=4;++offset) {
        vec2 delta=stepDirection*toBorder+tangent*float(offset);
        if(dot(delta,delta)>16)continue;
        float count=max(toBorder,abs(float(offset)));
        bool connected=true;
        for(int i=1;i<=4;++i) {
            if(float(i)>count)break;
            if(coverage(p+floor(delta*float(i)/count+vec2_splat(.5)))<=0){connected=false;break;}
        }
        if(connected)return true;
    }
    return false;
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
    float alpha=max(expanded-center,0.0);
    // Add the complementary inside/outside sample coverage at clipped edges.
    if(center>0 && (borderRun(p,vec2(-1,0),p.x)||borderRun(p,vec2(1,0),u_outlineSize.x-1-p.x)||borderRun(p,vec2(0,-1),p.y)||borderRun(p,vec2(0,1),u_outlineSize.y-1-p.y)))alpha+=center;
    alpha=max(alpha,center*u_outlineSize.z);
    if(alpha<=0)discard;
    gl_FragColor=vec4(u_outlineColor.rgb,u_outlineColor.a*alpha);
}
