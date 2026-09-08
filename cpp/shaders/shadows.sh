SAMPLER2DSHADOW(s_shadowAtlas, 3);
SAMPLER2DARRAY(s_localShadow, 4);
uniform vec4 u_shadowConfig, u_shadowSplits, u_shadowTexels;
uniform mat4 u_shadowMatrix[3], u_localMatrix[8];
uniform vec4 u_localPosition[8], u_localParams[8], u_localSlots[100];
float directionalCompare(vec2 uv,float reference,vec2 lo,vec2 hi) {
    return shadow2D(s_shadowAtlas,vec3(clamp(uv,lo,hi),reference));
}
float directionalShadow(vec3 world,vec3 normal,float viewDepth) {
    int cascade=2;
    for(int i=0;i<3;++i) if(viewDepth<u_shadowSplits[i]) { cascade=i; break; }
    vec3 p=vec3(0); bool inside=false;
    for(;cascade<3;++cascade) {
        vec4 clip=mul(u_shadowMatrix[cascade],vec4(world,1));
        p=clip.xyz/clip.w*.5+.5;
        if(all(greaterThanEqual(p,vec3(0)))&&all(lessThanEqual(p,vec3(1)))) {inside=true;break;}
    }
    if(!inside) return 1.0;
    // bgfx compiles matrix indexing through HLSL row semantics on Metal/Vulkan.
    // Multiply a basis vector to extract the depth row consistently on every backend.
    vec3 axis=mul(vec4(0,0,1,0),u_shadowMatrix[cascade]).xyz;
    float ndl=dot(normalize(normal),-normalize(axis));
    if(ndl<=0) return 0.0;
    float slope=sqrt(max(1.0-ndl*ndl,0.0))/max(ndl,.15);
    float bias=u_shadowTexels[cascade]*(1.0+2.5*slope)*.5*length(axis);
    if(u_shadowConfig.w>.5) p.y=1.0-p.y;
    vec2 origin=vec2(mod(float(cascade),2.0),floor(float(cascade)/2.0))*.5;
    vec2 uv=origin+p.xy*.5;
    vec2 lo=origin+vec2(.5/4096.0),hi=origin+vec2(.5-.5/4096.0);
    float ref=p.z-bias;
    if(u_shadowConfig.z<.5) return directionalCompare(uv,ref,lo,hi);
    float offset=(u_shadowConfig.z<1.5?.75:1.25)/4096.0;
    float lit=u_shadowConfig.z<1.5?0.0:directionalCompare(uv,ref,lo,hi);
    for(int y=-1;y<=1;y+=2) for(int x=-1;x<=1;x+=2)
        lit+=directionalCompare(uv+vec2(float(x),float(y))*offset,ref,lo,hi);
    return lit*(u_shadowConfig.z<1.5?.25:.2);
}
float localSample(vec2 uv,int layer) {
    if(u_shadowConfig.w>.5) uv.y=1.0-uv.y;
    return texture2DArrayLod(s_localShadow,vec3(uv,float(layer)),0.0).r;
}
float bilinearLocal(vec2 uv,int layer,float reference) {
    vec2 dims=vec2(u_localParams[0].w);
    vec2 t=uv*dims-.5,base=floor(t),f=fract(t);
    float a=step(reference,localSample((base+.5)/dims,layer));
    float b=step(reference,localSample((base+vec2(1.5,.5))/dims,layer));
    float c=step(reference,localSample((base+vec2(.5,1.5))/dims,layer));
    float d=step(reference,localSample((base+1.5)/dims,layer));
    return mix(mix(a,b,f.x),mix(c,d,f.x),f.y);
}
float filteredLocal(vec2 uv,int layer,float reference) {
    if(u_shadowConfig.z<.5) return bilinearLocal(uv,layer,reference);
    vec2 delta=vec2(u_shadowConfig.z<1.5?.75:1.25)/vec2(u_localParams[0].w);
    float lit=u_shadowConfig.z<1.5?0.0:bilinearLocal(uv,layer,reference);
    for(int y=-1;y<=1;y+=2) for(int x=-1;x<=1;x+=2)
        lit+=bilinearLocal(uv+vec2(float(x),float(y))*delta,layer,reference);
    return lit*(u_shadowConfig.z<1.5?.25:.2);
}
float localBias(int slot,float distance,vec3 normal,vec3 l) {
    float ndl=max(dot(normalize(normal),l),.15);
    float slope=sqrt(max(1.0-ndl*ndl,0.0))/ndl;
    return distance*(u_localParams[slot].x*(1.0+2.5*slope)+1.0/1024.0);
}
vec3 pointLayerUv(int slot,vec3 d) {
    vec3 a=abs(d);float face,ma;vec2 sc;
    if(a.x>=a.y&&a.x>=a.z){ma=a.x;if(d.x>0){face=0;sc=vec2(-d.z,-d.y);}else{face=1;sc=vec2(d.z,-d.y);}}
    else if(a.y>=a.z){ma=a.y;if(d.y>0){face=2;sc=vec2(d.x,d.z);}else{face=3;sc=vec2(d.x,-d.z);}}
    else{ma=a.z;if(d.z>0){face=4;sc=vec2(d.x,-d.y);}else{face=5;sc=vec2(-d.x,-d.y);}}
    return vec3(sc/max(ma,1e-6)*.5+.5,u_localParams[slot].z+face);
}
float localShadow(int type,int slot,vec3 world,vec3 normal) {
    if(slot<0||float(slot)>=u_shadowConfig.y) return 1.0;
    vec3 ray=world-u_localPosition[slot].xyz;float distance=length(ray);
    if(u_localPosition[slot].w>0&&distance>u_localPosition[slot].w) return 1.0;
    vec3 direction=ray/max(distance,1e-6);
    float bias=localBias(slot,distance,normal,-direction);
    if(type==2) {
        vec4 clip=mul(u_localMatrix[slot],vec4(world,1));if(clip.w<=0)return 1.0;
        vec3 p=clip.xyz/clip.w*.5+.5;
        if(any(lessThan(p,vec3(0)))||any(greaterThan(p,vec3(1))))return 1.0;
        return filteredLocal(p.xy,int(u_localParams[slot].z+.5),distance-bias);
    }
    vec3 seed=abs(direction.z)<.9?vec3(0,0,1):vec3(0,1,0);
    vec3 right=normalize(cross(direction,seed)),up=cross(right,direction);
    bool area=u_localParams[slot].y>0;
    int radius=u_shadowConfig.z<.5?(area?1:0):u_shadowConfig.z<1.5?(area?2:1):(area?3:2);
    float angle=max(u_localParams[slot].x,u_localParams[slot].y/max(distance*float(max(radius,1)),1e-6));
    float lit=0,taps=0;
    for(int y=-3;y<=3;++y)for(int x=-3;x<=3;++x){
        if(abs(x)>radius||abs(y)>radius)continue;
        vec3 p=pointLayerUv(slot,direction+(right*float(x)+up*float(y))*angle);
        lit+=step(distance-bias,localSample(p.xy,int(p.z+.5)));taps+=1;
    }
    return lit/taps;
}
