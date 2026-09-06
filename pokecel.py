#!/usr/bin/env python3
"""Small real-time cel-shading viewer for character-model experiments."""
from __future__ import annotations

import argparse, math, sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import moderngl
import numpy as np
import pygame
import trimesh
from PIL import Image

VS = r'''#version 330
uniform mat4 mvp, model;
uniform mat3 normal_matrix;
uniform bool flat_normals;
uniform float outline_width;
in vec3 in_pos, in_normal, in_flat_normal;
in vec2 in_uv;
in vec4 in_color;
out vec3 world_pos, normal;
out vec2 uv;
out vec4 color;
void main(){
    vec3 n = flat_normals ? in_flat_normal : in_normal;
    vec3 p = in_pos + n * outline_width;
    vec4 w = model * vec4(p,1.0);
    world_pos=w.xyz; normal=normalize(normal_matrix*n); uv=in_uv; color=in_color;
    gl_Position=mvp*vec4(p,1.0);
}'''

FS = r'''#version 330
uniform sampler2D tex;
uniform bool has_texture, outline_pass;
uniform int mode;
uniform vec3 light_dir, camera_pos;
in vec3 world_pos, normal;
in vec2 uv;
in vec4 color;
out vec4 frag;
void main(){
    if(outline_pass){frag=vec4(0.055,0.045,0.065,1.0); return;}
    vec4 base=(has_texture?texture(tex,uv):vec4(1.0))*color;
    if(base.a<0.08) discard;
    vec3 n=normalize(normal), l=normalize(light_dir);
    float ndl=max(dot(n,l),0.0);
    vec3 c;
    if(mode==0){
        float d=0.24+0.76*ndl;
        vec3 v=normalize(camera_pos-world_pos), h=normalize(l+v);
        c=base.rgb*d+pow(max(dot(n,h),0.0),40.0)*0.10;
    }else if(mode==1){
        float b=ndl>0.62?1.08:(ndl>0.28?0.82:0.55);
        c=base.rgb*b;
    }else{
        float lit=smoothstep(0.30,0.34,ndl), hi=smoothstep(0.78,0.82,ndl);
        c=mix(base.rgb*vec3(0.57,0.66,0.82),base.rgb,lit);
        c=mix(c,min(base.rgb*1.12,vec3(1.0)),hi*0.45);
        vec3 v=normalize(camera_pos-world_pos);
        float rim=smoothstep(0.72,0.92,1.0-max(dot(n,v),0.0))*smoothstep(0.05,0.45,ndl);
        c=min(c+rim*vec3(0.10,0.10,0.08),vec3(1.0));
    }
    frag=vec4(c,base.a);
}'''

@dataclass
class Batch:
    vao: object
    tex: object
    textured: bool


def unit(v):
    n=float(np.linalg.norm(v)); return v if n<1e-8 else v/n


def perspective(fov, aspect, near=.05, far=100.):
    f=1/math.tan(fov/2); m=np.zeros((4,4),np.float32)
    m[0,0]=f/max(aspect,1e-6); m[1,1]=f
    m[2,2]=(far+near)/(near-far); m[2,3]=2*far*near/(near-far); m[3,2]=-1
    return m


def look_at(eye):
    f=unit(-eye); s=unit(np.cross(f,np.array([0,1,0],np.float32))); u=np.cross(s,f)
    m=np.eye(4,dtype=np.float32); m[0,:3]=s; m[1,:3]=u; m[2,:3]=-f
    m[0,3]=-s@eye; m[1,3]=-u@eye; m[2,3]=f@eye
    return m


def rot_x(a):
    c,s=math.cos(a),math.sin(a)
    return np.array([[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]],np.float32)


def rot_y(a):
    c,s=math.cos(a),math.sin(a)
    return np.array([[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]],np.float32)


def rgba(mesh):
    mat=getattr(mesh.visual,'material',None)
    for name in ('baseColorFactor','diffuse','main_color'):
        v=getattr(mat,name,None) if mat is not None else None
        if v is not None:
            a=np.asarray(v,np.float32).reshape(-1)
            if len(a)>=3:
                if a.max(initial=0)>1: a/=255
                if len(a)==3: a=np.r_[a,1]
                return np.clip(a[:4],0,1)
    return np.ones(4,np.float32)


def image_for(mesh):
    mat=getattr(mesh.visual,'material',None)
    if mat is None: return None
    for name in ('baseColorTexture','image'):
        im=getattr(mat,name,None)
        if isinstance(im,Image.Image): return im
        if im is not None:
            try: return Image.fromarray(np.asarray(im))
            except Exception: pass
    return None


def pack(mesh):
    faces=np.asarray(mesh.faces,np.int64)
    if faces.ndim!=2 or faces.shape[1]!=3: raise ValueError('mesh is not triangular')
    idx=faces.reshape(-1); p=np.asarray(mesh.vertices,np.float32)[idx]
    n=np.asarray(mesh.vertex_normals,np.float32)[idx]
    tri=p.reshape(-1,3,3); fn=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    fn/=np.maximum(np.linalg.norm(fn,axis=1,keepdims=True),1e-12); fn=np.repeat(fn,3,0)
    uv=np.zeros((len(p),2),np.float32); src=getattr(mesh.visual,'uv',None)
    if src is not None and len(src)==len(mesh.vertices): uv=np.asarray(src,np.float32)[idx]
    col=np.repeat(rgba(mesh)[None,:],len(p),0)
    vc=getattr(mesh.visual,'vertex_colors',None); fc=getattr(mesh.visual,'face_colors',None)
    if vc is not None and len(vc)==len(mesh.vertices):
        x=np.asarray(vc,np.float32); x=np.c_[x,np.full(len(x),255)] if x.shape[1]==3 else x; col=x[idx,:4]/255
    elif fc is not None and len(fc)==len(faces):
        x=np.asarray(fc,np.float32); x=np.c_[x,np.full(len(x),255)] if x.shape[1]==3 else x; col=np.repeat(x[:,:4]/255,3,0)
    return np.ascontiguousarray(np.c_[p,n,fn,uv,col],dtype=np.float32), image_for(mesh)


def load_meshes(path):
    if path is None:
        m=trimesh.creation.icosphere(subdivisions=4)
        y=np.asarray(m.vertices)[:,1]; t=((y-y.min())/max(np.ptp(y),1e-6))[:,None]
        lo=np.array([.20,.42,.92,1]); hi=np.array([1,.78,.20,1])
        m.visual.vertex_colors=np.clip((lo*(1-t)+hi*t)*255,0,255).astype(np.uint8)
        return [m]
    obj=trimesh.load(path,force='scene',process=False)
    if isinstance(obj,trimesh.Trimesh): return [obj]
    out=[]
    for node in obj.graph.nodes_geometry:
        xf,name=obj.graph[node]; g=obj.geometry.get(name)
        if isinstance(g,trimesh.Trimesh):
            g=g.copy(); g.apply_transform(xf); out.append(g)
    if not out: raise ValueError('no triangle meshes found')
    return out


def batches(ctx,prog,meshes):
    items=[]
    for m in meshes:
        try: items.append(pack(m))
        except Exception as e: print('Skipping mesh:',e,file=sys.stderr)
    if not items: raise ValueError('no renderable meshes')
    allp=np.concatenate([x[0][:,:3] for x in items]); lo=allp.min(0); hi=allp.max(0)
    center=(lo+hi)/2; scale=2.2/max(float(np.max(hi-lo)),1e-6)
    out=[]
    for data,im in items:
        data[:,:3]=(data[:,:3]-center)*scale
        vbo=ctx.buffer(data.tobytes())
        vao=ctx.vertex_array(prog,[(vbo,'3f 3f 3f 2f 4f','in_pos','in_normal','in_flat_normal','in_uv','in_color')])
        if im is None:
            tex=ctx.texture((1,1),4,b'\xff\xff\xff\xff'); tex.filter=(moderngl.NEAREST,moderngl.NEAREST); textured=False
        else:
            im=im.convert('RGBA').transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            tex=ctx.texture(im.size,4,im.tobytes()); tex.build_mipmaps(); tex.filter=(moderngl.LINEAR_MIPMAP_LINEAR,moderngl.LINEAR); textured=True
        out.append(Batch(vao,tex,textured))
    return out


def mat(uniform,m): uniform.write(np.asarray(m.T,np.float32).tobytes())


def save_screen(ctx,w,h):
    raw=ctx.screen.read(components=3,alignment=1)
    im=Image.frombytes('RGB',(w,h),raw).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    d=Path('screenshots'); d.mkdir(exist_ok=True)
    p=d/f"pokecel_{datetime.now():%Y%m%d_%H%M%S}.png"; im.save(p); print('Saved',p)


def run(path,w,h):
    pygame.init()
    for flag,val in ((pygame.GL_CONTEXT_MAJOR_VERSION,3),(pygame.GL_CONTEXT_MINOR_VERSION,3),(pygame.GL_CONTEXT_PROFILE_MASK,pygame.GL_CONTEXT_PROFILE_CORE),(pygame.GL_DEPTH_SIZE,24)):
        pygame.display.gl_set_attribute(flag,val)
    pygame.display.set_mode((w,h),pygame.OPENGL|pygame.DOUBLEBUF|pygame.RESIZABLE)
    ctx=moderngl.create_context(require=330); ctx.enable(moderngl.DEPTH_TEST|moderngl.CULL_FACE|moderngl.BLEND)
    ctx.blend_func=(moderngl.SRC_ALPHA,moderngl.ONE_MINUS_SRC_ALPHA)
    prog=ctx.program(vertex_shader=VS,fragment_shader=FS); prog['tex'].value=0
    draw=batches(ctx,prog,load_meshes(path)); name=path.name if path else 'demo mesh'
    yaw=pitch=0.; dist=4.2; mode=2; outline=True; flat=False; drag=False; last=(0,0); clock=pygame.time.Clock()
    print('1 smooth | 2 toon | 3 anime | O outline | F flat normals | drag orbit | wheel zoom | P screenshot | Esc quit')
    running=True
    while running:
        for e in pygame.event.get():
            if e.type==pygame.QUIT: running=False
            elif e.type==pygame.KEYDOWN:
                if e.key==pygame.K_ESCAPE: running=False
                elif e.key in (pygame.K_1,pygame.K_2,pygame.K_3): mode=e.key-pygame.K_1
                elif e.key==pygame.K_o: outline=not outline
                elif e.key==pygame.K_f: flat=not flat
                elif e.key==pygame.K_r: yaw=pitch=0.; dist=4.2
                elif e.key==pygame.K_p: save_screen(ctx,w,h)
            elif e.type==pygame.MOUSEBUTTONDOWN and e.button==1: drag=True; last=e.pos
            elif e.type==pygame.MOUSEBUTTONUP and e.button==1: drag=False
            elif e.type==pygame.MOUSEMOTION and drag:
                yaw+=(e.pos[0]-last[0])*.008; pitch=float(np.clip(pitch+(e.pos[1]-last[1])*.008,-1.45,1.45)); last=e.pos
            elif e.type==pygame.MOUSEWHEEL: dist=float(np.clip(dist*(.9**e.y),2.1,12))
            elif e.type==pygame.VIDEORESIZE: w,h=max(e.w,1),max(e.h,1)
        clock.tick(120); pygame.display.set_caption(f'PokeCel — {name} — {("smooth","toon","anime")[mode]} | outline {outline} | flat normals {flat}')
        ctx.viewport=(0,0,w,h); ctx.clear(.93,.94,.96,1,depth=1)
        eye=np.array([0,0,dist],np.float32); model=rot_x(pitch)@rot_y(yaw); mvp=perspective(math.radians(42),w/max(h,1))@look_at(eye)@model
        mat(prog['mvp'],mvp); mat(prog['model'],model); prog['normal_matrix'].write(np.asarray(np.linalg.inv(model[:3,:3]),np.float32).tobytes())
        prog['camera_pos'].value=tuple(float(x) for x in eye); prog['light_dir'].value=tuple(unit(np.array([-.35,.70,.62],np.float32)))
        prog['mode'].value=mode; prog['flat_normals'].value=flat
        if outline:
            prog['outline_pass'].value=True; prog['outline_width'].value=.018; ctx.cull_face='front'
            for b in draw: b.tex.use(0); prog['has_texture'].value=b.textured; b.vao.render(moderngl.TRIANGLES)
        prog['outline_pass'].value=False; prog['outline_width'].value=0.; ctx.cull_face='back'
        for b in draw: b.tex.use(0); prog['has_texture'].value=b.textured; b.vao.render(moderngl.TRIANGLES)
        pygame.display.flip()
    pygame.quit()


def main():
    ap=argparse.ArgumentParser(description='Minimal cel-shading model viewer')
    ap.add_argument('model',nargs='?',type=Path,help='OBJ/GLB/GLTF/DAE/PLY/STL model')
    ap.add_argument('--width',type=int,default=1100); ap.add_argument('--height',type=int,default=800); a=ap.parse_args()
    if a.model and not a.model.exists(): print('Model not found:',a.model,file=sys.stderr); return 2
    try: run(a.model,max(a.width,320),max(a.height,240))
    except Exception as e:
        print('PokeCel failed:',e,file=sys.stderr); print('For FBX assets, use the DAE or OBJ variant instead.',file=sys.stderr); return 1
    return 0

if __name__=='__main__': raise SystemExit(main())
