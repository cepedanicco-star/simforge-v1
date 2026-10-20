"""Scenario-only offsets and measured geometry/contact preflight; no policy tuning."""
import json
from pathlib import Path

import numpy as np

CUP='/World/Environment/Source/cup_V1'
CABINET='/World/Environment/Source/SM_AI_vol8_04_kitchen_furniture_01'


def apply_offsets(stage,dx=0.,dy=0.):
    from pxr import Gf,UsdGeom
    prim=stage.GetPrimAtPath(CUP);cache=UsdGeom.XformCache(0)
    old=cache.GetLocalToWorldTransform(prim)
    matrix=Gf.Matrix4d(old)
    matrix.SetTranslateOnly(old.ExtractTranslation()+Gf.Vec3d(float(dx),float(dy),0.))
    if dx or dy:
        local=matrix*cache.GetLocalToWorldTransform(prim.GetParent()).GetInverse()
        op=UsdGeom.Xformable(prim).MakeMatrixXform();op.Set(local);op.Set(local,0)
    return dict(cup_offset_x_m=float(dx),cup_offset_y_m=float(dy),
        before_position_m=list(old.ExtractTranslation()),after_position_m=list(matrix.ExtractTranslation()),
        authored_offset=bool(dx or dy))


def classify_initial_contacts(events):
    return 'init_infeasible_v2' if any(x.get('reports',0)>0 for x in events) else 'no_hand_cup_contact_in_original_setup'


def start_y(original,offset):
    return original+float(offset) if offset else original


def triangles(prim,relative_to=None):
    """Authored collider mesh triangles, including instance-proxy descendants."""
    from pxr import Usd,UsdGeom,UsdPhysics
    cache=UsdGeom.XformCache(0);result=[]
    for child in Usd.PrimRange(prim,Usd.TraverseInstanceProxies()):
        if not (child.IsA(UsdGeom.Mesh) or child.IsA(UsdGeom.Cube)):continue
        ancestor=child;colliding=False
        while ancestor and ancestor!=prim.GetParent():
            colliding=colliding or ancestor.HasAPI(UsdPhysics.CollisionAPI)
            ancestor=ancestor.GetParent()
        if not colliding:continue
        if child.IsA(UsdGeom.Cube):
            size=float(UsdGeom.Cube(child).GetSizeAttr().Get(0))/2
            points=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],float)*size
            indices=np.array([0,1,2,3,4,7,6,5,0,4,5,1,1,5,6,2,2,6,7,3,3,7,4,0]);counts=[4]*6
        else:
            mesh=UsdGeom.Mesh(child);points=np.asarray(mesh.GetPointsAttr().Get(0),dtype=float)
            indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get(0));counts=mesh.GetFaceVertexCountsAttr().Get(0)
        transform=cache.GetLocalToWorldTransform(child)
        if relative_to is not None:transform=transform*cache.GetLocalToWorldTransform(relative_to).GetInverse()
        points=(np.c_[points,np.ones(len(points))]@np.asarray(transform))[:,:3]
        start=0
        for count in counts:
            face=indices[start:start+count];start+=count
            result.extend(points[[face[0],face[i],face[i+1]]] for i in range(1,count-1))
    if not result:raise ValueError('No authored collider triangles at '+str(prim.GetPath()))
    return np.asarray(result)


def _point_triangle_min(points,tris):
    minimum=float('inf')
    a,b,c=tris[:,0],tris[:,1],tris[:,2];ab=b-a;ac=c-a
    normal=np.cross(ab,ac);n2=np.einsum('ij,ij->i',normal,normal)
    aa=np.einsum('ij,ij->i',ab,ab);bb=np.einsum('ij,ij->i',ab,ac);cc=np.einsum('ij,ij->i',ac,ac)
    den=aa*cc-bb*bb
    for batch in np.array_split(points,max(1,int(np.ceil(len(points)/24)))):
        v=batch[:,None,:]-a
        d=np.einsum('nmi,mi->nm',v,ab);e=np.einsum('nmi,mi->nm',v,ac)
        u=(cc*d-bb*e)/np.maximum(den,1e-30);w=(aa*e-bb*d)/np.maximum(den,1e-30)
        inside=(u>=0)&(w>=0)&(u+w<=1)&(den>1e-24)
        sq=np.einsum('nmi,mi->nm',v,normal)**2/np.maximum(n2,1e-30)
        sq=np.where(inside,sq,np.inf)
        for start,end in ((a,b),(b,c),(c,a)):
            edge=end-start;pv=batch[:,None,:]-start
            t=np.clip(np.einsum('nmi,mi->nm',pv,edge)/np.maximum(np.einsum('mi,mi->m',edge,edge),1e-30),0,1)
            diff=pv-t[...,None]*edge
            sq=np.minimum(sq,np.einsum('nmi,nmi->nm',diff,diff))
        minimum=min(minimum,float(sq.min()))
    return np.sqrt(max(0.,minimum))


def surface_distance(a,b):
    """Exact triangle-surface minimum incl. interior edge pairs/intersections.

    This measures authored collision geometry, not a signed penetration depth
    or a replacement for native cooked-collider contact observations.
    """
    a=np.asarray(a,float);b=np.asarray(b,float)
    from scipy.spatial import cKDTree
    # Fast vertex upper bound; triangle AABB broadphase preserves all pairs
    # capable of improving it, including edge-only minima.
    pa=np.unique(a.reshape(-1,3),axis=0);pb=np.unique(b.reshape(-1,3),axis=0)
    best=float(cKDTree(pb).query(pa)[0].min())
    blo=b.min(axis=(0,1));bhi=b.max(axis=(0,1))
    delta=np.maximum(np.maximum(blo-a.max(axis=1),a.min(axis=1)-bhi),0.)
    a=a[np.linalg.norm(delta,axis=1)<=best+1e-9]
    low=a.min(axis=(0,1));high=a.max(axis=(0,1))
    delta=np.maximum(np.maximum(low-b.max(axis=1),b.min(axis=1)-high),0.)
    b=b[np.linalg.norm(delta,axis=1)<=best+1e-9]
    assert len(b)
    best=min(best,_point_triangle_min(np.unique(a.reshape(-1,3),axis=0),b))
    if best<1e-9:return float(best)
    best=min(best,_point_triangle_min(np.unique(b.reshape(-1,3),axis=0),a))
    ea=np.concatenate([a[:,[0,1]],a[:,[1,2]],a[:,[2,0]]])
    eb=np.concatenate([b[:,[0,1]],b[:,[1,2]],b[:,[2,0]]])
    for batch in np.array_split(ea,max(1,int(np.ceil(len(ea)/24)))):
        # Discard edge pairs whose bounding boxes already exceed the best
        # distance. This matters for the dense hand meshes (~43k triangles).
        delta=np.maximum(np.maximum(batch.min(axis=1)[:,None,:]-eb.max(axis=1),eb.min(axis=1)-batch.max(axis=1)[:,None,:]),0.)
        ix,iy=np.where(np.linalg.norm(delta,axis=-1)<=best+1e-10)
        if not len(ix):continue
        p=batch[ix,0];v=batch[ix,1]-p
        q=eb[iy,0];w=eb[iy,1]-q;r=p-q
        aa=np.sum(v*v,axis=-1);bb=np.sum(v*w,axis=-1);cc=np.sum(w*w,axis=-1)
        dd=np.sum(v*r,axis=-1);ee=np.sum(w*r,axis=-1);den=aa*cc-bb*bb
        s=(bb*ee-cc*dd)/np.maximum(den,1e-30);t=(aa*ee-bb*dd)/np.maximum(den,1e-30)
        valid=(den>1e-24)&(s>=0)&(s<=1)&(t>=0)&(t<=1)
        sq=np.sum((r+s[...,None]*v-t[...,None]*w)**2,axis=-1)
        if valid.any():best=min(best,float(np.sqrt(sq[valid].min())))
    # Segment-through-face intersections are missed by vertex/edge minima.
    for edges,faces in ((ea,b),(eb,a)):
        e1=faces[:,1]-faces[:,0];e2=faces[:,2]-faces[:,0]
        for batch in np.array_split(edges,max(1,int(np.ceil(len(edges)/24)))):
            lo=batch.min(axis=1)[:,None,:];hi=batch.max(axis=1)[:,None,:]
            overlap=np.all((lo<=faces.max(axis=1)+1e-10)&(hi>=faces.min(axis=1)-1e-10),axis=-1)
            ix,iy=np.where(overlap)
            if not len(ix):continue
            origin=batch[ix,0];direction=batch[ix,1]-origin
            cross=np.cross(direction,e2[iy]);det=np.sum(e1[iy]*cross,axis=-1)
            inv=np.where(abs(det)>1e-20,1./np.where(abs(det)>1e-20,det,1.),0.)
            offset=origin-faces[iy,0];u=np.sum(offset*cross,axis=-1)*inv
            q=np.cross(offset,e1[iy]);v=np.sum(direction*q,axis=-1)*inv
            t=np.sum(e2[iy]*q,axis=-1)*inv
            if np.any((abs(det)>1e-20)&(u>=0)&(v>=0)&(u+v<=1)&(t>=0)&(t<=1)):return 0.
    return float(best)


def write_scenario(path,data):
    path=Path(path);assert not path.exists()
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')


def measure_before_step(stage,robot,cup,asset,offsets):
    from pxr import Usd,UsdPhysics
    from pipeline.ue_isaac_pipeline.arm_kinematics import pose_matrix
    cup_local=triangles(stage.GetPrimAtPath(CUP),stage.GetPrimAtPath(CUP))
    cp=cup.data.root_state_w[0,:7].cpu().numpy()
    cup_world=(np.c_[cup_local.reshape(-1,3),np.ones(cup_local.size//3)]@pose_matrix(cp[:3],cp[3:]))[:,:3].reshape(-1,3,3)
    cabinet=triangles(stage.GetPrimAtPath(CABINET))
    distance=surface_distance(cup_world,cabinet)
    robot_stage=Usd.Stage.Open(str(asset));hand_distances=[];candidates=[]
    for link in robot_stage.Traverse():
        if not link.HasAPI(UsdPhysics.RigidBodyAPI) or '_hand_' not in link.GetName():continue
        local=triangles(link,link);index=robot.body_names.index(link.GetName())
        pose=robot.data.body_state_w[0,index,:7].cpu().numpy()
        world=(np.c_[local.reshape(-1,3),np.ones(local.size//3)]@pose_matrix(pose[:3],pose[3:]))[:,:3].reshape(-1,3,3)
        lower=np.linalg.norm(np.maximum(np.maximum(cup_world.min(axis=(0,1))-world.max(axis=(0,1)),world.min(axis=(0,1))-cup_world.max(axis=(0,1))),0.))
        candidates.append((float(lower),link.GetName(),world))
    best=float('inf')
    for lower,name,world in sorted(candidates,key=lambda x:x[0]):
        if lower>=best:
            hand_distances.append(dict(link=name,lower_bound_m=lower,pruned_by_proven_closer_link=True));continue
        measured=surface_distance(world,cup_world);best=min(best,measured)
        hand_distances.append(dict(link=name,distance_m=measured))
    closest=min((x for x in hand_distances if 'distance_m' in x),key=lambda x:x['distance_m'])
    return dict(scenario='v2',offsets=offsets,timing='After original default-state writes; BEFORE first original physics step',
        cup_world_position_m=cp[:3].tolist(),t0_cup_cabinet_surface_distance_m=distance,
        t0_hand_cup_surface_distance_m=closest['distance_m'],closest_hand_link=closest['link'],hand_distances=hand_distances,
        geometry_method='Authored collider triangles in native measured body frames; exact vertex-face/edge-edge/segment-face surface minimum; unsigned, not cooked convex penetration depth',
        support_included_in_cabinet_distance=True,required_cabinet_clearance_m=.08,
        geometric_clearance_pass=distance>=.08)


def complete_preflight(before,observer,ledger):
    native=ledger.export()
    setup=[x for x in observer.events if x.get('reports',0)>0]
    support=[x for x in native['pairs'] if x['kind']=='cup_environment' and x.get('last_vertical_force_N',0)>.1]
    cabinet_contacts=[x for x in native['pairs'] if x['kind']=='cup_environment' and CABINET in x['actors']]
    result=dict(before)
    result.update(S2_classification=classify_initial_contacts(setup),first_two_step_hand_cup_contacts=setup,
        native_support_contacts=support,native_cup_cabinet_contacts=cabinet_contacts,
        support_contact_verified=bool(support),no_initial_hand_cup_contacts=not setup,
        G0_pass=bool(before['geometric_clearance_pass'] and support and not setup),
        native_cup_cabinet_min_separation_m=min((x['minimum_separation_m'] for x in cabinet_contacts),default=None),
        original_initializer_not_changed=True,BC_inference_calls=0)
    return result
