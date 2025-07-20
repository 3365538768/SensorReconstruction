import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    stage="fine",
    cam_type=None
):
    # 屏幕占位
    screenspace_points = torch.zeros_like(pc.get_xyz,
                                         dtype=pc.get_xyz.dtype,
                                         requires_grad=True,
                                         device=bg_color.device)
    try: screenspace_points.retain_grad()
    except: pass

    means3D = pc.get_xyz

    # 选择协方差 or scale+rotation
    if pipe.compute_cov3D_python:
        scales = None
        rotations = None
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation
        cov3D_precomp = None

    # Raster 设置
    if cam_type != "PanopticSports":
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,  # 保留给 cov3D 分支
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug,
           
        )
        time_tensor = torch.tensor(viewpoint_camera.time,
                                   device=means3D.device
                                  ).repeat(means3D.shape[0],1)
    else:
        raster_settings = viewpoint_camera['camera']
        time_tensor = torch.tensor(viewpoint_camera['time'],
                                   device=means3D.device
                                  ).repeat(means3D.shape[0],1)

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # deformation or coarse
    if "coarse" in stage:
        means3D_final   = means3D
        scales_final    = scales
        rotations_final = rotations
        opacity_final   = pc._opacity
        shs_final       = pc.get_features
    else:
        (means3D_final,
         scales_final,
         rotations_final,
         opacity_final,
         shs_final) = pc._deformation(
            means3D, scales, rotations,
            pc._opacity, pc.get_features,
            time_tensor
        )
        with torch.no_grad():
            disp = torch.abs(means3D_final - means3D)
            if hasattr(pc, "_deformation_accum"):
                pc._deformation_accum = torch.maximum(pc._deformation_accum, disp)

    # —— 关键修改 ——
    if scales_final is not None:
        # 乘上 scaling_modifier 后再激活
        scales_final = pc.scaling_activation(scales_final * scaling_modifier)

    if rotations_final is not None:
        rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)

    # 颜色预处理
    if override_color is not None:
        colors_precomp = override_color
        shs_for_gpu    = None
    elif pipe.convert_SHs_python:
        shs_for_gpu    = None
        shs_view       = pc.get_features.transpose(1,2).view(
                            -1, 3, (pc.max_sh_degree+1)**2)
        dirs           = pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(
                            pc.get_features.shape[0],1)
        dirs           = dirs/dirs.norm(dim=1,keepdim=True)
        rgb            = eval_sh(pc.active_sh_degree, shs_view, dirs)
        colors_precomp = torch.clamp_min(rgb + 0.5, 0.0)
    else:
        colors_precomp = None
        shs_for_gpu    = shs_final

    # 执行 rasterizer
    rendered_image, radii, depth = rasterizer(
        means3D        = means3D_final,
        means2D        = screenspace_points,
        shs            = shs_for_gpu,
        colors_precomp = colors_precomp,
        opacities      = opacity,
        scales         = scales_final,
        rotations      = rotations_final,
        cov3D_precomp  = cov3D_precomp
    )

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": depth
    }