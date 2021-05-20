import torch

import neural_renderer as nr
from . import visutils

# should be in meter #TODO 
# in cuda

def render(
    verts,
    faces,
    input_res,
    camintrs=None,
    colors=None,
    fill_back=True,
    near=0.05,
    far=2,
    crop_to_img=True,
    bg_color=None,
):
    """
    Given vertices, faces, resolution input_res of image and camera
    intrinsic parameters, render the corresponding image using
    neural_renderer as a backend
    """
    max_size = max(input_res)
    neurenderer = nr.Renderer(
        camera_mode='projection',
        image_size=max_size,
        R=torch.eye(3).unsqueeze(0).cuda(),
        t=torch.zeros(1, 3).cuda(),
        K=camintrs,
        orig_size=max_size,
        anti_aliasing=False,
        fill_back=fill_back,
        near=near,
        far=far,
        light_intensity_ambient=0.8,
    )
    if colors is None:
        colors = torch.ones_like(verts)
    textures = visutils.batch_vertex_textures(faces, colors[:, :, :3])

    render_rgb, render_depth, render_alpha = neurenderer(verts, faces, textures)

    # crop to original image dimensions
    if crop_to_img:
        render_rgb = render_rgb[:, :, : input_res[1], : input_res[0]]
        render_alpha = render_alpha[:, : input_res[1], : input_res[0]].unsqueeze(1)

    # add background
    if bg_color is None:
      bg_color = torch.ones_like(render_rgb)
    render_rgb = (render_rgb * render_alpha) + bg_color * (1 - render_alpha) * torch.ones_like(render_rgb)

    return render_rgb.clamp(0,1).permute(0, 2, 3, 1) # b, H, W, c
