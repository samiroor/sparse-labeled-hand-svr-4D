import os
import cv2
import shutil
import numpy as np

import trimesh
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import torch

from . import fastrender
from . import visutils


links  = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [0, 5],
    [5, 6],
    [6, 7],
    [7, 8],
    [0, 9],
    [9, 10],
    [10, 11],
    [11, 12],
    [0, 13],
    [13, 14],
    [14, 15],
    [15, 16],
    [0, 17],
    [17, 18],
    [18, 19],
    [19, 20]
    ]


links_colors = [
  [0.4, 0.4, 0.4],
  [0.4, 0.0, 0.0],
  [0.6, 0.0, 0.0],
  [0.8, 0.0, 0.0],
  [1.0, 0.0, 0.0],
  [0.4, 0.4, 0.0],
  [0.6, 0.6, 0.0],
  [0.8, 0.8, 0.0],
  [1.0, 1.0, 0.0],
  [0.0, 0.4, 0.2],
  [0.0, 0.6, 0.3],
  [0.0, 0.8, 0.4],
  [0.0, 1.0, 0.5],
  [0.0, 0.2, 0.4],
  [0.0, 0.3, 0.6],
  [0.0, 0.4, 0.8],
  [0.0, 0.5, 1.0],
  [0.4, 0.0, 0.4],
  [0.6, 0.0, 0.6],
  [0.7, 0.0, 0.8],
  [1.0, 0.0, 1.0]
  ]


# skeleton 2D (numpy)
def skeleton2D(
    axis,
    coords2D,
    visibility=None,
    color=None,
    linewidth='1',
    draw_kp=True,
    title='',
    ):

  if visibility is None:
    visibility = np.ones_like(coords2D[:, 0]) == 1.0

  for link, c in zip(links, links_colors):
    if (visibility[link[0]] == False) or (visibility[link[1]] == False):
      continue
    coord1 = coords2D[link[0], :]
    coord2 = coords2D[link[1], :]
    pts = np.stack([coord1, coord2])
    axis.plot(
        pts[:, 0],
        pts[:, 1],
        color=c if color is None else color,
        linewidth=linewidth,
        )

  for i in range(coords2D.shape[0]):
    if visibility[i]:
      continue
    axis.plot(
        coords2D[i, 0],
        coords2D[i, 1],
        'o',
        color=links_colors[i, :] if color is None else color,
        )

  if title is not None:
    axis.set_title('{}'.format(title))
  axis.set_xticklabels([])
  axis.set_yticklabels([])
  axis.axis("off")


# skeleton 3D (numpy)
def skeleton3D(
    axis,
    coords3D,
    visibility=None,
    color=None,
    linewidth='3',
    draw_kp=True,
    title=None,
    ):

  if visibility is None:
    visibility = np.ones_like(coords3D[:, 0]) == 1.0

  for link, c in zip(links, links_colors):
    if (visibility[link[0]] == False) or (visibility[link[1]] == False):
      continue
    coord1 = coords3D[link[0], :]
    coord2 = coords3D[link[1], :]
    pts = np.stack([coord1, coord2])
    axis.plot(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        color=c if color is None else color,
        linewidth=linewidth,
        )

  for i in range(coords3D.shape[0]):
    if visibility[i] :
      continue
    axis.plot(
        coords3D[i, 0],
        coords3D[i, 1],
        coords3D[i, 2],
        'o',
        color=c if color is None else color,
        )

  if title is not None:
    axis.set_title('{}'.format(title))

  #axis.axis('equal')
  axis.set_zticklabels([])
  axis.set_yticklabels([])
  axis.set_xticklabels([])
  axis.view_init(elev=-85, azim=-75)


# mesh 3D (numpy)
def mesh(
    axis,
    verts,
    faces,
    alpha=0.1,
    flip_x=False,
    color=(0.,0.,1.),
    title=None,
    ):
  axis.view_init(elev=90, azim=-90)
  mesh = Poly3DCollection(verts[faces], alpha=alpha)

  face_color = np.array(color) * 0.5 + 127./255.
  edge_color = np.array(color) * 0.5

  mesh.set_edgecolor(edge_color)
  mesh.set_facecolor(face_color)
  axis.add_collection3d(mesh)
  cam_3d(axis, verts, flip_x=flip_x)

  if title is not None:
    axis.set_title('{}'.format(title))


def cam_3d(
    axis,
    verts,
    flip_x=False,
    ):

  """
    Centers view on cuboid containing hand and flips y and z axis
    and fixes azimuth
  """
  extents = np.stack([verts.min(0), verts.max(0)], axis=1)
  sz = extents[:, 1] - extents[:, 0]
  centers = np.mean(extents, axis=1)
  maxsize = max(abs(sz))

  r = maxsize / 2
  if flip_x:
    axis.set_xlim(centers[0] + r, centers[0] - r)
  else:
    axis.set_xlim(centers[0] - r, centers[0] + r)

  # Invert y and z axis
  axis.set_ylim(centers[1] + r, centers[1] - r)
  axis.set_zlim(centers[2] + r, centers[2] - r)


# fast renderer in cuda (torch)
def render(
    verts_gt=None,
    verts_prd=None,
    faces=None,
    images=None,#b,c,H,W
    camsintr=None,
    colors_prd=None,
    colors_gt=None,
    is_hand=True,
    ):

  input_res = (images.shape[3], images.shape[2]) if images is not None else (256,256)

  if verts_gt is not None:
    if colors_gt is None:
      colors_gt = visutils.get_verts_colors(verts_gt, [0.4, 0.6, 0.9])
    if colors_prd is None and verts_prd is not None:
      colors_prd = err_cmap_colors(verts_prd, verts_gt)

  if colors_prd is None and verts_prd is not None:
    colors_prd = visutils.get_verts_colors(verts_prd, [0.90, 0.70, 0.7])

  #if is_hand:
  #  faces,_ = visutils.get_closed_faces() #TODO approach

  render_gt = None; render_prd = None; render_rotated_prd = None
  render_list = list()

  if images is not None:
    render_list.append(images.permute(0, 2, 3, 1))

  # gt
  if verts_gt is not None:
    render_gt = fastrender.render(
        verts_gt,
        faces,
        input_res,
        camsintr,
        colors_gt,
        bg_color=images,
        )
    render_list.append(render_gt)
    #render_gt = fastrender.render(
    #    verts_gt,
    #    faces,
    #    input_res,
    #    camsintr,
    #    colors_gt,
    #    fill_back=False,
    #    bg_color=None,
    #    )
    #render_list.append(render_gt)


  # prd
  if verts_prd is not None:
    render_prd = fastrender.render(
        verts_prd,
        faces,
        input_res,
        camsintr,
        colors_prd,
        bg_color=images,
        )
    render_list += [render_prd]

    verts_rotated_prd = verts_prd
    for i in range(2):
      verts_rotated_prd = visutils.rotate_verts(verts_rotated_prd)
      render_rotated_prd = fastrender.render(
        verts_rotated_prd,
        faces,
        input_res,
        camsintr,
        colors_prd,
        fill_back=False,
        bg_color=None,
        )
      render_list +=  [render_rotated_prd]

  assert len(render_list) > 0., 'wrong'
  return torch.cat(render_list, dim=2) # b, H, W, c


# save images (numpy)
def save_images(
    images,
    figaspect=0.16,
    fontsize=8,
    get_cmap=True,
    title=None,
    save_dir='tmp',
    ):

  os.makedirs(save_dir, exist_ok=True)

  for i, img in enumerate(images):
    w,h = matplotlib.figure.figaspect(figaspect)
    fig = plt.figure(figsize=(w,h))
    ax = fig.subplots(nrows=1, ncols=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.imshow(img)

    if get_cmap:
      x0 = ax.get_position().x0; y0 = ax.get_position().y0
      x1 = ax.get_position().x1; y1 = ax.get_position().y1
      cb, _ = cmap_bar(fig, xlim=[x0, y0-0.02, x1-x0, 0.02])

    if title is not None:
      ax.set_title('{}'.format(title), fontsize=fontsize)

    fig.savefig(
        os.path.join(save_dir,f'{i}.png'), dpi=300
        )
    plt.close(fig)


# make video
def create_video(video_file, img_folder='tmp', suffix='%d.png'):
  print(f'Creating \n {video_file}')
  visutils.images_to_video(img_folder=img_folder, video_file=video_file, suffix=suffix)
  #shutil.rmtree(img_folder)
  print('-  END')


# cmap bar to show errors
def cmap_bar(fig, xlim=[0.27, 0.05, 0.5, 0.02], max_val=0.1):
  cax = fig.add_axes(xlim)
  cmap = matplotlib.cm.get_cmap("jet")
  norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
  cb = matplotlib.colorbar.ColorbarBase(
      cax,
      cmap=cmap,
      norm=norm,
      ticks=[0, 1],
      orientation="horizontal",
      )
  cb.ax.set_xticklabels(["0", f'{max_val * 1000}'])
  cb.set_label("3D mesh error (mm)")
  return cb, fig


def err_cmap_colors(verts_prd, verts_gt, max_val=0.1):
  verts_errs = (verts_prd - verts_gt).norm(p=2, dim=-1)
  errs_colors = verts_errs / max_val

  jet_cmap = matplotlib.cm.get_cmap("jet")
  errs_cmap = jet_cmap(errs_colors.detach().cpu().numpy())
  return verts_gt.new(errs_cmap)

# save ply meshes
def save_mesh_ply(verts, faces, name):
  #TODO memory leak
  mesh = trimesh.Trimesh(
      vertices=verts,
      faces=faces,
      process=False,
      )
  mesh.export(name)

