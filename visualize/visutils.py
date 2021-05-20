import os
import subprocess

import torch
from manopth import rodrigues_layer
from manopth import manolayer


def vertex_textures(
    faces, vertex_colors, texture_size=2, with_artefacts=False
):
    """
    Create textures for neural_renderer from vertex colors
    Args:
        faces (nb_faces, 3): contains the indexes of vertex for each face
        vertex_colors (vertex_nb, 3)
        with_artefacts (bool): Create artefacts that highlight vertices
    """
    assert faces.dim() == 2, f"expected shape (face_nb, 3), got {faces.shape}"
    assert (
        vertex_colors.dim() == 2
    ), f"expected shape (face_nb, 3), got {faces.shape}"
    assert (
        faces.shape[1] == 3
    ), f"expected shape (face_nb, 3), got {faces.shape}"
    assert (
        vertex_colors.shape[1] == 3
    ), f"expect shape (face_nb, 3), got {faces.shape}"
    textures = torch.zeros(
        faces.shape[0], texture_size, texture_size, texture_size, 3
    )
    f1_colors = vertex_colors[faces[:, 0]]
    f2_colors = vertex_colors[faces[:, 1]]
    f3_colors = vertex_colors[faces[:, 2]]
    if texture_size > 1:
        for tex_idx1 in range(texture_size):
            for tex_idx2 in range(texture_size):
                for tex_idx3 in range(texture_size):
                    sum_idxs = tex_idx1 + tex_idx2 + tex_idx3
                    if sum_idxs:
                        if with_artefacts:
                            textures[:, tex_idx1, tex_idx2, tex_idx3] = (
                                tex_idx1 * f1_colors
                                + tex_idx2 * f2_colors
                                + tex_idx3 * f3_colors
                            ) / sum_idxs
                        else:
                            textures[:, tex_idx1, tex_idx2, tex_idx3] = (
                                tex_idx1 * f1_colors
                                + tex_idx2 * f2_colors
                                + tex_idx3 * f3_colors
                            )
    else:
        textures[:, 0, 0, 0] = f1_colors
    return textures


def batch_vertex_textures(
    faces, vertex_colors, texture_size=2, with_artefacts=False
):
    """
    Create textures for neural_renderer from vertex colors
    Args:
        faces (batch_size, nb_faces, 3): contains the indexes of vertex for each face
        vertex_colors (batch_size, vertex_nb, 3)
    """
    assert faces.dim() == 3, f"expected shape (face_nb, 3), got {faces.shape}"
    assert (
        vertex_colors.dim() == 3
    ), f"expected shape (batch_size, face_nb, 3), got {faces.shape}"
    assert (
        faces.shape[2] == 3
    ), f"expected shape (batch_size, face_nb, 3), got {faces.shape}"
    assert (
        vertex_colors.shape[2] == 3
    ), f"expect shape (face_nb, 3), got {faces.shape}"
    textures = vertex_colors.new_zeros(
        faces.shape[0],
        faces.shape[1],
        texture_size,
        texture_size,
        texture_size,
        3,
    )
    f1_colors = torch.gather(
        vertex_colors, 1, faces[:, :, 0:1].repeat(1, 1, 3).long()
    )
    f2_colors = torch.gather(
        vertex_colors, 1, faces[:, :, 1:2].repeat(1, 1, 3).long()
    )
    f3_colors = torch.gather(
        vertex_colors, 1, faces[:, :, 2:].repeat(1, 1, 3).long()
    )
    if texture_size > 1:
        for tex_idx1 in range(texture_size):
            for tex_idx2 in range(texture_size):
                for tex_idx3 in range(texture_size):
                    sum_idxs = tex_idx1 + tex_idx2 + tex_idx3
                    if sum_idxs:
                        if with_artefacts:
                            textures[:, :, tex_idx1, tex_idx2, tex_idx3] = (
                                tex_idx1 * f1_colors
                                + tex_idx2 * f2_colors
                                + tex_idx3 * f3_colors
                            ) / sum_idxs
                        else:
                            textures[:, :, tex_idx1, tex_idx2, tex_idx3] = (
                                tex_idx1 * f1_colors
                                + tex_idx2 * f2_colors
                                + tex_idx3 * f3_colors
                            )
    else:
        textures[:, :, 0, 0, 0] = f1_colors
    return textures


def rotate_verts(verts, axisang=(0, 1, 1)):
    centroids = verts.mean(1).unsqueeze(1)
    verts_c = verts - centroids
    rot_mats = rodrigues_layer.batch_rodrigues(verts.new(axisang).unsqueeze(0)).view(1, 3, 3)
    verts_cr = rot_mats.repeat(verts.shape[0], 1, 1).bmm(verts_c.transpose(1, 2)).transpose(1, 2)
    verts_final = verts_cr + centroids
    return verts_final


def get_closed_faces():
    mano_layer = manolayer.ManoLayer(
        joint_rot_mode="axisang", use_pca=False, mano_root="misc/mano_v1_2/models", center_idx=None, flat_hand_mean=True
    )
    close_faces = torch.Tensor(
        [
            [92, 38, 122],
            [234, 92, 122],
            [239, 234, 122],
            [279, 239, 122],
            [215, 279, 122],
            [215, 122, 118],
            [215, 118, 117],
            [215, 117, 119],
            [215, 119, 120],
            [215, 120, 108],
            [215, 108, 79],
            [215, 79, 78],
            [215, 78, 121],
            [214, 215, 121],
        ]
    )
    closed_faces = torch.cat([mano_layer.th_faces, close_faces.long()])
    # Indices of faces added during closing --> should be ignored as they match the wrist
    # part of the hand, which is not an external surface of the human

    # Valid because added closed faces are at the end
    hand_ignore_faces = [1538, 1539, 1540, 1541, 1542, 1543, 1544, 1545, 1546, 1547, 1548, 1549, 1550, 1551]

    return closed_faces, hand_ignore_faces


def get_verts_colors(verts, color=None):
    """
    Args:
        color: either [r, g, b] with each color in [0, 1] or None
            if None, maps xyz as color
    """
    if color is None:
        batch_size = verts.shape[0]
        colors = verts - verts.mean(1).unsqueeze(1)
        colors = colors / colors.norm(2, 2).max(1)[0].view(batch_size, 1, 1)
        colors = colors / 2 + 0.5
    else:
        colors = torch.ones_like(verts)
        colors[:, :, 0] = color[0] * colors[:, :, 0]
        colors[:, :, 1] = color[1] * colors[:, :, 1]
        colors[:, :, 2] = color[2] * colors[:, :, 2]
    return colors


def images_to_video(img_folder, video_file, suffix='%d.png'):
    command = [
        'ffmpeg','-framerate','1', '-y', '-threads', '16', '-i', f'{os.path.join(img_folder, suffix)}', '-profile:v', 'baseline',
        '-level', '3.0', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-an', '-v', 'error', video_file,
    ]

    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)
