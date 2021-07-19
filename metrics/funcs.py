#! https://github.com/lmb-freiburg/freihand
import os
import pip
import json
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import open3d as o3d
from scipy.linalg import orthogonal_procrustes


def verts2pcd(verts, color=None):
    pcd = o3d.PointCloud()
    pcd.points = o3d.Vector3dVector(verts)
    if color is not None:
        if color == 'r':
            pcd.paint_uniform_color([1, 0.0, 0])
        if color == 'g':
            pcd.paint_uniform_color([0, 1.0, 0])
        if color == 'b':
            pcd.paint_uniform_color([0, 0, 1.0])
    return pcd


def calculate_fscore(gt, pr, th=0.01):
    gt = verts2pcd(gt)
    pr = verts2pcd(pr)
    d1 = o3d.compute_point_cloud_to_point_cloud_distance(gt, pr)  # closest dist for each gt point
    d2 = o3d.compute_point_cloud_to_point_cloud_distance(pr, gt)  # closest dist for each pred point
    if len(d1) and len(d2):
        recall = float(sum(d < th for d in d2)) / float(
            len(d2))  # how many of our predicted points lie close to a gt point?
        precision = float(sum(d < th for d in d1)) / float(len(d1))  # how many of gt points are matched?

        if recall + precision > 0:
            fscore = 2 * recall * precision / (recall + precision)
        else:
            fscore = 0
    else:
        fscore = 0
        precision = 0
        recall = 0
    return fscore, precision, recall

def align_tr(mtx1, mtx2):
    """ Align the 3D joint location with the ground truth by scaling and translation """

    predCurr = mtx2.copy()
    # make preditions translation of the wrist joint same as ground truth
    predCurrRel = predCurr - predCurr[0:1, :]
    preds_tr_al = predCurrRel + mtx1[0:1, :]

    return preds_tr_al


def align_sc_tr(mtx1, mtx2, ref_bone):
    """ Align the 3D joint location with the ground truth by scaling and translation """

    predCurr = mtx2.copy()
    # normalize the predictions
    s = np.sqrt(np.sum(np.square(predCurr[ref_bone[1]] - predCurr[ref_bone[0]])))
    if s > 0:
        predCurr = predCurr / s

    # get the scale of the ground truth
    sGT = np.sqrt(np.sum(np.square(mtx1[ref_bone[1]] - mtx1[ref_bone[0]])))

    # make predictions scale same as ground truth scale
    predCurr = predCurr * sGT

    # make preditions translation of the wrist joint same as ground truth
    predCurrRel = predCurr - predCurr[0:1, :]
    preds_sc_tr_al = predCurrRel + mtx1[0:1, :]

    return preds_sc_tr_al


def align_sc_depth(mtx1, mtx2, return_trafo=False, fw=None):
    """ Align the predicted entity in some depth & scale sense with the ground truth. """
    # center
    t1 = mtx1.mean(0)
    t2 = mtx2.mean(0)
    mtx1_t = mtx1 - t1
    mtx2_t = mtx2 - t2

    # scale
    s1 = np.linalg.norm(mtx1_t) + 1e-8
    mtx1_t /= s1
    s2 = np.linalg.norm(mtx2_t) + 1e-8
    mtx2_t /= s2

    # orth alignment
    R, s = orthogonal_procrustes(mtx1_t, mtx2_t)

    # scale alignment (has same scale as mtx1)
    mtx2_t = mtx2_t * s * s1
    # depth alignment (has same depth as mtx1)
    mtx2_t += np.concatenate([t2[:2], t1[-1:]], axis=-1)

    return mtx2_t


def align_w_scale(mtx1, mtx2, return_trafo=False, fw=None):
    """ Align the predicted entity in some optimality sense with the ground truth. """
    # center
    t1 = mtx1.mean(0)
    t2 = mtx2.mean(0)
    mtx1_t = mtx1 - t1
    mtx2_t = mtx2 - t2

    # scale
    s1 = np.linalg.norm(mtx1_t) + 1e-8
    mtx1_t /= s1
    s2 = np.linalg.norm(mtx2_t) + 1e-8
    mtx2_t /= s2

    # orth alignment
    R, s = orthogonal_procrustes(mtx1_t, mtx2_t)

    # apply trafos to the second matrix
    mtx2_t = np.dot(mtx2_t, R.T) * s
    mtx2_t = mtx2_t * s1 + t1

    if fw is not None:
        fw.write('{},{},{}\n'.format(s,s1,s2))

    if return_trafo:
        return R, s, s1, t1 - t2
    else:
        return mtx2_t


def align_w_scale_woR(mtx1, mtx2, return_trafo=False):
    """ Align the predicted entity in some optimality sense with the ground truth. """
    # center
    t1 = mtx1.mean(0)
    t2 = mtx2.mean(0)
    mtx1_t = mtx1 - t1
    mtx2_t = mtx2 - t2

    # scale
    s1 = np.linalg.norm(mtx1_t) + 1e-8
    mtx1_t /= s1
    s2 = np.linalg.norm(mtx2_t) + 1e-8
    mtx2_t /= s2

    # orth alignment
    R, s = orthogonal_procrustes(mtx1_t, mtx2_t)

    # apply trafos to the second matrix
    mtx2_t = mtx2_t * s #np.dot(mtx2_t, R.T) * s
    mtx2_t = mtx2_t * s1 + t1
    if return_trafo:
        return R, s, s1, t1 - t2
    else:
        return mtx2_t


def align_by_trafo(mtx, trafo):
    t2 = mtx.mean(0)
    mtx_t = mtx - t2
    R, s, s1, t1 = trafo
    return np.dot(mtx_t, R.T) * s * s1 + t1 + t2


class curve:
    def __init__(self, x_data, y_data, x_label, y_label, text):
        self.x_data = x_data
        self.y_data = y_data
        self.x_label = x_label
        self.y_label = y_label
        self.text = text


def createHTML(outputDir, curve_list):
    curve_data_list = list()
    for item in curve_list:
        fig1 = plt.figure()
        ax = fig1.add_subplot(111)
        ax.plot(item.x_data, item.y_data)
        ax.set_xlabel(item.x_label)
        ax.set_ylabel(item.y_label)
        img_path = os.path.join(outputDir, "img_path_path.png")
        plt.savefig(img_path, bbox_inches=0, dpi=300)

        # write image and create html embedding
        data_uri1 = open(img_path, 'rb').read().encode('base64').replace('\n', '')
        img_tag1 = 'src="data:image/png;base64,{0}"'.format(data_uri1)
        curve_data_list.append((item.text, img_tag1))

        os.remove(img_path)

    htmlString = '''<!DOCTYPE html>
    <html>
    <body>
    <h1>Detailed results:</h1>'''

    for i, (text, img_embed) in enumerate(curve_data_list):
        htmlString += '''
        <h2>%s</h2>
        <p>
        <img border="0" %s alt="FROC" width="576pt" height="432pt">
        </p>
        <p>Raw curve data:</p>

        <p>x_axis: <small>%s</small></p>
        <p>y_axis: <small>%s</small></p>

        ''' % (text, img_embed, curve_list[i].x_data, curve_list[i].y_data)

    htmlString += '''
    </body>
    </html>'''

    htmlfile = open(os.path.join(outputDir, "scores.html"), "w")
    htmlfile.write(htmlString)
    htmlfile.close()


def angle_distance(R_gt, R):
    d = 1. - np.trace(R_gt.dot(R.T))
    return d

