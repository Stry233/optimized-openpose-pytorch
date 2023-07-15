# -*- coding: utf-8 -*-
"""
OpenPose body pose estimator.
Edited on Fri July 14 12:00:00 2023
Author: Yuetian Chen
GitHub: https://github.com/Stry233/optimized-openpose-pytorch

Created on Wed Sep 11 19:00:00 2019
Author: Prasun Roy | CVPRU-ISICAL (http://www.isical.ac.in/~cvpr)
GitHub: https://github.com/prasunroy/openpose-pytorch

Original author: Zhizhong Huang
Original source: https://github.com/Hzzone/pytorch-openpose

"""

import cv2
import os
import numpy as np
import shutil
import tempfile
import torch
import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter
from tqdm import tqdm
from urllib.parse import urlparse
from urllib.request import urlopen
from .model import BodyPoseModel
import torch.nn.functional as F

model_url = 'https://www.dropbox.com/s/mun9eh2509pw32n/openpose_body_coco_pose_iter_440000.pth?dl=1'
model_dir = os.path.join(os.path.expanduser('~'), '.cache/torch/checkpoints/')
from line_profiler import LineProfiler


def process_output(tensor, pads, image_padded_shape, image_shape, stride):
    tensor = tensor.squeeze(0)
    tensor = F.interpolate(tensor.unsqueeze(0), scale_factor=stride, mode='bilinear',
                           align_corners=False).squeeze(0)
    tensor = tensor[:, :image_padded_shape[0] - pads[3], :image_padded_shape[1] - pads[2]]
    tensor = F.interpolate(tensor.unsqueeze(0), size=(image_shape[0], image_shape[1]),
                           mode='bilinear', align_corners=False).squeeze(0)
    return tensor.permute(1, 2, 0)


def _download_url_to_file(url, path, progress=True):
    link = urlopen(url)
    meta = link.info()
    content_length = meta.get_all('Content-Length')
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])
    else:
        file_size = None
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    try:
        with tqdm(total=file_size, disable=not progress,
                  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = link.read(8192)
                if len(buffer) == 0:
                    break
                temp_file.write(buffer)
                pbar.update(len(buffer))
        temp_file.close()
        shutil.move(temp_file.name, path)
    finally:
        temp_file.close()
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)


def _load_state_dict_from_url(model_url, model_dir, progress=True):
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    urlparts = urlparse(model_url)
    filename = os.path.basename(urlparts.path.split('/')[-1])
    cached_file = os.path.join(model_dir, filename)
    if not os.path.isfile(cached_file):
        print(f'Downloading: "{model_url}" to {cached_file}')
        _download_url_to_file(model_url, cached_file, progress)
    return torch.load(cached_file)


def _load_state_dict(model, state_dict):
    model_state_dict = {}
    for name in model.state_dict().keys():
        model_state_dict[name] = state_dict['.'.join(name.split('.')[1:])]
    model.load_state_dict(model_state_dict)
    return model


def _pad_image(image, stride=1, padvalue=0):
    assert len(image.shape) == 2 or len(image.shape) == 3
    h, w = image.shape[:2]
    pads = [None] * 4
    pads[0] = 0  # left
    pads[1] = 0  # top
    pads[2] = 0 if (w % stride == 0) else stride - (w % stride)  # right
    pads[3] = 0 if (h % stride == 0) else stride - (h % stride)  # bottom
    num_channels = 1 if len(image.shape) == 2 else image.shape[2]
    image_padded = torch.ones((h + pads[3], w + pads[2], num_channels)) * padvalue
    image_padded = torch.squeeze(image_padded)
    image_padded[:h, :w] = torch.tensor(image)
    return image_padded, pads


def _get_keypoints(candidates, subsets):
    keypoints = torch.zeros((1, 18, 3), dtype=torch.int32, device=candidates.device)

    valid_mask = subsets[:, :18].long() >= 0

    # Extract the corresponding rows from the candidates tensor
    selected_candidates = candidates[:, :3].to(torch.int32)
    selected_candidates[:, 2] = 1  # Set the third entry of each row to 1

    # Create a mask for the keypoints tensor with the same shape
    keypoints_mask = torch.zeros_like(keypoints, dtype=torch.bool).to(keypoints.device)

    # Set the corresponding indices in the keypoints mask to True
    keypoints_mask[0, torch.where(valid_mask[0])[0], :] = True

    # Use masked_scatter_ to assign the values from the selected candidates to the keypoints tensor
    keypoints.masked_scatter_(keypoints_mask, selected_candidates)
    return keypoints



class BodyPoseEstimator(object):

    def __init__(self, pretrained=False):
        self._model = BodyPoseModel()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.map_idx = torch.tensor([[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44],
                                     [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30],
                                     [47, 48], [49, 50], [53, 54], [51, 52], [55, 56], [37, 38],
                                     [45, 46]]).to(self.device)
        self.limbseq = torch.tensor([[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
                                     [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
                                     [1, 16], [16, 18], [3, 17], [6, 18]]).to(self.device)

        # self.transform = T.GaussianBlur(kernel_size=3, sigma=3)

        self._model = self._model.half().to(self.device)

        if pretrained:
            state_dict = _load_state_dict_from_url(model_url, model_dir)
            self._model = _load_state_dict(self._model, state_dict)
        self._model.eval()

    # @do_profile()
    def __call__(self, image):
        scales = [0.5]
        stride = 8
        bboxsize = 368
        padvalue = 128
        thresh_1 = 0.1
        thresh_2 = 0.05

        bbox_scale = bboxsize / image.shape[0]
        multipliers = [scale * bbox_scale for scale in scales]
        heatmap_avg = torch.zeros((image.shape[0], image.shape[1], 19)).to(self.device)
        paf_avg = torch.zeros((image.shape[0], image.shape[1], 38)).to(self.device)

        for scale in multipliers:
            image_scaled = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            image_padded, pads = _pad_image(image_scaled, stride, padvalue)

            # Combining operations, converting to float type earlier
            image_tensor = torch.from_numpy(
                np.ascontiguousarray(
                    np.expand_dims(np.transpose(image_padded, (2, 0, 1)), 0).astype(np.float32) / 255.0 - 0.5
                )
            ).float().half()

            if torch.cuda.is_available():
                image_tensor = image_tensor.cuda()

            with torch.no_grad():
                Mconv7_stage6_L1, Mconv7_stage6_L2 = self._model(image_tensor)

            paf = process_output(Mconv7_stage6_L1, pads, image_padded.shape, image.shape, stride)
            heatmap = process_output(Mconv7_stage6_L2, pads, image_padded.shape, image.shape, stride)

            heatmap_avg += (heatmap / len(multipliers))
            paf_avg += (paf / len(multipliers))

        all_peaks = []
        num_peaks = 0

        for part in range(18):
            map_orig = heatmap_avg[:, :, part]

            map_filt = torch.as_tensor((gaussian_filter(cp.asarray(map_orig), sigma=3)), device=self.device)

            map_L = torch.zeros_like(map_filt)
            map_T = torch.zeros_like(map_filt)
            map_R = torch.zeros_like(map_filt)
            map_B = torch.zeros_like(map_filt)

            map_L[1:, :] = map_filt[:-1, :]
            map_T[:, 1:] = map_filt[:, :-1]
            map_R[:-1, :] = map_filt[1:, :]
            map_B[:, :-1] = map_filt[:, 1:]

            # Assuming that map_filt, map_L, map_T, map_R, map_B and thresh_1 are all tensors
            conditions = torch.stack([(map_filt >= map_L),
                                      (map_filt >= map_T),
                                      (map_filt >= map_R),
                                      (map_filt >= map_B),
                                      (map_filt > thresh_1)], dim=0)

            peaks_binary = torch.all(conditions, dim=0)

            peaks = torch.nonzero(peaks_binary)[:, [1, 0]]

            # Adding scores to peaks
            scores = map_orig[peaks[:, 1], peaks[:, 0]].unsqueeze(-1)
            peaks_with_scores = torch.cat((peaks, scores), dim=1)

            # Adding ids to peaks
            peaks_ids = torch.arange(num_peaks, num_peaks + len(peaks)).unsqueeze(-1).to(peaks.device)
            peaks_with_scores_and_ids = torch.cat((peaks_with_scores, peaks_ids), dim=1)

            # Append to list of all peaks as tensors
            all_peaks.append(peaks_with_scores_and_ids)

            # Update num_peaks
            num_peaks += len(peaks)

        all_connections = []
        spl_k = []
        mid_n = 10

        for k in range(len(self.map_idx)):
            score_mid = paf_avg[:, :, torch.tensor([x - 19 for x in self.map_idx[k]])]
            candidate_A = all_peaks[self.limbseq[k][0] - 1]
            candidate_B = all_peaks[self.limbseq[k][1] - 1]
            n_A = len(candidate_A)
            n_B = len(candidate_B)

            if n_A != 0 and n_B != 0:
                connection_candidates = []
                for i in range(n_A):
                    for j in range(n_B):
                        v = torch.subtract(candidate_B[j][:2], candidate_A[i][:2])
                        n = torch.sqrt(v[0] * v[0] + v[1] * v[1])
                        v.div_(n)

                        # Using torch.linspace instead of np.linspace
                        ab_x = torch.linspace(candidate_A[i][0], candidate_B[j][0], steps=mid_n,
                                              device=self.device).round().int()
                        ab_y = torch.linspace(candidate_A[i][1], candidate_B[j][1], steps=mid_n,
                                              device=self.device).round().int()

                        # Using tensor indexing instead of list comprehension
                        vx = score_mid[ab_y, ab_x, 0]
                        vy = score_mid[ab_y, ab_x, 1]

                        score_midpoints = vx * v[0] + vy * v[1]

                        score_with_dist_prior = score_midpoints.mean() + max(0.5 * image.shape[0] / n - 1, 0)
                        criterion_1 = torch.count_nonzero(score_midpoints > thresh_2) > 0.8 * len(score_midpoints)
                        criterion_2 = score_with_dist_prior > 0

                        if criterion_1 and criterion_2:
                            connection_candidate = [i, j, score_with_dist_prior,
                                                    score_with_dist_prior + candidate_A[i][2] + candidate_B[j][2]]
                            connection_candidates.append(connection_candidate)
                connection_candidates = sorted(connection_candidates, key=lambda x: x[2], reverse=True)
                connection = torch.zeros((0, 5))
                for candidate in connection_candidates:
                    i, j, s = candidate[0:3]
                    if i not in connection[:, 3] and j not in connection[:, 4]:
                        connection = torch.vstack([connection,
                                                   torch.tensor([candidate_A[i][3], candidate_B[j][3], s, i, j])])
                        if len(connection) >= min(n_A, n_B):
                            break
                all_connections.append(connection)
            else:
                spl_k.append(k)
                all_connections.append([])

        candidate = torch.cat(all_peaks, dim=0)
        subset = torch.full((0, 20), -1, device=self.device)

        for k in range(len(self.map_idx)):
            if k not in spl_k:
                part_As = all_connections[k][:, 0]
                part_Bs = all_connections[k][:, 1]
                index_A, index_B = self.limbseq[k] - 1
                for i in range(len(all_connections[k])):
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)):
                        if subset[j][index_A] == part_As[i] or subset[j][index_B] == part_Bs[i]:
                            subset_idx[found] = j
                            found += 1
                    if found == 1:
                        j = subset_idx[0]
                        if subset[j][index_B] != part_Bs[i]:
                            subset[j][index_B] = part_Bs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[part_Bs[i].int(), 2] + all_connections[k][i][2]
                    elif found == 2:
                        j1, j2 = subset_idx
                        membership = ((subset[j1] >= 0).int() + (subset[j2] >= 0).int())[:-2]
                        if torch.count_nonzero(membership == 2) == 0:
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += all_connections[k][i][2]
                            subset = torch.cat((subset[:j2], subset[j2 + 1:]))
                        else:
                            subset[j1][index_B] = part_Bs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[part_Bs[i].int(), 2] + all_connections[k][i][2]
                    elif not found and k < 17:
                        row = torch.ones(20, device=self.device) * -1
                        row[index_A] = part_As[i]
                        row[index_B] = part_Bs[i]
                        row[-1] = 2
                        row[-2] = torch.sum(candidate[all_connections[k][i, :2].long(), 2]) + all_connections[k][i][2]
                        subset = torch.cat([subset, row.unsqueeze(0)], dim=0)

        del_idx = []

        for i in range(len(subset)):
            if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
                del_idx.append(i)
        # Assuming del_idx is a list of indices to be deleted
        mask = torch.ones(len(subset), dtype=bool)
        mask[del_idx] = False
        subset = subset[mask]


        return _get_keypoints(candidate, subset)
