# nuScenes dev-kit.
# Code written by Oscar Beijbom, 2019.

import re
from typing import Callable

import numpy as np

from nuscenes.eval.common.data_classes import EvalBoxes, EvalBox
from nuscenes.eval.common.utils import center_distance, scale_iou, yaw_diff, velocity_l2, attr_acc, cummean, ade, fde, miss_rate
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionMetrics, DetectionBox, DetectionMetricDataList, DetectionMetricData

from tqdm import tqdm 
import copy
import pdb

def match(gt, pred, class_name):
    if gt == class_name:
        return True 

    if gt == "ignore" and pred == class_name:
        return True 

    return False 

match_idx = {0.5: 0, 1: 1, 2 : 2, 4: 3}
car_forecast_match = [[0.5, 1, 2, 4],
                  [0.58, 1.17, 2.53, 4.67],
                  [0.66, 1.33, 2.67, 5.33],
                  [0.75, 1.5, 3, 6],
                  [0.83, 1.67, 3.33, 6.67],
                  [0.92, 1.83, 3.67, 7.33],
                  [1, 2, 4, 8]
                  ]
#pedestrian_forecast_match = [[0.25, .5, 1, 2],
#                      [0.29, 0.58, 1.17, 2.33],
#                      [0.33, 0.67, 1.33, 2.67],
#                      [0.38, 0.75, 1.5, 3],
#                      [0.42, 0.83, 1.67, 3.33],
#                      [0.46, 0.92, 1.83, 3.67],
#                      [0.5, 1, 2, 4]
#                    ]

pedestrian_forecast_match = [[0.13, .25, 0.5, 1],
                      [0.15, 0.29, 0.58, 1.17],
                      [0.17, 0.33, 0.67, 1.33],
                      [0.19, 0.48, 0.75, 1.5],
                      [0.21, 0.42, 0.83, 1.67],
                      [0.23, 0.46, 0.92, 1.83],
                      [0.25, 0.5, 1, 2]
                    ]

def accumulate(nusc, 
               gt_boxes: EvalBoxes,
               pred_boxes: EvalBoxes,
               class_name: str,
               dist_fcn: Callable,
               dist_th: float,
               forecast: int,
               K : int,
               cohort_analysis: bool = False,
               oracle: bool = False,
               verbose: bool = False) -> DetectionMetricData:
    """
    Average Precision over predefined different recall thresholds for a single distance threshold.
    The recall/conf thresholds and other raw metrics will be used in secondary metrics.
    :param gt_boxes: Maps every sample_token to a list of its sample_annotations.
    :param pred_boxes: Maps every sample_token to a list of its sample_results.
    :param class_name: Class to compute AP on.
    :param dist_fcn: Distance function used to match detections and ground truths.
    :param dist_th: Distance threshold for a match.
    :param verbose: If true, print debug messages.
    :return: (average_prec, metrics). The average precision value and raw data for a number of metrics.
    """
    # ---------------------------------------------
    # Organize input and initialize accumulators.
    # ---------------------------------------------

    # Count the positives.
    npos = len([1 for gt_box in gt_boxes.all if gt_box.detection_name == class_name])
    if verbose:
        print("Found {} GT of class {} out of {} total across {} samples.".
              format(npos, class_name, len(gt_boxes.all), len(gt_boxes.sample_tokens)))

    # For missing classes in the GT, return a data structure corresponding to no predictions.

    # Organize the predictions in a single list.

    if cohort_analysis:
        classname = "car" if "car" in class_name else "pedestrian"
    else:
        classname = class_name


    if "car" in classname:
        forecast_match = car_forecast_match
    else:
        forecast_match = pedestrian_forecast_match

    pred_boxes_list = [box for box in pred_boxes.all if classname in box.detection_name]
    pred_confs = [box.detection_score for box in pred_boxes_list]

    if npos == 0:
        return DetectionMetricData.no_predictions(timesteps=forecast)

    if verbose:
        print("Found {} PRED of class {} out of {} total across {} samples.".
              format(len(pred_confs), class_name, len(pred_boxes.all), len(pred_boxes.sample_tokens)))

    # Sort by confidence.
    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

    midx = match_idx[dist_th]
    # Do the actual matching.
    tp = []  # Accumulator of true positives.
    fp = []  # Accumulator of false positives.

    ftp = []
    ffp = []

    tp_mr = []  # Accumulator of true positives.
    fp_mr = []  # Accumulator of false positives.
    conf = []  # Accumulator of confidences.

    gt_name = []
    pred_name = []

    gt_forecast_name = []
    pred_forecast_name = []

    for i in range(forecast):
        ftp.append([])
        ffp.append([])

        gt_forecast_name.append([])
        pred_forecast_name.append([])

    # match_data holds the extra metrics we calculate for each match.
    match_data = {'trans_err': [],
                  'vel_err': [],
                  'scale_err': [],
                  'orient_err': [],
                  'attr_err': [],
                  'avg_disp_err' : [],
                  'final_disp_err' : [],
                  'miss_rate' : [],
                  #'reverse_avg_disp_err' : [],
                  #'reverse_final_disp_err' : [],
                  #'reverse_miss_rate' : [],
                  'conf': []}

    # ---------------------------------------------
    # Match and accumulate match data.
    # ---------------------------------------------

    for i in range(forecast):
        taken = set()  # Initially no gt bounding box is matched.
        for ind in sortind:
            pred_box = pred_boxes_list[ind].forecast_boxes[i]

            if K == 0 and i == 0:
                min_dist = forecast * [np.inf]
            else:
                min_dist = np.inf
            
            match_gt_idx = None

            for gt_idx, gt_box in enumerate(gt_boxes[pred_boxes_list[ind].sample_token]):

                # Find closest match among ground truth boxes
                if classname in gt_box.detection_name and not (pred_boxes_list[ind].sample_token, gt_idx) in taken:
                    if K == 0 and i == 0:
                        this_distance = [dist_fcn(gt_box.forecast_boxes[t], pred_boxes_list[ind].forecast_boxes[t]) for t in range(forecast)]
                        if np.sum(this_distance) < np.sum(min_dist):
                            min_dist = this_distance
                            match_gt_idx = gt_idx
                    else:
                        this_distance = dist_fcn(gt_box.forecast_boxes[i], pred_box)
                    
                        if this_distance < min_dist:
                            min_dist = this_distance
                            match_gt_idx = gt_idx

            # If the closest match is close enough according to threshold we have a match!

            if oracle:
                is_match = min_dist < forecast_match[-1][midx]

            elif K == 0 and i == 0:
                is_match = all([min_dist[t] < forecast_match[t][midx] for t in range(forecast)])

            else:
                is_match = min_dist < dist_th

            if is_match:
                taken.add((pred_boxes_list[ind].sample_token, match_gt_idx))
                #  Update tp, fp and confs.
                ftp[i].append(1)
                ffp[i].append(0)

                gt_box_match = gt_boxes[pred_boxes_list[ind].sample_token][match_gt_idx]

                gt_forecast_name[i].append(gt_box_match.detection_name)
                pred_forecast_name[i].append("ignore")

                if i == 0:
                    mr = miss_rate(nusc, copy.deepcopy(gt_box_match.forecast_boxes[-1]), copy.deepcopy(pred_boxes_list[ind].forecast_boxes[-1]), forecast_match[-1][midx])

                    if mr == 0:
                        tp_mr.append(1)
                        fp_mr.append(0) 
                    else:
                        tp_mr.append(0)
                        fp_mr.append(1)  

                    tp.append(1)
                    fp.append(0)
                    conf.append(pred_boxes_list[ind].detection_score) 

                    # Since it is a match, update match data also.
                    
                    gt_name.append(gt_box_match.detection_name)
                    pred_name.append("ignore")

                    match_data['trans_err'].append(center_distance(gt_box_match, pred_boxes_list[ind]))
                    match_data['vel_err'].append(velocity_l2(gt_box_match, pred_boxes_list[ind]))
                    match_data['scale_err'].append(1 - scale_iou(gt_box_match, pred_boxes_list[ind]))

                    # Barrier orientation is only determined up to 180 degree. (For cones orientation is discarded later)
                    period = np.pi if class_name == 'barrier' else 2 * np.pi
                    match_data['orient_err'].append(yaw_diff(gt_box_match, pred_boxes_list[ind], period=period))
                    match_data['attr_err'].append(1 - attr_acc(gt_box_match, pred_boxes_list[ind]))
                    
                    match_data['avg_disp_err'].append(ade(nusc, gt_box_match, pred_boxes_list[ind]))
                    match_data['final_disp_err'].append(fde(nusc, gt_box_match, pred_boxes_list[ind]))
                    match_data['miss_rate'].append(miss_rate(nusc, gt_box_match.forecast_boxes[-1], pred_boxes_list[ind].forecast_boxes[-1]))

                    #match_data['reverse_avg_disp_err'].append(ade(nusc, gt_box_match, pred_box, reverse=True))
                    #match_data['reverse_final_disp_err'].append(fde(nusc, gt_box_match, pred_box, reverse=True))
                    #match_data['reverse_miss_rate'].append(miss_rate(nusc, gt_box_match, pred_box, reverse=True))

                    match_data['conf'].append(pred_boxes_list[ind].detection_score)

            else:                
                # No match. Mark this as a false positive.
                ftp[i].append(0)
                ffp[i].append(1)   

                gt_forecast_name[i].append("ignore")
                pred_forecast_name[i].append(pred_box.detection_name)

                if i == 0:
                    tp_mr.append(0)
                    fp_mr.append(1)

                    gt_name.append("ignore")
                    pred_name.append(pred_box.detection_name)

                    tp.append(0)
                    fp.append(1)
          
                    conf.append(pred_boxes_list[ind].detection_score)

    # Check if we have any matches. If not, just return a "no predictions" array.
    if len(match_data['trans_err']) == 0:
        return DetectionMetricData.no_predictions(timesteps=forecast)

    # ---------------------------------------------
    # Calculate and interpolate precision and recall
    # ---------------------------------------------

    # Accumulate.
    if cohort_analysis:
        select = [match(gt, pred, class_name) for pred, gt in zip(pred_name, gt_name)]

        tp = np.array(tp)[select]
        fp = np.array(fp)[select]
        tp_mr = np.array(tp_mr)[select]
        fp_mr = np.array(fp_mr)[select]
        conf = np.array(conf)[select]

        select_forecast = [[match(gt, pred, class_name) for pred, gt in zip(pred_forecast_name[i], gt_forecast_name[i])] for i in range(len(ftp))]
        ftp = [np.array(tp)[select] for tp, select in zip(ftp, select_forecast)]
        ffp = [np.array(fp)[select] for fp, select in zip(ffp, select_forecast)]
 
        if len(tp) == 0:
            return DetectionMetricData.no_predictions()

    true_pos = np.sum(tp)
    tp = np.cumsum(tp).astype(np.float)
    fp = np.cumsum(fp).astype(np.float)
    tp_mr = np.cumsum(tp_mr).astype(np.float)
    fp_mr = np.cumsum(fp_mr).astype(np.float)
    conf = np.array(conf)

    true_fpos = [np.sum(ftp[i]) for i in range(len(ftp))]
    ftp = [np.cumsum(ftp[i]).astype(np.float) for i in range(len(ftp))]
    ffp = [np.cumsum(ffp[i]).astype(np.float) for i in range(len(ffp))]

    # Calculate precision and recall.
    prec = tp / (fp + tp)
    rec = tp / float(npos)

    prec_mr = tp_mr / (fp_mr + tp_mr) 
    rec_mr = tp_mr / float(npos)

    fprec = [ftp[i] / (ffp[i] + ftp[i]) for i in range(len(ftp))]
    frec = [ftp[i] / float(npos) for i in range(len(ftp))]

    try:
        rec_interp = np.linspace(0, 1, DetectionMetricData.nelem)  # 101 steps, from 0% to 100% recall.
        prec = np.interp(rec_interp, rec, prec, right=0)
        prec_mr = np.interp(rec_interp, rec_mr, prec_mr, right=0)
        fprec = [np.interp(rec_interp, frec[i], fprec[i], right=0) for i in range(len(fprec))]
        conf = np.interp(rec_interp, rec, conf, right=0)
    except:
        return DetectionMetricData.no_predictions()

    rec_val = float(true_pos) / npos
    rec_fval = [float(true_fpos[i]) / npos for i in range(len(true_fpos))]
    rec = rec_interp
    # ---------------------------------------------
    # Re-sample the match-data to match, prec, recall and conf.
    # ---------------------------------------------

    for key in match_data.keys():
        if key == "conf":
            continue  # Confidence is used as reference to align with fp and tp. So skip in this step.

        else:
            # For each match_data, we first calculate the accumulated mean.
            tmp = cummean(np.array(match_data[key]))

            # Then interpolate based on the confidences. (Note reversing since np.interp needs increasing arrays)
            match_data[key] = np.interp(conf[::-1], match_data['conf'][::-1], tmp[::-1])[::-1]

    # ---------------------------------------------
    # Done. Instantiate MetricData and return
    # ---------------------------------------------
    return DetectionMetricData(recall=rec,
                               precision=prec,
                               precision_mr=prec_mr,
                               forecast_precision=fprec,
                               confidence=conf,
                               trans_err=match_data['trans_err'],
                               vel_err=match_data['vel_err'],
                               scale_err=match_data['scale_err'],
                               orient_err=match_data['orient_err'],
                               attr_err=match_data['attr_err'],
                               avg_disp_err=match_data['avg_disp_err'],
                               final_disp_err=match_data['final_disp_err'],
                               miss_rate=match_data['miss_rate'],
                               rec_val=rec_val,
                               rec_fval=rec_fval,
                               #reverse_avg_disp_err=match_data['reverse_avg_disp_err'],
                               #reverse_final_disp_err=match_data['reverse_final_disp_err'],
                               #reverse_miss_rate=match_data['reverse_miss_rate'],
                               )


def calc_ap(md: DetectionMetricData, min_recall: float, min_precision: float) -> float:
    """ Calculated average precision. """

    assert 0 <= min_precision < 1
    assert 0 <= min_recall <= 1

    prec = np.copy(md.precision)
    prec = prec[round(100 * min_recall) + 1:]  # Clip low recalls. +1 to exclude the min recall bin.
    prec -= min_precision  # Clip low precision
    prec[prec < 0] = 0
    return float(np.mean(prec)) / (1.0 - min_precision)

def calc_fap_mr(md: DetectionMetricData, min_recall: float, min_precision: float) -> float:
    """ Calculated average precision. """

    assert 0 <= min_precision < 1
    assert 0 <= min_recall <= 1

    prec = np.copy(md.precision_mr)
    prec = prec[round(100 * min_recall) + 1:]  # Clip low recalls. +1 to exclude the min recall bin.
    prec -= min_precision  # Clip low precision
    prec[prec < 0] = 0

    return float(np.mean(prec)) / (1.0 - min_precision)


def calc_ar(md: DetectionMetricData) -> float:
    """ Calculated average recall. """

    return md.rec_val if md.rec_val is not None else 0


def calc_fap(md: DetectionMetricData, min_recall: float, min_precision: float) -> float:
    """ Calculated average foreast precision. """

    assert 0 <= min_precision < 1
    assert 0 <= min_recall <= 1

    prec = np.copy(md.forecast_precision[-1])
    prec = prec[round(100 * min_recall) + 1:]  # Clip low recalls. +1 to exclude the min recall bin.
    prec -= min_precision  # Clip low precision
    prec[prec < 0] = 0
    return float(np.mean(prec)) / (1.0 - min_precision)


def calc_far(md: DetectionMetricData) -> float:
    """ Calculated average recall. """

    return md.rec_fval[-1] if md.rec_fval is not None else 0

def calc_aap(md: DetectionMetricData, min_recall: float, min_precision: float) -> float:
    """ Calculated average foreast precision. """

    assert 0 <= min_precision < 1
    assert 0 <= min_recall <= 1
    
    ap = []
    for i in range(len(md.forecast_precision)):
        prec = np.copy(md.forecast_precision[i])
        prec = prec[round(100 * min_recall) + 1:]  # Clip low recalls. +1 to exclude the min recall bin.
        prec -= min_precision  # Clip low precision
        prec[prec < 0] = 0
        ap.append(float(np.mean(prec)) / (1.0 - min_precision))

    return np.mean(ap)


def calc_aar(md: DetectionMetricData) -> float:
    """ Calculated average recall. """

    if md.rec_fval is None:
        return 0
        
    ar = []
    for i in range(len(md.rec_fval)):
        ar.append(md.rec_fval[i])
    
    return np.mean(ar)

def calc_tp(md: DetectionMetricData, min_recall: float, metric_name: str, pct=-1) -> float:
    """ Calculates true positive errors. """

    first_ind = round(100 * min_recall) + 1  # +1 to exclude the error at min recall.
    last_ind = md.max_recall_ind  # First instance of confidence = 0 is index of max achieved recall.
    if last_ind < first_ind:
        return 1.0  # Assign 1 here. If this happens for all classes, the score for that TP metric will be 0.
    else:
        if pct == -1:
            return float(np.mean(getattr(md, metric_name)[first_ind: last_ind + 1]))  # +1 to include error at max recall.
        else:
            pct_ind = np.where(getattr(md, "recall") == pct)
            return float(getattr(md, metric_name)[pct_ind])
