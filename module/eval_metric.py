import re
import string
from collections import Counter
from statistics import mean

import editdistance as ed
from nlgeval import NLGEval


def _normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        if len(text) > 1:
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        else:
            return text

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _f1_score(prediction, ground_truth):
    if len(prediction) == 0:
        return 0
    prediction_tokens = _normalize_answer(prediction).split()
    ground_truth_tokens = _normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def cer_cal(groundtruth, hypothesis):
    err = 0
    tot = 0
    for p, t in zip(hypothesis, groundtruth):
        err += float(ed.eval(p.lower(), t.lower()))
        tot += len(t)
    return err / tot


def wer_cal(groundtruth, hypothesis):
    err = 0
    tot = 0
    for p, t in zip(hypothesis, groundtruth):
        p = p.lower().split(' ')
        t = t.lower().split(' ')
        err += float(ed.eval(p, t))
        tot += len(t)
    return err / tot


nlgeval = NLGEval(no_skipthoughts=True, no_glove=True, metrics_to_omit=["METEOR", "CIDEr"])


def compute_metrics_fn(predictions, labels):
    result_dict = {}
    print("pred_result")
    print("=================================")
    for i in range(10):
        print("target:" + labels[i])
        print("pred:" + predictions[i])
        print("-----------------")
    print("=================================")
    # cal em f1
    em_list = []
    f1_list = []
    for target, predict in zip(labels, predictions):
        if len(str(predict)) > 0 and \
                _normalize_answer(str(predict)) == _normalize_answer(str(target)) and \
                len(_normalize_answer(str(predict))) > 0 or len(str(predict)) == len(str(target)) == 0:
            em_score = 1
            f1_score = 1
        else:
            em_score = 0
            f1_score = _f1_score(str(predict), str(target))
        em_list.append(em_score)
        f1_list.append(f1_score)
    result_dict.update({"em": mean(em_list), "f1": mean(f1_list)})

    # cal wer/cer
    cer = cer_cal(labels, predictions)
    wer = wer_cal(labels, predictions)
    result_dict.update({"cer": cer, "wer": wer})

    # cal bleu rouge score
    result_dict.update(nlgeval.compute_metrics(ref_list=list(map(list, zip(*labels))), hyp_list=predictions))
    return result_dict
