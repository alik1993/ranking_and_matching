from math import log2

from torch import Tensor, sort


# ys_true = torch.rand(5)
# ys_pred = torch.rand(5)

# Tensor([0.2048, 0.3451, 0.3451, 0.2961, 0.4787, 0.8041])


# for i, el in enumerate(list,1): - индекс с 1
# return без else
# ideal dcg - dcg(true,true)
# len([v for v in ys_true_sort if v==1]) заменить на ys_true_sort.sum()

# me
def num_swapped_pairs(ys_true: Tensor, ys_pred: Tensor) -> int:
    
    ys_pred_sort, sort_indices = sort(ys_pred, descending=True, dim=0)
    ys_true_sort = ys_true[sort_indices]
    cnt = 0
    for x_ix, x in enumerate(ys_true_sort):
        for y_ix, y in enumerate(ys_true_sort):
            if bool(y_ix>x_ix) & bool(y>x):
                cnt+=1
    return cnt

# ideal
def num_swapped_pairs(ys_true: torch.Tensor, ys_pred: torch.Tensor) -> int:
    ys_pred_sorted, argsort = torch.sort(ys_pred, descending=True, dim=0)
    ys_true_sorted = ys_true[argsort]
    
    num_objects = ys_true_sorted.shape[0]
    swapped_cnt = 0
    for cur_obj in range(num_objects - 1):
        for next_obj in range(cur_obj + 1, num_objects):
            if ys_true_sorted[cur_obj] < ys_true_sorted[next_obj]:
                if ys_pred_sorted[cur_obj] > ys_pred_sorted[next_obj]:
                    swapped_cnt += 1
            elif ys_true_sorted[cur_obj] > ys_true_sorted[next_obj]:
                if ys_pred_sorted[cur_obj] < ys_pred_sorted[next_obj]:
                    swapped_cnt += 1
    return swapped_cnt

# me
def compute_gain(y_value: float, gain_scheme: str) -> float:
    if gain_scheme=="const":
        return y_value
    elif gain_scheme=="exp2":
        # return 2**y_value - 1
        return pow(2, y_value) - 1
    else:
        return None

# ideal
def compute_gain(y_value: float, gain_scheme: str) -> float:
    if gain_scheme == "exp2":
        gain = 2 ** y_value - 1
    elif gain_scheme == "const":
        gain = y_value
    else:
        raise ValueError(f"{gain_scheme} method not supported, only exp2 and const.")
    return float(gain)


# me
def dcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str) -> float:
    ys_pred_sort, sort_indices = sort(ys_pred, descending=True, dim=0)
    ys_true_sort = ys_true[sort_indices] 
    dcg_ = 0
    for ix, e in enumerate(ys_true_sort):
        dcg_ += compute_gain(float(e), gain_scheme)/log2(ix+2)
        
    return dcg_

# ideal
def dcg(ys_true: torch.Tensor, ys_pred: torch.Tensor, gain_scheme: str) -> float:
    _, argsort = torch.sort(ys_pred, descending=True, dim=0)
    ys_true_sorted = ys_true[argsort]
    ret = 0
    for idx, cur_y in enumerate(ys_true_sorted, 1):
        gain = compute_gain(cur_y, gain_scheme)
        ret += gain / math.log2(idx + 1)
    return ret


# me
def ndcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str = 'const') -> float:
    ys_ideal_true_sort, _ = sort(ys_true, descending=True, dim=0)
    
    ideal_dcg = 0
    for ix, e in enumerate(ys_ideal_true_sort):
        ideal_dcg += compute_gain(float(e), gain_scheme)/log2(ix+2)

    dcg_ = dcg(ys_true, ys_pred, gain_scheme)
    
    return dcg_/ideal_dcg


# ideal
def ndcg(ys_true: torch.Tensor, ys_pred: torch.Tensor, gain_scheme: str = 'const') -> float:
    pred_dcg = dcg(ys_true, ys_pred, gain_scheme)
    ideal_dcg = dcg(ys_true, ys_true, gain_scheme)
    
    ndcg = pred_dcg / ideal_dcg
    return ndcg


# me
def compute_ideal_dcg(ys_true: Tensor, ndcg_scheme: str = 'exp2') -> float:
    ys_ideal_true_sort, _ = sort(ys_true, descending=True, dim=0)
    
    ideal_dcg = 0
    for ix, e in enumerate(ys_ideal_true_sort):
        ideal_dcg += compute_gain(float(e), ndcg_scheme)/log2(ix+2)
    
    return ideal_dcg


# me
def precission_at_k(ys_true: Tensor, ys_pred: Tensor, k: int) -> float:
    
    ys_pred_sort, sort_indices = sort(ys_pred, descending=True, dim=0)
    ys_true_sort = ys_true[sort_indices]

    ones_cnt_init = len([v for v in ys_true_sort if v==1])
    
    if k > len(ys_true_sort):
        k = len(ys_true_sort)
        
    ys_true_sort_topk = ys_true_sort[:k]
    
    ones_cnt = len([v for v in ys_true_sort_topk if v==1])

    if ones_cnt_init>0:
        if ones_cnt_init < k:
            return ones_cnt/ones_cnt_init
        else:
            return ones_cnt/k
    else:
        return -1

# ideal   
def precission_at_k(ys_true: torch.Tensor, ys_pred: torch.Tensor, k: int) -> float:
    if ys_true.sum() == 0:
        return -1
    _, argsort = torch.sort(ys_pred, descending=True, dim=0)
    ys_true_sorted = ys_true[argsort]
    hits = ys_true_sorted[:k].sum()
    prec = hits / min(ys_true.sum(), k)
    return float(prec)


# me
def reciprocal_rank(ys_true: Tensor, ys_pred: Tensor) -> float:
    ys_pred_sort, sort_indices = sort(ys_pred, descending=True, dim=0)
    ys_true_sort = ys_true[sort_indices]
    ones_cnt = len([v for v in ys_true_sort if v==1])
    if ones_cnt>0:
        one_idx = float((ys_true_sort==1).nonzero(as_tuple=False)[0])
        return 1/(one_idx+1)
    else:
        return 0

# ideal   
def reciprocal_rank(ys_true: torch.Tensor, ys_pred: torch.Tensor) -> float:
    _, argsort = torch.sort(ys_pred, descending=True, dim=0)
    ys_true_sorted = ys_true[argsort]
    
    for idx, cur_y in enumerate(ys_true_sorted, 1):
        if cur_y == 1:
            return 1 / idx
    return 0


# me
def p_found(ys_true: Tensor, ys_pred: Tensor, p_break: float = 0.15 ) -> float:
    ys_pred_sort, sort_indices = sort(ys_pred, descending=True, dim=0)
    ys_true_sort = ys_true[sort_indices]
    
    pfound_total = 0
    
    for ix, el in enumerate(ys_true_sort):
        if ix==0:
            p_look = 1
            p_f = p_look * ys_true_sort[ix]
        else: 
            p_look = p_look*(1-ys_true_sort[ix-1])*(1-p_break)
            p_f = p_look * ys_true_sort[ix]
        
        pfound_total += p_f
        
    return pfound_total

# ideal
def p_found(ys_true: torch.Tensor, ys_pred: torch.Tensor, p_break: float = 0.15 ) -> float:
    p_look = 1
    p_found = 0
    _, argsort = torch.sort(ys_pred, descending=True, dim=0)
    ys_true_sorted = ys_true[argsort]

    for cur_y in ys_true_sorted:
        p_found += p_look * float(cur_y)
        p_look = p_look * (1 - float(cur_y)) * (1 - p_break)
    
    return p_found


# me
def average_precision(ys_true: Tensor, ys_pred: Tensor) -> float:
    
    ys_pred_sort, sort_indices = sort(ys_pred, descending=True, dim=0)
    ys_true_sort = ys_true[sort_indices]
    
    ones_cnt = len([v for v in ys_true_sort if v==1])
    if ones_cnt>0:
        ap = 0
        for ix, el in enumerate(ys_true_sort):
            if el==1:
                ys_true_sort_filt = ys_true_sort[:ix+1]
                rec = len([v for v in ys_true_sort_filt if v==1])/len(ys_true_sort_filt)
                ap += rec
        return ap/ones_cnt
    else:
        return -1

# ideal
def average_precision(ys_true: torch.Tensor, ys_pred: torch.Tensor) -> float:
    if ys_true.sum() == 0:
        return -1
    _, argsort = torch.sort(ys_pred, descending=True, dim=0)
    ys_true_sorted = ys_true[argsort]
    rolling_sum = 0
    num_correct_ans = 0
    
    for idx, cur_y in enumerate(ys_true_sorted, start=1):
        if cur_y == 1:
            num_correct_ans += 1
            rolling_sum += num_correct_ans / idx
    if num_correct_ans == 0:
        return 0
    else:
        return rolling_sum / num_correct_ans

