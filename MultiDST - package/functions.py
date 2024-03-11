from utils.weighting import weighted_p_list
from MultiDST.bonferroni import bonferroni
from MultiDST.holm import holm
from MultiDST.sgof import sgof_test
from MultiDST.BH import bh_method
from MultiDST.qval import q_value
from MultiDST.BY import BY_method

from utils.visualization import draw_histogram
from utils.visualization import sig_index_plot
from utils.visualization import draw_p_bar_chart
from utils.visualization import plot_heatmap 

import pandas as pd
import numpy as np
import random

from utils.weighting import weighted_p_list
from MultiDST.bonferroni import bonferroni
from MultiDST.holm import holm
from MultiDST.sgof import sgof_test
from MultiDST.BH import bh_method
from MultiDST.qval import q_value
from MultiDST.BY import BY_method

from utils.visualization import draw_histogram
from utils.visualization import sig_index_plot
from utils.visualization import draw_p_bar_chart
from utils.visualization import plot_heatmap 
from utils.visualization import fire_hist

from simulation_functions import DSTmulti_testing

import pandas as pd




def multi_DST(p_values, alpha=0.05, weights=False):
    # 0 - Uncorrected
    sig_index = [index for index, p in enumerate(p_values) if p < alpha]
    uncorrected_count = len(sig_index)
    print("Uncorrected count:", uncorrected_count)

    # 1 - Bonferroni
    bonf_results = bonferroni(p_values, alpha=alpha, weights=weights)
    bonf_p, sig_bonf_p = bonf_results[0], bonf_results[1]
    bonf_count = len(sig_bonf_p)
    print("Bonferroni count:", bonf_count)

    # 2 - Holm
    holm_results = holm(p_values, alpha=alpha, weights=weights)
    holm_p, sig_holm_p = holm_results[0], holm_results[1]
    holm_count = len(sig_holm_p)
    print("Holm count:", holm_count)

    # 3 - SGoF
    sgof_results = sgof_test(p_values, alpha=alpha, weights=weights)
    sgof_p, sig_sgof_p = sgof_results[0], sgof_results[1]
    sgof_count = len(sig_sgof_p)
    print("SGoF count:", sgof_count)

    # 4 - BH
    bh_results = bh_method(p_values, alpha=alpha, weights=weights)
    bh_p, sig_bh_p = bh_results[0], bh_results[1]
    bh_count = len(sig_bh_p)
    print("BH count:", bh_count)

    # 5 - BY
    by_results = BY_method(p_values, alpha=alpha, weights=weights)
    by_p, sig_by_p = by_results[0], by_results[1]
    by_count = len(sig_by_p)
    print("BY count:", by_count)

    # 6 - Qval
    q_results = q_value(p_values, alpha=alpha, weights=weights)
    q, sig_q,pi0_est = q_results[0], q_results[1],q_results[2]
    q_count = len(sig_q)
    print("Q-value count:", q_count)

    return {
        "Uncorrected": sig_index,
        "Bonferroni": sig_bonf_p,
        "Holm": sig_holm_p,
        "SGoF": sig_sgof_p,
        "BH": sig_bh_p,
        "BY": sig_by_p,
        "Q-value": sig_q,
        "pi0 estimate": pi0_est
    }

p_values = [random.uniform(0,0.05) for i in range(10)]
multi_DST(p_values, alpha=0.05, weights=False)



