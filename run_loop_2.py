import os
import numpy as np
from multiprocessing import Pool


def one_run(zeta, teq, inc, sn = 20):

	outfile = "non_cv_for_sn_zeta%d_teq%d_inc%d_sn_%d" % (zeta, teq, inc, sn)

	zeta = zeta * np.pi/180.0
	teq = teq *np.pi /180.0
	inc = inc * np.pi / 180.0
	run_str = "python run_cv.py -root_f './result_non_cv_for_sn/' -f '%s' -inc %f -zeta %f -teq %f -sn %d -l1 -2 1 -ltsv -2 1" % (outfile, inc, zeta, teq, sn)
	print (run_str)
	os.system(run_str)

def wrapper_onerun(args):
    return one_run(*args)

arr = []


zeta_arr =np.array([60])
teq_arr = np.array([180])
inc_arr = np.array([30])
sn_arr = np.array([5,10,15,20])

for dmy in zeta_arr:
	for dmy1 in teq_arr:
		for dmy2 in inc_arr:
			for dmy3 in sn_arr:

				arr.append([dmy, dmy1, dmy2, dmy3])

p = Pool(processes=4)
p.map(wrapper_onerun, arr)