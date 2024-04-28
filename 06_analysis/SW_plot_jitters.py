import matplotlib.pyplot as plt
import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from esme.control.pattern import get_beam_regions


def get_sa2_bunches():
    bp = np.load("./jitter-data/BUNCH_PATTERN.npz")["data"]

    brs = get_beam_regions(bp)
    times = brs[0].get_times()

    # bunches = np.where(brs[0].subpattern == 67666985)

    # SA2 bunches (..?)
    indices, = np.where(brs[0].subpattern > 6700000)

    sase2_times = times[indices]

    return sase2_times


def xjitter():
    # fig, ax = plt.subplots()

    sase2_times = get_sa2_bunches()

    result = []
    for f in glob.glob("./jitter-data/XFEL.DIAG_BPM_BPMI.1889.TL_X.TD_*.npz"):
        # from IPython import embed; embed()

        data = np.load(f)["data"]
        times = data[...,0]
        bpmb = data[...,1]


        bpm_sa2 = bpmb[np.searchsorted(times, sase2_times)]

        result.append(bpm_sa2)

    xbpmsa2 = np.vstack(result)

    return xbpmsa2

def yjitter():
    # fig, ax = plt.subplots()

    sase2_times = get_sa2_bunches()

    result = []
    for f in glob.glob("./jitter-data/XFEL.DIAG_BPM_BPMI.1889.TL_Y.TD_*.npz"):
        # from IPython import embed; embed()

        data = np.load(f)["data"]
        times = data[...,0]
        bpmb = data[...,1]


        bpm_sa2 = bpmb[np.searchsorted(times, sase2_times)]

        result.append(bpm_sa2)

    xbpmsa2 = np.vstack(result)

    return xbpmsa2

def ejitter():
    # fig, ax = plt.subplots()

    sase2_times = get_sa2_bunches()

    result = []
    for f in glob.glob("./jitter-data/XFEL.DIAG_BEAM_ENERGY_MEASUREMENT_CL_ENERGY_SLOW.TD_*.npz"):
        # from IPython import embed; embed()

        data = np.load(f)["data"]
        times = data[...,0]
        bpmb = data[...,1]

        
        bpm_sa2 = bpmb[np.searchsorted(times, sase2_times)]

        result.append(bpm_sa2)

    edata = np.vstack(result)

    return edata

def tjitter():
    # fig, ax = plt.subplots()

    sase2_times = get_sa2_bunches()

    result = []
    for f in glob.glob("./jitter-data/XFEL.SDIAG_BAM_1932M.TL_ARRIVAL_TIME.relative.TD_*.npz"):
        # from IPython import embed; embed()

        data = np.load(f)["data"]
        times = data[...,0]
        bpmb = data[...,1]


        bpm_sa2 = bpmb[np.searchsorted(times, sase2_times)]

        result.append(bpm_sa2)

    tdata = np.vstack(result)

    return tdata


    # from IPython import embed; embed()

        # data.append()

    # # x =  np.load("./jitter-data/XFEL.DIAG_BPM_BPMI.1889.TL_X.TD_2482.npz")["data"]

    # from IPython import embed; embed()


def main():
    x = xjitter()
    y = yjitter()
    e = ejitter()
    t = tjitter()

    fig, ax = plt.subplots()
    seconds = np.linspace(0, len(x) / 10.0, num=len(x))

    x = x[...,-1]
    y = y[...,-1]
    e = e[...,-1] / 1e3
    # t = t[...,-1]

    # from IPython import embed; embed()

    ax.plot(seconds, x, label="$x$")
    ax.plot(seconds, y, label="$y$")
    ax.legend()
    ax.set_xlabel("$t$ / s")
    ax.set_ylabel("BPM reading / mm")
    ax.set_title("BPMI.1889.TL at 10 Hz during 5 minutes of SA2 tuning")


    fig, ax = plt.subplots()
    ax.plot(seconds, e)
    ax.set_title("Slow (1 Hz) Energy Measurement in the Dogleg during 5 minutes of SA2 tuning")
    ax.set_xlabel("$t$ / s")
    ax.set_ylabel("$E$ / GeV")

    # fig, ax = plt.subplots()
    # ax.plot(seconds, t)

    plt.show()

    

    # from IPython import embed; embed()



# 0.75 + 14**2 / 0.75

if __name__ == '__main__':
    main()
