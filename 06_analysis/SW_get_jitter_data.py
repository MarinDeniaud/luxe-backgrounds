import esme.control.pattern as pt
import pandas as pd
import matplotlib.pyplot as plt
import pydoocs
import numpy as np

from esme.control.pattern import get_bunch_pattern, get_beam_regions, any_injector_laser

import time as t

XBPM = "XFEL.DIAG/BPM/BPMI.1889.TL/X.TD"
YBPM = "XFEL.DIAG/BPM/BPMI.1889.TL/Y.TD"
TIME = "XFEL.SDIAG/BAM/1932M.TL/ARRIVAL_TIME.relative.TD"
ENERGY = "XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/CL/ENERGY_SLOW.TD"



def get_data():
    # xbpm = []
    # ybpm = []
    # time = []
    # energy = []

    bp = pt.get_bunch_pattern()
    np.savez("BUNCH_PATTERN.npz", data=bp)

    for i in range(100_000):
        xbpm = pydoocs.read(XBPM)["data"]
        ybpm = pydoocs.read(YBPM)["data"]
        time = pydoocs.read(TIME)["data"]
        energy = pydoocs.read(ENERGY)["data"]

        t.sleep(0.1)

        np.savez(XBPM.replace("/", "_") + f"_{i}.npz", data=xbpm)
        np.savez(YBPM.replace("/", "_") + f"_{i}.npz", data=ybpm)
        np.savez(TIME.replace("/", "_") + f"_{i}.npz", data=time)
        np.savez(ENERGY.replace("/", "_") + f"_{i}.npz", data=energy)

        print(i)


        


# def get_bpm_data(dimension="x"):
#     if dimension == "x":
#         addy = XBPM
#     bpm = pydoocs.read(addy)["data"]

#     time, pos = bpm[...,0], bpm[...,1]

#     br1, *_ = get_beam_regions(get_bunch_pattern())
#     br1_time = br1.get_times()

#     from IPython import embed; embed()

#     plt.plot(time, pos)
#     plt.show()



    


def main():
    data = get_data()


if __name__ == "__main__":
    main()
