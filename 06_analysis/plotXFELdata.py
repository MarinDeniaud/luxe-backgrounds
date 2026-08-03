import numpy as _np
import matplotlib.pyplot as _plt
import matplotlib.patches as _patch
from pdf2image import convert_from_path

import pymad8 as _m8

import XFEL_BPM
import loadXFELdata


_plt.rcParams['font.size'] = 15


def addLattice(lattice, s, margin=1.5):
    twiss = _m8.Output(lattice)
    _m8.Plot.AddMachineLatticeToFigure(_plt.gcf(), twiss)
    _plt.xlim(min(s)-margin, max(s)+margin)


def plotSignalJitterNoisePerPlane(xfel_data, bpms=None, trains=None, bunches=None,
                                  lattice=None, figsize=[12, 8]):
    df_reduced = xfel_data.reduceDFbyBPMTrainBunchByIndex(bpms=bpms, trains=trains, bunches=bunches)
    s = xfel_data.getS(df_reduced)
    _plt.subplots(2, 1, height_ratios=[1, 1], sharex=True, figsize=figsize)
    _plt.subplot(211)
    plotSignalJitterNoiseX(xfel_data, df_reduced, inSubplots=True, lattice=None)
    _plt.subplot(212)
    plotSignalJitterNoiseY(xfel_data, df_reduced, inSubplots=True, lattice=None)
    _plt.xlabel(r"$s$ [m]")
    if lattice is not None:
        addLattice(lattice, s)


def plotSignalJitterNoiseX(xfel_data, df_reduced, inSubplots=False, lattice=None, figsize=[12, 5]):
    s = xfel_data.getS(df_reduced)
    sx, sy = xfel_data.calcSignal(df_reduced)
    jx, nx = xfel_data.calcJitterAndNoise(df_reduced, 'X')
    if not inSubplots:
        _plt.figure(figsize=figsize)
        _plt.xlabel(r"$s$ [m]")
    _plt.plot(s, sx * 1e6, '+-', color='C0', label=r"$\sigma_{S, x}$", markersize=10, markeredgewidth=1)
    _plt.plot(s, jx * 1e6, '--', color='C0', label=r"$\sigma_{J, x}$", markersize=10, markeredgewidth=1)
    _plt.plot(s, nx * 1e6, ':', color='C0', label=r"$\sigma_{N, x}$", markersize=10, markeredgewidth=1)
    _plt.ylabel(r"$\sigma_x$ [$\rm \mu m$]")
    _plt.legend()
    if lattice is not None:
        addLattice(lattice, s)


def plotSignalJitterNoiseY(xfel_data, df_reduced, inSubplots=False, lattice=None, figsize=[12, 5]):
    s = xfel_data.getS(df_reduced)
    sx, sy = xfel_data.calcSignal(df_reduced)
    jy, ny = xfel_data.calcJitterAndNoise(df_reduced, 'Y')
    if not inSubplots:
        _plt.figure(figsize=figsize)
        _plt.xlabel(r"$s$ [m]")
    _plt.plot(s, sy * 1e6, '+-', color='C1', label=r"$\sigma_{S, y}$", markersize=10, markeredgewidth=1)
    _plt.plot(s, jy * 1e6, '--', color='C1', label=r"$\sigma_{J, y}$", markersize=10, markeredgewidth=1)
    _plt.plot(s, ny * 1e6, ':', color='C1', label=r"$\sigma_{N, y}$", markersize=10, markeredgewidth=1)
    _plt.ylabel(r"$\sigma_y$ [$\rm \mu m$]")
    _plt.legend()
    if lattice is not None:
        addLattice(lattice, s)


def plotSignalJitterNoise(xfel_data, bpms=None, trains=None, bunches=None,
                          lattice=None, figsize=[12, 8]):
    df_reduced = xfel_data.reduceDFbyBPMTrainBunchByIndex(bpms=bpms, trains=trains, bunches=bunches)
    s = xfel_data.getS(df_reduced)
    _plt.subplots(3, 1, height_ratios=[1, 1, 1], sharex=True, figsize=figsize)
    _plt.subplot(311)
    plotSignal(xfel_data, df_reduced, inSubplots=True, lattice=None)
    _plt.subplot(312)
    plotJitter(xfel_data, df_reduced, inSubplots=True, lattice=None)
    _plt.subplot(313)
    plotNoise(xfel_data, df_reduced, inSubplots=True, lattice=None)
    _plt.xlabel(r"$s$ [m]")
    if lattice is not None:
        addLattice(lattice, s)


def plotSignal(xfel_data, df_reduced, inSubplots=False, lattice=None, figsize=[12, 5]):
    s = xfel_data.getS(df_reduced)
    sx, sy = xfel_data.calcSignal(df_reduced)
    if not inSubplots:
        _plt.figure(figsize=figsize)
        _plt.xlabel(r"$s$ [m]")
    _plt.plot(s, sx * 1e6, '+-', label=r"$\sigma_{S, x}$", markersize=10, markeredgewidth=1)
    _plt.plot(s, sy * 1e6, '+-', label=r"$\sigma_{S, y}$", markersize=10, markeredgewidth=1)
    _plt.ylabel(r"$\sigma_S$ [$\rm \mu m$]")
    _plt.legend()
    if lattice is not None:
        addLattice(lattice, s)


def plotJitter(xfel_data, df_reduced, inSubplots=False, lattice=None, figsize=[12, 5]):
    s = xfel_data.getS(df_reduced)
    jx, nx = xfel_data.calcJitterAndNoise(df_reduced, 'X')
    jy, ny = xfel_data.calcJitterAndNoise(df_reduced, 'Y')
    if not inSubplots:
        _plt.figure(figsize=figsize)
        _plt.xlabel(r"$s$ [m]")
    _plt.plot(s, jx * 1e6, '+-', label=r"$\sigma_{J,x}$", markersize=10, markeredgewidth=1)
    _plt.plot(s, jy * 1e6, '+-', label=r"$\sigma_{J,y}$", markersize=10, markeredgewidth=1)
    _plt.ylabel(r"$\sigma_J$ [$\rm \mu m$]")
    _plt.legend()
    if lattice is not None:
        addLattice(lattice, s)


def plotJitterNoiseCharge(xfel_data, df_reduced, lattice=None, figsize=[12, 5]):
    s = xfel_data.getS(df_reduced)
    jq, nq = xfel_data.calcJitterAndNoise(df_reduced, 'Charge')
    _plt.subplots(2, 1, figsize=figsize, sharex=True)
    _plt.subplot(211)
    _plt.plot(s, jq * 1e12, '+-', color='C2', label=r"$\sigma_{J,q}$", markersize=10, markeredgewidth=1)
    _plt.ylabel(r"$\sigma_J$ [pC]")
    _plt.legend()
    _plt.subplot(212)
    _plt.plot(s, nq * 1e12, '+-', color='C2', label=r"$\sigma_{N,q}$", markersize=10, markeredgewidth=1)
    _plt.ylabel(r"$\sigma_N$ [pC]")
    _plt.xlabel(r"$s$ [m]")
    _plt.legend()
    if lattice is not None:
        addLattice(lattice, s)


def plotJitterSigmaRatio(xfel_data, df_reduced, inSubplots=False, lattice=None, figsize=[12, 5]):
    jx, nx = xfel_data.calcJitterAndNoise(df_reduced, 'X')
    jy, ny = xfel_data.calcJitterAndNoise(df_reduced, 'Y')
    s, sx, sy = xfel_data.calcJitterSigmaRatio(df_reduced, jx, jy)
    if not inSubplots:
        _plt.figure(figsize=figsize)
        _plt.xlabel(r"$s$ [m]")
    _plt.plot(s, sx, '+-', label=r"$\frac{\sigma_{J,x}}{\sigma_x}$", markersize=10, markeredgewidth=1)
    _plt.plot(s, sy, '+-', label=r"$\frac{\sigma_{J,y}}{\sigma_y}$", markersize=10, markeredgewidth=1)
    _plt.axhline(5, ls='--', color='C3')
    _plt.text(min(s)-3, 4, "5%", color='C3', ha='right', va='center')
    _plt.ylabel(r"$\frac{\sigma_{J}}{\sigma}$ [%]")
    _plt.legend()
    if lattice is not None:
        addLattice(lattice, s)


def plotNoise(xfel_data, df_reduced, bpmType=False, inSubplots=False, lattice=None,
              ylim=None, legendpos=[0, 0], figsize=[12, 5]):
    s = xfel_data.getS(df_reduced)
    jx, nx = xfel_data.calcJitterAndNoise(df_reduced, 'X')
    jy, ny = xfel_data.calcJitterAndNoise(df_reduced, 'Y')
    if not inSubplots:
        _plt.figure(figsize=figsize)
        _plt.xlabel(r"$s$ [m]")
    if bpmType:
        bpm_names = df_reduced.index.get_level_values('BPM').unique()
        th = 1
        label1 = r"Button BPM ($R_{\rm pipe}=40$ mm)"
        label2 = r"Button BPM ($R_{\rm pipe}=100$ mm)"
        label3 = "Cavity BPM"
        for bpm_name, s_bpm in zip(bpm_names, s):
            if 'BPMA' in bpm_name:
                # _plt.gca().axvspan(s_bpm - th / 2, s_bpm + th / 2, alpha=0.2, color='red', edgecolor='none', label=label1)
                _plt.gca().plot([s_bpm, s_bpm], [10, 12], alpha=0.6, color='red', label=label1)
                label1 = None
            elif 'BPMD' in bpm_name:
                # _plt.gca().axvspan(s_bpm - th / 2, s_bpm + th / 2, alpha=0.2, color='green', edgecolor='none', label=label2)
                _plt.gca().plot([s_bpm, s_bpm], [10, 12], alpha=0.6, color='green', label=label2)
                label2 = None
            else:
                # _plt.gca().axvspan(s_bpm - th / 2, s_bpm + th / 2, alpha=0.2, color='blue', edgecolor='none', label=label3)
                _plt.gca().plot([s_bpm, s_bpm], [10, 12], alpha=0.6, color='blue', label=label3)
                label3 = None
    _plt.plot(s, nx * 1e6, '+-', label=r"$\sigma_{N,x}$", markersize=10, markeredgewidth=1)
    _plt.plot(s, ny * 1e6, '+-', label=r"$\sigma_{N,y}$", markersize=10, markeredgewidth=1)
    _plt.ylabel(r"$\sigma_N$ [$\rm \mu m$]")
    _plt.ylim(ylim)
    _plt.legend(loc=6, bbox_to_anchor=(legendpos[0], legendpos[1]))
    if lattice is not None:
        addLattice(lattice, s)


def plotReconstructionBoth(xfel_data, bpms=None, trains=None, bunches=None, lattice=None, figsize=[12, 8]):
    df_reduced = xfel_data.reduceDFbyBPMTrainBunchByIndex(bpms=bpms, trains=trains, bunches=bunches)
    s = xfel_data.getS(df_reduced)
    _plt.subplots(2, 1, height_ratios=[2, 1], sharex=True, figsize=figsize)
    _plt.subplot(211)
    plotReconstructSignalAfterSVD(xfel_data, df_reduced, inSubplots=True, lattice=None)
    _plt.subplot(212)
    plotReconstructSignalSVDDifference(xfel_data, df_reduced, inSubplots=True, lattice=None)
    _plt.xlabel(r"$s$ [m]")
    if lattice is not None:
        addLattice(lattice, s)


def plotReconstructSignalAfterSVD(xfel_data, df_reduced, inSubplots=False, lattice=None, figsize=[12, 5]):
    s = xfel_data.getS(df_reduced)
    sx, sy = xfel_data.calcSignal(df_reduced)
    jx, nx = xfel_data.calcJitterAndNoise(df_reduced, 'X')
    jy, ny = xfel_data.calcJitterAndNoise(df_reduced, 'Y')
    if not inSubplots:
        _plt.figure(figsize=figsize)
        _plt.xlabel(r"$s$ [m]")
    _plt.plot(s, sx * 1e6, '-', label=r"$\sigma_{S,x}$", markersize=10, markeredgewidth=1)
    _plt.plot(s, sy * 1e6, '-', label=r"$\sigma_{S,y}$", markersize=10, markeredgewidth=1)
    _plt.plot(s, _np.sqrt(jx ** 2 + nx ** 2) * 1e6, '+', label=r"$\sqrt{\sigma_{J,x}^2+\sigma_{N,x}^2}$",
              markersize=10, markeredgewidth=1)
    _plt.plot(s, _np.sqrt(jy ** 2 + ny ** 2) * 1e6, '+', label=r"$\sqrt{\sigma_{J,y}^2+\sigma_{N,y}^2}$",
              markersize=10, markeredgewidth=1)
    _plt.ylabel(r"$\sigma$ [$\rm \mu m$]")
    _plt.legend()
    if lattice is not None:
        addLattice(lattice, s)


def plotReconstructSignalSVDDifference(xfel_data, df_reduced, inSubplots=False, lattice=None, figsize=[12, 5]):
    s = xfel_data.getS(df_reduced)
    sx, sy = xfel_data.calcSignal(df_reduced)
    jx, nx = xfel_data.calcJitterAndNoise(df_reduced, 'X', meanSub=True)
    jy, ny = xfel_data.calcJitterAndNoise(df_reduced, 'Y', meanSub=True)
    if not inSubplots:
        _plt.figure(figsize=figsize)
        _plt.xlabel(r"$s$ [m]")
    _plt.plot(s, _np.abs(sx - _np.sqrt(jx ** 2 + nx ** 2)) * 1e6, '+-', label=r"$\Delta{\sigma_x}$",
              markersize=10, markeredgewidth=1)
    _plt.plot(s, _np.abs(sy - _np.sqrt(jy ** 2 + ny ** 2)) * 1e6, '+-', label=r"$\Delta{\sigma_y}$",
              markersize=10, markeredgewidth=1)
    _plt.ylabel(r"$\sigma$ [$\rm \mu m$]")
    _plt.legend()
    _plt.semilogy()
    if lattice is not None:
        addLattice(lattice, s)


def plotAverageJitterPerBunchID(xfel_data, bpms=['BPMI.1910.TL', 'BPMI.1925.TL', 'BPMI.1930.TL', 'BPMI.1939.TL'],
                                sample=10, addBunchPattern=False, figsize=[12, 5]):
    bunchIDs, sx_mean, sy_mean = xfel_data.calcAverageJitterPerBunchID(bpms=bpms, sample=sample)
    if addBunchPattern:
        _plt.subplots(2, 1, height_ratios=[1, 4], sharex=True, figsize=figsize)
        _plt.subplot(211)
        ax = _plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        xfel_data.plotBunchPattern2(train=0, sample=49, inSubplot=True)
        _plt.subplot(212)
    else:
        _plt.figure(figsize=figsize)
    _plt.xlabel('Bunch ID')
    _plt.plot(bunchIDs, sx_mean, color='C0', label=r"$\frac{\sigma_{J,x}}{\sigma_x}$")
    _plt.plot(bunchIDs, sy_mean, color='C1', label=r"$\frac{\sigma_{J,y}}{\sigma_y}$")
    _plt.ylabel(r"$\frac{\sigma_{J}}{\sigma}$ [%]")
    _plt.legend(loc=9)
    _plt.grid()
    _plt.tight_layout()


def plotAverageChargeJitterPerBunchID(xfel_data, bpms=['BPMI.1910.TL', 'BPMI.1925.TL', 'BPMI.1930.TL', 'BPMI.1939.TL'],
                                      sample=10, addBunchPattern=False, figsize=[12, 5]):
    bunchIDs, jq_mean = xfel_data.calcAverageChargeJitterPerBunchID(bpms=bpms, sample=sample)
    if addBunchPattern:
        _plt.subplots(2, 1, height_ratios=[1, 4], sharex=True, figsize=figsize)
        _plt.subplot(211)
        ax = _plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        xfel_data.plotBunchPattern2(train=0, sample=49, inSubplot=True)
        _plt.subplot(212)
    else:
        _plt.figure(figsize=figsize)
    _plt.xlabel('Bunch ID')
    _plt.plot(bunchIDs, _np.array(jq_mean)*1e12, color='C2', label=r"$\sigma_{J,q}$")
    _plt.ylabel(r"$\sigma_{J,q}$ [pC]")
    _plt.legend(loc=9)
    _plt.grid()
    _plt.tight_layout()


def plotOneHist(ax, df_reduced, coord='X', label="x", unit=r"$\rm \mu m$", minCut=None, maxCut=None, nbins=50):
    X_cut = df_reduced[coord][minCut:maxCut]
    X_cut = X_cut - _np.mean(X_cut)
    ax.hist(X_cut, bins=nbins, histtype='step', color='C0',
            label=r"$\sigma_{{s,{}}}$ = {:.2f} {}".format(label, _np.std(X_cut), unit))
    ax.set_ylabel("counts")
    ax.set_xlabel("{} [{}]".format(label, unit))
    ax.legend()


def plotTwoHist(ax, df_reduced, coords=['X', 'Y'], labels=['x', 'y'], unit=r"$\rm \mu m$", minCut=None, maxCut=None, nbins=50):
    for coord, label in zip(coords, labels):
        plotOneHist(ax, df_reduced, coord, label, unit, minCut, maxCut, nbins)


def plotOneAlong(ax, df_reduced, slist, xlabel, coord='X', label='x', unit=r"$\rm \mu m$", minCut=None, maxCut=None):
    X = df_reduced[coord]
    X_cut = df_reduced[coord][minCut:maxCut]
    ax.plot(slist, X, color='gray')
    ax.plot(slist[minCut:maxCut], X_cut, color='C0',
            label=r"$\langle {} \rangle$ = {:.2f} {}".format(label, _np.mean(X_cut), unit))
    ax.set_ylabel(r"${}$ [{}]".format(label, unit))
    ax.set_xlabel(xlabel)
    ax.legend()


def PlotHist(ax, df_reduced, minCut=None, maxCut=None, nbins=50):
    X_cut = df_reduced['X'][minCut:maxCut] * 1e6
    Y_cut = df_reduced['Y'][minCut:maxCut] * 1e6
    X_cut = X_cut - _np.mean(X_cut)
    Y_cut = Y_cut - _np.mean(Y_cut)
    ax.hist(X_cut, bins=nbins, histtype='step', color='C0', label=r"$\sigma_{{s,x}}$ = {:.2f} $\rm \mu m$".format(_np.std(X_cut)))
    ax.hist(Y_cut, bins=nbins, histtype='step', color='C1', label=r"$\sigma_{{s,y}}$ = {:.2f} $\rm \mu m$".format(_np.std(Y_cut)))


def PlotAlong(ax_x, ax_y, df_reduced, bpms=None, trains=None, bunches=None, minCut=None, maxCut=None):
    X = df_reduced['X'] * 1e6
    Y = df_reduced['Y'] * 1e6
    X_cut = df_reduced['X'][minCut:maxCut] * 1e6
    Y_cut = df_reduced['Y'][minCut:maxCut] * 1e6
    # TODO : REMOVE PRINTS
    print(min(X_cut), max(X_cut))
    print(min(Y_cut), max(Y_cut))
    if bpms is None and trains is not None and bunches is not None:
        slist = df_reduced.index.get_level_values('BPM')
    elif bpms is not None and trains is None and bunches is not None:
        slist = df_reduced.index.get_level_values('TrainID')
        slist -= slist[0]
    elif bpms is not None and trains is not None and bunches is None:
        slist = df_reduced.index.get_level_values('BunchID')
    else:
        raise ValueError("Must use exactly 2 out of 3 between bpms, trains and bunches")
    ax_x.plot(slist, X, color='gray')
    ax_y.plot(slist, Y, color='gray')
    ax_x.plot(slist[minCut:maxCut], X_cut, color='C0', label=r"$\langle x \rangle$ = {:.2f} $\rm \mu m$".format(_np.mean(X_cut)))
    ax_y.plot(slist[minCut:maxCut], Y_cut, color='C1', label=r"$\langle y \rangle$ = {:.2f} $\rm \mu m$".format(_np.mean(Y_cut)))


def PlotAll(df, bpms=None, trains=None, bunches=None,
            minCut=None, maxCut=None, nbins=50,
            drawing=None, arrow_xy=[425, 30, 0, 170],
            hist_ylim=None, hist_xlim=None,
            figsize=[9, 6]):
    df_reduced = XFEL_BPM.reduceDFbyBPMTrainBunchByIndex(df, bpms=bpms, trains=trains, bunches=bunches)
    if bpms is None and trains is not None and bunches is not None:
        Xlabel = 'BPMs'
    elif bpms is not None and trains is None and bunches is not None:
        Xlabel = 'Train ID'
    elif bpms is not None and trains is not None and bunches is None:
        Xlabel = 'Bunch ID'
    else:
        raise ValueError("Must use exactly 2 out of 3 between bpms, trains and bunches")

    fig = _plt.figure(figsize=figsize)

    if drawing is not None:
        img = convert_from_path(drawing, dpi=300)[0]
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 3])
        ax_pdf = fig.add_axes([0, 0.75, 1, 0.2])
        ax_pdf.imshow(img)
        ax_pdf.axis('off')
        arrow = _patch.Rectangle((arrow_xy[0], arrow_xy[1]), arrow_xy[2], arrow_xy[3], lw=1.8, fill=False, color='r')
        ax_pdf.add_patch(arrow)
        ax_pdf.text(arrow_xy[0]+arrow_xy[2]/2, arrow_xy[1]-9, bpms, color='r', ha='center', va='bottom', fontsize=9)
        left_gs = gs[1, 0].subgridspec(2, 1)
        ax_left1 = fig.add_subplot(left_gs[0, 0])
        ax_left2 = fig.add_subplot(left_gs[1, 0], sharex=ax_left1)
        ax_left1.tick_params(labelbottom=False)
        ax_right = fig.add_subplot(gs[1, 1])

    else:
        gs = fig.add_gridspec(1, 2)
        left_gs = gs[0, 0].subgridspec(2, 1)
        ax_left1 = fig.add_subplot(left_gs[0, 0])
        ax_left2 = fig.add_subplot(left_gs[1, 0], sharex=ax_left1)
        ax_left1.tick_params(labelbottom=False)
        ax_right = fig.add_subplot(gs[0, 1])

    PlotAlong(ax_left1, ax_left2, df_reduced, bpms=bpms, trains=trains, bunches=bunches, minCut=minCut, maxCut=maxCut)
    ax_left1.set_ylabel(r"$x$ [$\rm \mu m$]")
    ax_left1.legend()
    ax_left2.set_xlabel(Xlabel)
    ax_left2.set_ylabel(r"$y$ [$\rm \mu m$]")
    ax_left2.legend()

    PlotHist(ax_right, df_reduced, minCut=minCut, maxCut=maxCut, nbins=nbins)
    ax_right.set_xlabel(r"$x,y$ [$\rm \mu m$]")
    ax_right.set_ylabel("counts")
    ax_right.set_ylim(hist_ylim)
    ax_right.set_xlim(hist_xlim)
    ax_right.legend(loc=1)

    _plt.tight_layout()


def plotTime(df, bpms="BAM", trains=None, bunches=None, nbins=50,
            drawing=None, arrow_xy=[425, 30, 0, 170],
            hist_ylim=None, hist_xlim=None, minCut=None, maxCut=None, figsize=[9, 6]):
    df_reduced = XFEL_BPM.reduceDFbyBPMTrainBunchByIndex(df, bpms="BPMA.1873.TL", trains=trains, bunches=bunches)
    if bpms is None and trains is not None and bunches is not None:
        xlabel = 'BPMs'
        slist = df_reduced.index.get_level_values('BPM')
    elif bpms is not None and trains is None and bunches is not None:
        xlabel = 'Train ID'
        slist = df_reduced.index.get_level_values('TrainID')
        slist -= slist[0]
    elif bpms is not None and trains is not None and bunches is None:
        xlabel = 'Bunch ID'
        slist = df_reduced.index.get_level_values('BunchID')
    else:
        raise ValueError("Must use exactly 2 out of 3 between bpms, trains and bunches")

    df_reduced.TimeS = df_reduced.TimeS*1e12
    fig = _plt.figure(figsize=figsize)

    if drawing is not None:
        img = convert_from_path(drawing, dpi=300)[0]
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 3])
        ax_pdf = fig.add_axes([0, 0.75, 1, 0.2])
        ax_pdf.imshow(img)
        ax_pdf.axis('off')
        arrow = _patch.Rectangle((arrow_xy[0], arrow_xy[1]), arrow_xy[2], arrow_xy[3], lw=1.8, fill=False, color='r')
        ax_pdf.add_patch(arrow)
        ax_pdf.text(arrow_xy[0]+arrow_xy[2]/2, arrow_xy[1]-9, bpms, color='r', ha='center', va='bottom', fontsize=9)
        ax_left = fig.add_subplot(gs[1, 0])
        ax_right = fig.add_subplot(gs[1, 1])

    else:
        gs = fig.add_gridspec(1, 2)
        ax_left = fig.add_subplot(gs[0, 0])
        ax_right = fig.add_subplot(gs[0, 1])

    plotOneAlong(ax_left, df_reduced, slist, xlabel, coord='TimeS', label="t", unit='ps', minCut=minCut, maxCut=maxCut)
    plotOneHist(ax_right, df_reduced, coord='TimeS', label="t", unit='ps', minCut=minCut, maxCut=maxCut, nbins=nbins)
    ax_right.set_ylim(hist_ylim)
    ax_right.set_xlim(hist_xlim)
    _plt.tight_layout()


def plotBunchProfile(crisp_data, train=0, figsize=[10, 8]):
    _plt.figure(figsize=figsize)
    time = crisp_data.df.Time
    for colname in crisp_data.df.columns[train]:
        _plt.plot(time, crisp_data.df[colname], label=crisp_data.df[colname].name)
    _plt.ylabel('Current')
    _plt.xlabel('$t$ [fs]')
    _plt.legend()


def plotBunchProfileAndCumsum(crisp_data, train=0, percentile=0.95, xlim=None, figsize=[10, 8]):
    _plt.subplots(1, 2, figsize=figsize)
    _plt.subplot(121)
    time = crisp_data.df.Time
    colname = crisp_data.df.columns[1+train]
    _plt.plot(time, crisp_data.df[colname]*1e-3, label=crisp_data.df[colname].name)
    _plt.ylabel('Current [kA]')
    _plt.xlabel('$t$ [fs]')
    _plt.legend()
    _plt.xlim(xlim)
    _plt.subplot(122)
    length = crisp_data.calcLengthOneTrainCUMSUM(train, percentile=percentile, plotCumsum=True)
    _plt.xlim(xlim)


def plotLengthPerTrain(crisp_data, percentile=0.95, minCut=None, maxCut=None, figsize=[14, 4]):
    _plt.figure(figsize=figsize)
    trainIDs, Lengths = crisp_data.calcLengthAllTrains(percentile=percentile)
    trainIDs_cut = trainIDs[minCut: maxCut]
    Lengths_cut = Lengths[minCut: maxCut]
    _plt.plot(trainIDs, Lengths)
    _plt.plot(trainIDs_cut, Lengths_cut, label=r"$\overline{{\sigma_t}}$ = {:.2f} fs".format(_np.mean(Lengths_cut)))
    _plt.ylabel(r"$\sigma_t$ [fs]")
    _plt.xlabel('Train ID')
    _plt.legend()


def plotLenghtHist(crisp_data, percentile=0.95, bins=50, minCut=None, maxCut=None, figsize=[9, 7]):
    _plt.figure(figsize=figsize)
    trainIDs, Lengths = crisp_data.calcLengthAllTrains(percentile=percentile)
    Lengths_cut = Lengths[minCut: maxCut]
    _plt.hist(Lengths_cut, bins=bins, label=r"$\delta\sigma_t$ = {:.2f} fs".format(_np.std(Lengths_cut)))
    _plt.xlabel(r"$\sigma_t$ [fs]")
    _plt.legend()


def plotLengthAndHistPer(self, percentile=0.95, bins=50, minCut=None, maxCut=None, figsize=[12, 5]):
    _plt.subplots(1, 2, figsize=figsize)
    trainIDs, Lengths = self.calcLengthAllTrains(percentile=percentile)
    trainIDs_cut = trainIDs[minCut: maxCut]
    Lengths_cut = Lengths[minCut: maxCut]

    _plt.subplot(121)
    _plt.plot(trainIDs, Lengths, color='gray')
    _plt.plot(trainIDs_cut, Lengths_cut, color='C0', label=r"$\overline{{\sigma_t}}$ = {:.2f} fs".format(_np.mean(Lengths_cut)))
    # _plt.ticklabel_format(axis="x", style="sci", scilimits=(3, 3))
    _plt.ylabel(r"$\sigma_t$ [fs]")
    _plt.xlabel('Train ID')
    _plt.legend()

    _plt.subplot(122)
    _plt.hist(Lengths_cut, bins=bins, histtype='step', color='C0', label=r"$\delta\sigma_t$ = {:.2f} fs".format(_np.std(Lengths_cut)))
    # _plt.ticklabel_format(axis="y", style="sci", scilimits=(3, 3))
    _plt.ylabel('counts')
    _plt.xlabel(r"$\sigma_t$ [fs]")
    _plt.legend()

    _plt.tight_layout()

