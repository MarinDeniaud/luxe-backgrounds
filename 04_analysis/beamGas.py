import pybdsim as _bd
import ROOT as _rt
import numpy as _np
import matplotlib.pyplot as _plt
from scipy.optimize import curve_fit


def linear(x, a, b):
    return a * x + b


def exponential(x, a, b):
    return a * _np.exp(-b * x)


def analysis(inputfilename, nbins=100):
    tag = inputfilename.split('/')[-1].split('.')[0]

    root_data = _bd.Data.Load(inputfilename)
    e = root_data.GetEvent()
    t = root_data.GetEventTree()

    print(t.GetEntries())

    h_PrimaryFirstHit_S                = _rt.TH1D("h_PFH_S",                "{} PFH wrt S all processes".format(tag),            nbins, 0, 300)
    h_PrimaryFirstHit_S_weight         = _rt.TH1D("h_PFH_S_weight",         "{} PFH wrt S all processes (weighted)".format(tag), nbins, 0, 300)
    h_PrimaryFirstHit_S_eBrem          = _rt.TH1D("h_PFH_S_eBrem",          "{} PFH wrt S eBrem".format(tag),                    nbins, 0, 300)
    h_PrimaryFirstHit_S_eBrem_weight   = _rt.TH1D("h_PFH_S_eBrem_weight",   "{} PFH wrt S eBrem (weighted)".format(tag),         nbins, 0, 300)
    h_PrimaryFirstHit_S_Coulomb        = _rt.TH1D("h_PFH_S_Coulomb",        "{} PFH wrt S Coulomb".format(tag),                  nbins, 0, 300)
    h_PrimaryFirstHit_S_Coulomb_weight = _rt.TH1D("h_PFH_S_Coulomb_weight", "{} PFH wrt S Coulomb (weighted)".format(tag),       nbins, 0, 300)
    h_PrimaryFirstHit_S_elecNuc        = _rt.TH1D("h_PFH_S_elecNuc",        "{} PFH wrt S elecNuc".format(tag),                  nbins, 0, 300)
    h_PrimaryFirstHit_S_elecNuc_weight = _rt.TH1D("h_PFH_S_elecNuc_weight", "{} PFH wrt S elecNuc (weighted)".format(tag),       nbins, 0, 300)

    for i, evt in enumerate(t):
        h_PrimaryFirstHit_S.Fill(evt.PrimaryFirstHit.S[0])
        h_PrimaryFirstHit_S_weight.Fill(evt.PrimaryFirstHit.S[0], evt.PrimaryFirstHit.weight[0])

        if evt.PrimaryFirstHit.postStepProcessSubType[0] == 1:
            h_PrimaryFirstHit_S_Coulomb.Fill(evt.PrimaryFirstHit.S[0])
            h_PrimaryFirstHit_S_Coulomb_weight.Fill(evt.PrimaryFirstHit.S[0], evt.PrimaryFirstHit.weight[0])
        if evt.PrimaryFirstHit.postStepProcessSubType[0] == 3:
            h_PrimaryFirstHit_S_eBrem.Fill(evt.PrimaryFirstHit.S[0])
            h_PrimaryFirstHit_S_eBrem_weight.Fill(evt.PrimaryFirstHit.S[0], evt.PrimaryFirstHit.weight[0])
        if evt.PrimaryFirstHit.postStepProcessSubType[0] == 121:
            h_PrimaryFirstHit_S_elecNuc.Fill(evt.PrimaryFirstHit.S[0])
            h_PrimaryFirstHit_S_elecNuc_weight.Fill(evt.PrimaryFirstHit.S[0], evt.PrimaryFirstHit.weight[0])

    f = _rt.TFile("{}_hist.root".format(tag), "recreate")
    h_PrimaryFirstHit_S.Write()
    h_PrimaryFirstHit_S_weight.Write()
    h_PrimaryFirstHit_S_eBrem.Write()
    h_PrimaryFirstHit_S_eBrem_weight.Write()
    h_PrimaryFirstHit_S_Coulomb.Write()
    h_PrimaryFirstHit_S_Coulomb_weight.Write()
    h_PrimaryFirstHit_S_elecNuc.Write()
    h_PrimaryFirstHit_S_elecNuc_weight.Write()
    f.Close()


def plot_all(inputfilename):
    f = _rt.TFile(inputfilename)

    plot(f.Get("h_PFH_S"), linearFit=False, exponentialFit=True, fitRange=[0, -5], fitOnly=True, logScale=True)
    _plt.title("Primaries first hits along the machine for all processes")

    #_plt.figure()
    #plot(f.Get("h_primaryFirstHit_eBrem_S"), linearFit=True, fitRange=[0, -10], color='blue')
    #plot(f.Get("h_primaryFirstHit_Coulomb_S"), linearFit=True, fitRange=[0, -10], color='orange')
    #plot(f.Get("h_primaryFirstHit_elecNuc_S"), linearFit=True, fitRange=[0, -10], color='green')
    #_plt.title("Primaries first hits along the machine for each processes")


def plot(inputfilename, histname, linearFit=False, exponentialFit=False, fitRange=None, fitOnly=False, color=None, logScale=False):
    f = _rt.TFile(inputfilename)
    root_hist = f.Get(histname)
    python_hist = _bd.Data.TH1(root_hist)

    name = python_hist.hist.GetName()
    title = python_hist.hist.GetTitle()
    centres = python_hist.xcentres
    contents = python_hist.contents
    errors = python_hist.errors
    widths = python_hist.xwidths

    if not fitOnly:
        _plt.errorbar(centres, contents, yerr=errors, xerr=widths * 0.5, ls='', marker='+', color=color, label=title)

    empty_bins = []
    for i, val in enumerate(contents):
        if val == 0:
            empty_bins.append(i)
    centres = _np.delete(centres, empty_bins)
    contents = _np.delete(contents, empty_bins)
    errors = _np.delete(errors, empty_bins)

    if fitRange:
        centres = centres[fitRange[0]:fitRange[1]]
        contents = contents[fitRange[0]:fitRange[1]]
        errors = errors[fitRange[0]:fitRange[1]]

    if linearFit:
        popt, pcov = curve_fit(linear, centres, contents, sigma=errors, absolute_sigma=True)
        _plt.plot(centres, linear(centres, *popt), ls='-', color=color, label='{} linear fit: slope = %1.3e'.format(title) % popt[0])
    if exponentialFit:
        popt, pcov = curve_fit(exponential, centres, contents, p0=[300, 0.01], sigma=errors, absolute_sigma=True)
        _plt.plot(centres, exponential(centres, *popt), ls='--', color=color, label='{} exponential fit: slope=-%1.3e'.format(title) % popt[1])

    if logScale:
        _plt.yscale("log")
    _plt.legend()


if __name__ == "__main__":

    if False:
        analysis("../03_bdsimModel/T20_bias_output.root")
        analysis("../03_bdsimModel/T20_bias_2_output.root")
        analysis("../03_bdsimModel/T20_bias_4_output.root")
        analysis("../03_bdsimModel/T20_bias_8_output.root")
        analysis("../03_bdsimModel/T20_bias_x2_output.root")
        analysis("../03_bdsimModel/T20_bias_x4_output.root")

    _plt.figure()
    plot("T20_bias_output_hist.root", "h_PFH_S", exponentialFit=True, color='green', logScale=True)
    plot("T20_bias_2_output_hist.root", "h_PFH_S", exponentialFit=True, color='blue', logScale=True)
    plot("T20_bias_4_output_hist.root", "h_PFH_S", exponentialFit=True, color='purple', logScale=True)
    # plot("T20_bias_8_output_hist.root", "h_PFH_S", exponentialFit=True, color='black', logScale=True)
    plot("T20_bias_x2_output_hist.root", "h_PFH_S", exponentialFit=True, color='orange', logScale=True)
    # plot("T20_bias_x4_output_hist.root", "h_PFH_S", exponentialFit=True, color='red', logScale=True)

    _plt.title("Primaries first hits along the machine for all processes")

    _plt.figure()
    plot("T20_bias_output_hist.root", "h_PFH_S_weight", linearFit=True, fitOnly=False, color='green', logScale=True)
    # plot("T20_bias_2_output_hist.root", "h_PFH_S_weight", linearFit=True, fitOnly=True, color='blue', logScale=True)
    plot("T20_bias_4_output_hist.root", "h_PFH_S_weight", linearFit=True, fitOnly=False, color='purple', logScale=True)
    # plot("T20_bias_8_output_hist.root", "h_PFH_S_weight", linearFit=True, fitOnly=True, color='black', logScale=True)
    # plot("T20_bias_x2_output_hist.root", "h_PFH_S_weight", linearFit=True, fitOnly=True, color='orange', logScale=True)
    plot("T20_bias_x4_output_hist.root", "h_PFH_S_weight", linearFit=True, fitOnly=False, color='red', logScale=True)

    _plt.title("Primaries first hits along the machine for all processes (weighted)")

    _plt.show()