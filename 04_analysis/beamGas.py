import pybdsim as _bd
import ROOT as _rt
import numpy as _np
import matplotlib.pyplot as _plt
from scipy.optimize import curve_fit


def linear(x, a, b):
    return a * x + b


def exponential(x, a, b):
    return a * _np.exp(-b * x)


def analysis(inputfilename):
    tag =inputfilename.split('/')[-1].split('.')[0]

    root_data = _bd.Data.Load(inputfilename)
    e = root_data.GetEvent()
    t = root_data.GetEventTree()

    print(t.GetEntries())

    h_PrimaryFirstHit_S         = _rt.TH1D("h_primaryFirstHit_S",           "{} Primary First Hit S".format(tag),           100, 0, 300)
    h_PrimaryFirstHit_eBrem_S   = _rt.TH1D("h_primaryFirstHit_eBrem_S",     "{} Primary First Hit eBrem S".format(tag),     100, 0, 300)
    h_PrimaryFirstHit_Coulomb_S = _rt.TH1D("h_primaryFirstHit_Coulomb_S",   "{} Primary First Hit Coulomb S".format(tag),   100, 0, 300)
    h_PrimaryFirstHit_elecNuc_S = _rt.TH1D("h_primaryFirstHit_elecNuc_S",   "{} Primary First Hit elecNuc S".format(tag),   100, 0, 300)

    for i, evt in enumerate(t):
        h_PrimaryFirstHit_S.Fill(evt.PrimaryFirstHit.S[0], evt.PrimaryFirstHit.weight[0])

        if evt.PrimaryFirstHit.postStepProcessSubType[0] == 1:
            h_PrimaryFirstHit_Coulomb_S.Fill(evt.PrimaryFirstHit.S[0], evt.PrimaryFirstHit.weight[0])
        if evt.PrimaryFirstHit.postStepProcessSubType[0] == 3:
            h_PrimaryFirstHit_eBrem_S.Fill(evt.PrimaryFirstHit.S[0], evt.PrimaryFirstHit.weight[0])
        if evt.PrimaryFirstHit.postStepProcessSubType[0] == 121:
            h_PrimaryFirstHit_elecNuc_S.Fill(evt.PrimaryFirstHit.S[0], evt.PrimaryFirstHit.weight[0])

    f = _rt.TFile("{}_hist.root".format(tag),"recreate")
    h_PrimaryFirstHit_S.Write()
    h_PrimaryFirstHit_eBrem_S.Write()
    h_PrimaryFirstHit_Coulomb_S.Write()
    h_PrimaryFirstHit_elecNuc_S.Write()
    f.Close()


def plot_all(inputfilename):
    f = _rt.TFile(inputfilename)

    _plt.figure()
    plot(f.Get("h_primaryFirstHit_S"), linearFit=True, exponentialFit=False, fitRange=[15, -10])
    _plt.title("Primaries first hits along the machine for all processes")

    #_plt.figure()
    #plot(f.Get("h_primaryFirstHit_eBrem_S"), linearFit=True, fitRange=[0, -10], color='blue')
    #plot(f.Get("h_primaryFirstHit_Coulomb_S"), linearFit=True, fitRange=[0, -10], color='orange')
    #plot(f.Get("h_primaryFirstHit_elecNuc_S"), linearFit=True, fitRange=[0, -10], color='green')
    #_plt.title("Primaries first hits along the machine for each processes")


def plot(root_hist, linearFit=False, exponentialFit=False, fitRange=None, color=None):
    python_hist = _bd.Data.TH1(root_hist)

    name = python_hist.hist.GetName()
    title = python_hist.hist.GetTitle()
    centres = python_hist.xcentres
    contents = python_hist.contents
    errors = python_hist.errors
    widths = python_hist.xwidths

    _plt.errorbar(centres, contents, yerr=errors, xerr=widths * 0.5, ls='', marker='+', color=color, label=title)

    if fitRange:
        centres_fit = centres[fitRange[0]:fitRange[1]]
        contents_fit = contents[fitRange[0]:fitRange[1]]
        errors_fit = errors[fitRange[0]:fitRange[1]]
    else:
        centres_fit = centres
        contents_fit = contents
        errors_fit = errors

    if linearFit:
        popt, pcov = curve_fit(linear, centres_fit, contents_fit)#, sigma=errors_fit, absolute_sigma=True)
        _plt.plot(centres_fit, linear(centres_fit, *popt), ls='-', color=color, label='linear fit: a=%1.3e, b=%1.3e' % tuple(popt))
    if exponentialFit:
        popt, pcov = curve_fit(exponential, centres_fit, contents_fit, p0=[300, 0.01], sigma=errors_fit, absolute_sigma=True)
        _plt.plot(centres_fit, exponential(centres_fit, *popt), ls='--', color=color, label='exponential fit: a=%1.3e, b=%1.3e' % tuple(popt))

    _plt.yscale("log")
    _plt.legend()




if __name__ == "__main__":
    #analysis("../03_bdsimModel/T20_bias_output.root")
    plot_all("T20_bias_output_hist.root")

    #analysis("../03_bdsimModel/T20_bias_2_output.root")
    plot_all("T20_bias_2_output_hist.root")

    #analysis("../03_bdsimModel/T20_bias_4_output.root")
    plot_all("T20_bias_4_output_hist.root")

    #analysis("../03_bdsimModel/T20_bias_8_output.root")
    plot_all("T20_bias_8_output_hist.root")

    #analysis("../03_bdsimModel/T20_bias_x2_output.root")
    plot_all("T20_bias_x2_output_hist.root")

    #analysis("../03_bdsimModel/T20_bias_x4_output.root")
    plot_all("T20_bias_x4_output_hist.root")

    _plt.show()