import pybdsim as _bd
import ROOT as _rt
import numpy as _np
import matplotlib.pyplot as _plt
import glob as _gl
import pickle as _pk
from scipy.optimize import curve_fit
from collections import defaultdict

ELECTRONS_PER_BUNCH = 2e9


def linear(x, a, b):
    return a * x + b


def exponential(x, a, b):
    return a * _np.exp(-b * x)


def poly2(x, a, b, c):
    X = _np.log(x)
    return a * X * X + b * X + c


def GetNbParticles(inputfilename):
    root_data = _bd.Data.Load(inputfilename)
    return root_data.header.nOriginalEvents


def analysisFilelist(tagfilelist):
    taglist = open(tagfilelist)
    for tag in taglist:
        analysis(_gl.glob('../04_dataLocal/*'+tag.replace('\n', '')+'*.root')[0])
        farmfilelist = _gl.glob('../05_dataFarm/*'+tag.replace('\n', '')+'*.root')
        for file in farmfilelist:
            analysis(file)
    taglist.close()


def analysisCombine(tagfilelist):
    taglist = open(tagfilelist)
    for tag in taglist:
        nb_part = 0
        tagstring = '*_part_'+tag.replace('\n', '')+'_hist.root'
        print(tagstring)
        filelist = _gl.glob(tagstring)
        print(filelist)
        for file in filelist:
            nb_part += GetNbParticles(file)
        _bd.Run.RebdsimHistoMerge(str(nb_part)+'_part_'+tag+'_merged_hist.root', tagstring, rebdsimHistoExecutable='rebdsimCombine')
    taglist.close()


def makeFileLists(tagfilelist):
    taglist = open(tagfilelist)
    localhistlist = open("localhistlist", "w")
    farmhistlist = open("farmhistlist", "w")
    for tag in taglist:
        for file in _gl.glob('10000*'+tag.replace('\n', '')+'*_hist.root'):
            localhistlist.write(file+'\n')
        for file in _gl.glob('*'+tag.replace('\n', '')+'*_merged_hist.root'):
            farmhistlist.write(file+'\n')
    taglist.close()
    localhistlist.close()
    farmhistlist.close()


def analysis(inputfilename, nbins=50):
    if type(inputfilename) == list:
        for file in inputfilename:
            analysis(file)
        return 0

    tag = inputfilename.split('/')[-1].split('.root')[0].split('part_')[-1]

    root_data = _bd.Data.Load(inputfilename)
    # e = root_data.GetEvent()
    t = root_data.GetEventTree()

    print("File :", inputfilename, " / Nb of entries = ", t.GetEntries())

    HIST_DICT = {}

    HIST_DICT['PFH_S_unweight']          = _rt.TH1D("PFH_S_unweight",          "{} PFH wrt S all processes (unweighted)".format(tag), nbins, 0, 300)
    HIST_DICT['PFH_S']                   = _rt.TH1D("PFH_S",                   "{} PFH wrt S all processes".format(tag),              nbins, 0, 300)
    HIST_DICT['PFH_S_eBrem_unweight']    = _rt.TH1D("PFH_S_eBrem_unweight",    "{} PFH wrt S eBrem".format(tag),                      nbins, 0, 300)
    HIST_DICT['PFH_S_eBrem']             = _rt.TH1D("PFH_S_eBrem",             "{} PFH wrt S eBrem".format(tag),                      nbins, 0, 300)
    HIST_DICT['PFH_S_Coulomb_unweight']  = _rt.TH1D("PFH_S_Coulomb_unweight",  "{} PFH wrt S Coulomb".format(tag),                    nbins, 0, 300)
    HIST_DICT['PFH_S_Coulomb']           = _rt.TH1D("PFH_S_Coulomb",           "{} PFH wrt S Coulomb".format(tag),                    nbins, 0, 300)
    HIST_DICT['PFH_S_elecNuc_unweight']  = _rt.TH1D("PFH_S_elecNuc_unweight",  "{} PFH wrt S elecNuc".format(tag),                    nbins, 0, 300)
    HIST_DICT['PFH_S_elecNuc']           = _rt.TH1D("PFH_S_elecNuc",           "{} PFH wrt S elecNuc".format(tag),                    nbins, 0, 300)

    HIST_DICT['PFH_x']                   = _rt.TH1D("PFH_x",                   "{} PFH wrt x all processes".format(tag),              nbins, -2e-4, 2e-4)
    HIST_DICT['PFH_y']                   = _rt.TH1D("PFH_y",                   "{} PFH wrt y all processes".format(tag),              nbins, -2e-4, 2e-4)
    HIST_DICT['PFH_z']                   = _rt.TH1D("PFH_z",                   "{} PFH wrt z all processes".format(tag),              nbins, -10,   10)
    HIST_DICT['PFH_E']                   = _rt.TH1D("PFH_E",                   "{} PFH wrt energy all processes".format(tag),         nbins,  0,    2e-4)

    HIST_DICT['StartSampler_x']          = _rt.TH1D("StartSampler_x",          "{} Beam profile in x at start sampler".format(tag),   nbins, -3,   3)
    HIST_DICT['StartSampler_xp']         = _rt.TH1D("StartSampler_xp",         "{} Beam profile in xp at start sampler".format(tag),  nbins, -1.5, 1.5)
    HIST_DICT['StartSampler_y']          = _rt.TH1D("StartSampler_y",          "{} Beam profile in y at start sampler".format(tag),   nbins, -3,   3)
    HIST_DICT['StartSampler_yp']         = _rt.TH1D("StartSampler_yp",         "{} Beam profile in yp at start sampler".format(tag),  nbins, -1.5, 1.5)
    HIST_DICT['StartSampler_E']          = _rt.TH1D("StartSampler_E",          "{} Beam energy profile at start sampler".format(tag), nbins,  0,   14)

    HIST_DICT['MidSampler_x']            = _rt.TH1D("MidSampler_x",            "{} Beam profile in x at mid sampler".format(tag),     nbins, -3,   3)
    HIST_DICT['MidSampler_xp']           = _rt.TH1D("MidSampler_xp",           "{} Beam profile in xp at mid sampler".format(tag),    nbins, -1.5, 1.5)
    HIST_DICT['MidSampler_y']            = _rt.TH1D("MidSampler_y",            "{} Beam profile in y at mid sampler".format(tag),     nbins, -3,   3)
    HIST_DICT['MidSampler_yp']           = _rt.TH1D("MidSampler_yp",           "{} Beam profile in yp at mid sampler".format(tag),    nbins, -1.5, 1.5)
    HIST_DICT['MidSampler_E']            = _rt.TH1D("MidSampler_E",            "{} Beam energy profile at mid sampler".format(tag),   nbins,  0,   14)

    HIST_DICT['EndSampler_x']            = _rt.TH1D("EndSampler_x",            "{} Beam profile in x at end sampler".format(tag),     nbins, -3,   3)
    HIST_DICT['EndSampler_xp']           = _rt.TH1D("EndSampler_xp",           "{} Beam profile in xp at end sampler".format(tag),    nbins, -1.5, 1.5)
    HIST_DICT['EndSampler_y']            = _rt.TH1D("EndSampler_y",            "{} Beam profile in y at end sampler".format(tag),     nbins, -3,   3)
    HIST_DICT['EndSampler_yp']           = _rt.TH1D("EndSampler_yp",           "{} Beam profile in yp at end sampler".format(tag),    nbins, -1.5, 1.5)
    HIST_DICT['EndSampler_E']            = _rt.TH1D("EndSampler_E",            "{} Beam energy profile at end sampler".format(tag),   nbins,  0,   14)

    HIST_DICT['EndSampler_x_electrons']  = _rt.TH1D("EndSampler_x_electrons",  "{} Beam profile in x at end sampler for electrons".format(tag),      nbins, -3,   3)
    HIST_DICT['EndSampler_xp_electrons'] = _rt.TH1D("EndSampler_xp_electrons", "{} Beam profile in xp at end sampler for electrons".format(tag),     nbins, -1.5, 1.5)
    HIST_DICT['EndSampler_y_electrons']  = _rt.TH1D("EndSampler_y_electrons",  "{} Beam profile in y at end sampler for electrons".format(tag),      nbins, -3,   3)
    HIST_DICT['EndSampler_yp_electrons'] = _rt.TH1D("EndSampler_yp_electrons", "{} Beam profile in yp at end sampler for electrons".format(tag),     nbins, -1.5, 1.5)
    HIST_DICT['EndSampler_E_electrons']  = _rt.TH1D("EndSampler_E_electrons",  "{} Beam energy profile at end sampler for electrons".format(tag),    nbins,  0,   14)

    HIST_DICT['EndSampler_x_positrons']  = _rt.TH1D("EndSampler_x_positrons",  "{} Beam profile in x at end sampler for positrons".format(tag),      nbins, -3,   3)
    HIST_DICT['EndSampler_xp_positrons'] = _rt.TH1D("EndSampler_xp_positrons", "{} Beam profile in xp at end sampler for positrons".format(tag),     nbins, -1.5, 1.5)
    HIST_DICT['EndSampler_y_positrons']  = _rt.TH1D("EndSampler_y_positrons",  "{} Beam profile in y at end sampler for positrons".format(tag),      nbins, -3,   3)
    HIST_DICT['EndSampler_yp_positrons'] = _rt.TH1D("EndSampler_yp_positrons", "{} Beam profile in yp at end sampler for positrons".format(tag),     nbins, -1.5, 1.5)
    HIST_DICT['EndSampler_E_positrons']  = _rt.TH1D("EndSampler_E_positrons",  "{} Beam energy profile at end sampler for positrons".format(tag),    nbins,  0,   14)

    HIST_DICT['EndSampler_x_photons']    = _rt.TH1D("EndSampler_x_photons",    "{} Beam profile in x at end sampler for photons".format(tag),        nbins, -3,   3)
    HIST_DICT['EndSampler_xp_photons']   = _rt.TH1D("EndSampler_xp_photons",   "{} Beam profile in xp at end sampler for photons".format(tag),       nbins, -1.5, 1.5)
    HIST_DICT['EndSampler_y_photons']    = _rt.TH1D("EndSampler_y_photons",    "{} Beam profile in y at end sampler for photons".format(tag),        nbins, -3,   3)
    HIST_DICT['EndSampler_yp_photons']   = _rt.TH1D("EndSampler_yp_photons",   "{} Beam profile in yp at end sampler for photons".format(tag),       nbins, -1.5, 1.5)
    HIST_DICT['EndSampler_E_photons']    = _rt.TH1D("EndSampler_E_photons",    "{} Beam energy profile at end sampler for photons".format(tag),      nbins,  0,   14)

    PART_DICT = defaultdict(float)

    for i, evt in enumerate(t):

        if len(evt.PrimaryFirstHit.weight) != 0:
            HIST_DICT['PFH_S_unweight'].Fill(evt.PrimaryFirstHit.S[0])
            HIST_DICT['PFH_S'].Fill(evt.PrimaryFirstHit.S[0], evt.PrimaryFirstHit.weight[0])
            if evt.PrimaryFirstHit.postStepProcessSubType[0] == 1:
                HIST_DICT['PFH_S_Coulomb_unweight'].Fill(evt.PrimaryFirstHit.S[0])
                HIST_DICT['PFH_S_Coulomb'].Fill(evt.PrimaryFirstHit.S[0], evt.PrimaryFirstHit.weight[0])
            if evt.PrimaryFirstHit.postStepProcessSubType[0] == 3:
                HIST_DICT['PFH_S_eBrem_unweight'].Fill(evt.PrimaryFirstHit.S[0])
                HIST_DICT['PFH_S_eBrem'].Fill(evt.PrimaryFirstHit.S[0], evt.PrimaryFirstHit.weight[0])
            if evt.PrimaryFirstHit.postStepProcessSubType[0] == 121:
                HIST_DICT['PFH_S_elecNuc_unweight'].Fill(evt.PrimaryFirstHit.S[0])
                HIST_DICT['PFH_S_elecNuc'].Fill(evt.PrimaryFirstHit.S[0], evt.PrimaryFirstHit.weight[0])

            HIST_DICT['PFH_x'].Fill(evt.PrimaryFirstHit.x[0], evt.PrimaryFirstHit.weight[0])
            HIST_DICT['PFH_y'].Fill(evt.PrimaryFirstHit.y[0], evt.PrimaryFirstHit.weight[0])
            HIST_DICT['PFH_z'].Fill(evt.PrimaryFirstHit.z[0], evt.PrimaryFirstHit.weight[0])
            HIST_DICT['PFH_E'].Fill(evt.PrimaryFirstHit.energy[0], evt.PrimaryFirstHit.weight[0])

        if len(evt.QFH41CL.weight) != 0:
            HIST_DICT['StartSampler_x'].Fill(evt.QFH41CL.x[0], evt.QFH41CL.weight[0])
            HIST_DICT['StartSampler_xp'].Fill(evt.QFH41CL.xp[0], evt.QFH41CL.weight[0])
            HIST_DICT['StartSampler_y'].Fill(evt.QFH41CL.y[0], evt.QFH41CL.weight[0])
            HIST_DICT['StartSampler_yp'].Fill(evt.QFH41CL.yp[0], evt.QFH41CL.weight[0])
            HIST_DICT['StartSampler_E'].Fill(evt.QFH41CL.energy[0], evt.QFH41CL.weight[0])

        if len(evt.KL2TL.weight) != 0:
            HIST_DICT['MidSampler_x'].Fill(evt.KL2TL.x[0], evt.KL2TL.weight[0])
            HIST_DICT['MidSampler_xp'].Fill(evt.KL2TL.xp[0], evt.KL2TL.weight[0])
            HIST_DICT['MidSampler_y'].Fill(evt.KL2TL.y[0], evt.KL2TL.weight[0])
            HIST_DICT['MidSampler_yp'].Fill(evt.KL2TL.yp[0], evt.KL2TL.weight[0])
            HIST_DICT['MidSampler_E'].Fill(evt.KL2TL.energy[0], evt.KL2TL.weight[0])

        if len(evt.D70899L.weight) != 0:
            HIST_DICT['EndSampler_x'].Fill(evt.D70899L.x[0], evt.D70899L.weight[0])
            HIST_DICT['EndSampler_xp'].Fill(evt.D70899L.xp[0], evt.D70899L.weight[0])
            HIST_DICT['EndSampler_y'].Fill(evt.D70899L.y[0], evt.D70899L.weight[0])
            HIST_DICT['EndSampler_yp'].Fill(evt.D70899L.yp[0], evt.D70899L.weight[0])
            HIST_DICT['EndSampler_E'].Fill(evt.D70899L.energy[0], evt.D70899L.weight[0])

            partID = evt.D70899L.partID[0]
            PART_DICT[0] += evt.D70899L.weight[0]
            PART_DICT[partID] += evt.D70899L.weight[0]

            if partID == 11:
                HIST_DICT['EndSampler_x_electrons'].Fill(evt.D70899L.x[0], evt.D70899L.weight[0])
                HIST_DICT['EndSampler_xp_electrons'].Fill(evt.D70899L.xp[0], evt.D70899L.weight[0])
                HIST_DICT['EndSampler_y_electrons'].Fill(evt.D70899L.y[0], evt.D70899L.weight[0])
                HIST_DICT['EndSampler_yp_electrons'].Fill(evt.D70899L.yp[0], evt.D70899L.weight[0])
                HIST_DICT['EndSampler_E_electrons'].Fill(evt.D70899L.energy[0], evt.D70899L.weight[0])
            elif partID == -11:
                HIST_DICT['EndSampler_x_positrons'].Fill(evt.D70899L.x[0], evt.D70899L.weight[0])
                HIST_DICT['EndSampler_xp_positrons'].Fill(evt.D70899L.xp[0], evt.D70899L.weight[0])
                HIST_DICT['EndSampler_y_positrons'].Fill(evt.D70899L.y[0], evt.D70899L.weight[0])
                HIST_DICT['EndSampler_yp_positrons'].Fill(evt.D70899L.yp[0], evt.D70899L.weight[0])
                HIST_DICT['EndSampler_E_positrons'].Fill(evt.D70899L.energy[0], evt.D70899L.weight[0])
            elif partID == 22:
                HIST_DICT['EndSampler_x_photons'].Fill(evt.D70899L.x[0], evt.D70899L.weight[0])
                HIST_DICT['EndSampler_xp_photons'].Fill(evt.D70899L.xp[0], evt.D70899L.weight[0])
                HIST_DICT['EndSampler_y_photons'].Fill(evt.D70899L.y[0], evt.D70899L.weight[0])
                HIST_DICT['EndSampler_yp_photons'].Fill(evt.D70899L.yp[0], evt.D70899L.weight[0])
                HIST_DICT['EndSampler_E_photons'].Fill(evt.D70899L.energy[0], evt.D70899L.weight[0])

    for hist in HIST_DICT:
        HIST_DICT[hist].Scale(ELECTRONS_PER_BUNCH/t.GetEntries())

    for key in PART_DICT:
        PART_DICT[key] = PART_DICT[key]*ELECTRONS_PER_BUNCH/t.GetEntries()

    outputfilename = inputfilename.replace('04_dataLocal', '06_analysis').replace('05_dataFarm', '06_analysis').replace('.root', '')

    partfile = open('{}_partfile.pk'.format(outputfilename), 'wb')
    _pk.dump(PART_DICT, partfile)
    partfile.close()

    outfile = _bd.Data.CreateEmptyRebdsimFile('{}_hist.root'.format(outputfilename), root_data.header.nOriginalEvents)
    _bd.Data.WriteROOTHistogramsToDirectory(outfile, "Event/MergedHistograms", list(HIST_DICT.values()))
    outfile.Close()


def plot_var(rootlistfile, histname, errorBars=False, fit=False, xLogScale=False, color=None, printLegend=True):
    entries = 0
    X_fit = _np.linspace(0.025, 4, 200)
    X = _np.array([])
    Y = _np.array([])
    ERR = _np.array([])
    filelist = open(rootlistfile)
    for file in filelist:
        f = _rt.TFile(file.replace('\n', ''))
        root_hist = f.Get("Event/MergedHistograms/"+histname)
        python_hist = _bd.Data.TH1(root_hist)
        X = _np.append(X, float(file.replace('\n', '').split('/')[-1].split('_hist.root')[0].split('_merged')[0].split('_')[-1]))

        contents = python_hist.contents[0:-1]
        VALUE = contents.std()/contents.mean()*100

        errors = python_hist.errors[0:-1]
        error_mean = 1/python_hist.entries * _np.sqrt(sum(errors**2))
        error_std = contents.std() / _np.sqrt(python_hist.entries)
        ERR = _np.append(ERR, VALUE*_np.sqrt((error_std/contents.std())**2 + (error_mean/contents.mean())**2))

        entries = max(entries, python_hist.entries)
        Y = _np.append(Y, VALUE)
        if X[-1] == 0.5:
            print("Value for {} particles and {} biasing factor : {}%".format(entries, X[-1], Y[-1]))
    filelist.close()

    if errorBars:
        _plt.errorbar(X, Y, yerr=ERR, fmt='.{}'.format(color), capsize=3, label='$V_{\\rm PFH}$ for %1.1e particles' % entries)
    else:
        _plt.plot(X, Y, ls='', marker='+', markersize=13, markeredgewidth=2, color=color, label='$V_{\\rm PFH}$ for %1.1e particles' % entries)

    if fit:
        popt, pcov = curve_fit(poly2, X, Y)
        _plt.plot(X_fit, poly2(X_fit, *popt), ls='-', color=color, label='Polynomial fit : min for factor = %5.3f' % _np.exp(-popt[1]/(2*popt[0])))

    if xLogScale:
        _plt.xscale("log")
    if printLegend:
        _plt.legend()


def plot_hist(inputfilename, histname, particlenames=False, errorbars=False, steps=True,
              linFit=False, expFit=False, fitRange=None, yLogScale=False, color=None, printLegend=True):
    f = _rt.TFile(inputfilename)
    test_bd_load = _bd.Data.Load(inputfilename)
    npart = test_bd_load.header.nOriginalEvents
    root_hist = f.Get("Event/MergedHistograms/"+histname)
    python_hist = _bd.Data.TH1(root_hist)

    title = python_hist.hist.GetTitle()
    centres = python_hist.xcentres[:-2]
    contents = python_hist.contents[:-2]
    errors = python_hist.errors[:-2]
    widths = python_hist.xwidths[:-2]

    if particlenames:
        _plt.plot([root_hist.GetXaxis().GetBinLabel(i+1) for i in range(len(centres))], contents,
                  ls='', marker='o', markersize=12, label=' %2.3e initial particles' % npart)
    if errorbars:
        _plt.errorbar(centres, contents, yerr=errors, xerr=widths * 0.5, ls='', marker='+', color=color)  # , label=title)
    if steps:
        _plt.step(centres, contents, where='mid', color=color, label=title)

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

    if linFit:
        popt, pcov = curve_fit(linear, centres, contents, sigma=errors, absolute_sigma=True)
        _plt.plot(centres, linear(centres, *popt), ls='--', color=color, label='linear fit: slope = %1.3e, b = %1.3e' % tuple(popt))
    if expFit:
        popt, pcov = curve_fit(exponential, centres, contents, p0=[300, 0.01], sigma=errors, absolute_sigma=True)
        _plt.plot(centres, exponential(centres, *popt), ls='--', color=color, label='exponential fit: slope=-%1.3e' % popt[1])

    if yLogScale:
        _plt.yscale("log")
    if printLegend:
        _plt.legend()


def plot_multiple_hist(filelist, histlist, **args):
    color = 0
    for file in filelist:
        for hist in histlist:
            plot_hist(file, hist, color='C{}'.format(color), **args)
            color += 1


if __name__ == "__main__":

    makeFileLists("tagfilelist")
    print("File Lists Created")
    analysisFilelist("tagfilelist")
    print("Analysis Completed")
    # analysisCombine("tagfilelist")
    # print("Histograms Combined")
