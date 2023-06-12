import pybdsim as _bd
import pymad8 as _m8
import ROOT as _rt
import numpy as _np
import matplotlib.pyplot as _plt
import glob as _gl
import pickle as _pk
import sympy as _sp
import jinja2 as _jj
import subprocess as _sub
from scipy.optimize import curve_fit
from collections import defaultdict
from scipy.integrate import simps
from scipy.interpolate import interp1d


def _printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


ELECTRONS_PER_BUNCH = 2e9


def linear(x, a, b):
    return a * x + b


def gaus(x, a, sigma, mu):
    return a / (_np.sqrt(2 * _np.pi) * sigma) * _np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def semicircle(x, b, R):
    circ = b / (_np.pi * R ** 2 / 2) * _np.sqrt(R ** 2 - x ** 2)
    _np.nan_to_num(circ, False, 0.0)
    return circ


def func_conv(X, A=1, sigma=3e-5, mu=0, R=1e-4):
    y_gaus = gaus(X, 1, sigma, mu)
    y_circ = semicircle(X, 1, R)

    conv = _np.convolve(y_gaus, y_circ, "same")
    conv = conv / simps(conv, X)

    func = interp1d(X, conv, fill_value='extrapolate')
    return A * func(X)


def analyticConvolution():
    x = _sp.Symbol("x", real=True)

    sigma = _sp.Symbol("sigma", real=True, positive=True)
    mu = _sp.Symbol("mu", real=True)
    g = 1 / (_sp.sqrt(2 * _sp.pi) * sigma) * _sp.exp(-(x-mu) ** 2 / (2 * sigma ** 2))

    R = _sp.Symbol("R", real=True, positive=True)
    w = 1 / (_sp.pi * R ** 2 / 2) * _sp.sqrt(R ** 2 - x ** 2)

    xt = _sp.Symbol("xt", real=True)
    print(_sp.integrate(g.subs(x, x - xt) * w.subs(x, xt), (xt, -_sp.oo, _sp.oo)))
    return 0

    # integ = _sp.sqrt(R**2-xt**2)*_sp.exp(-xt**2/(2*sigma**2))*_sp.exp(x*xt/sigma**2)
    # _sp.integrate(integ, (xt,  -_sp.oo, _sp.oo)


def GenerateOneGmadFile(gmadfilename, templatefilename, templatefolder="../03_bdsimModel/", paramdict=None):
    env = _jj.Environment(loader=_jj.FileSystemLoader(templatefolder))
    template = env.get_template(templatefilename)
    if paramdict is None:
        raise ValueError("No dictionary is provided to set the different parameters in file {}".format(templatefolder+templatefilename))
    f = open(templatefolder+gmadfilename, 'w')
    f.write(template.render(paramdict))
    f.close()


def GenerateSetGmadFiles(tag="T20_for_wire", X0=0, Xp0=0, Y0=0, Yp0=0,
                         sigmaX=10e-6, sigmaXp=10e-6, sigmaY=10e-6, sigmaYp=10e-6, sigmaT=100e-15, sigmaE=1e-6, energy=14,
                         wireDiameter=0.5, wireLength=0.03, material='tungsten', wireOffsetX='+0.00',
                         T=300, density=1e-12, xsecfact='5e0', printPhysicsProcesses=0, checkOverlaps=0, line='l6, l7'):
    extendedtag = tag+"_with_offset_"+wireOffsetX+"_bias_"+xsecfact
    beamdict = dict(X0=X0, Xp0=Xp0, Y0=Y0, Yp0=Yp0, sigmaX=sigmaX, sigmaXp=sigmaXp, sigmaY=sigmaY, sigmaYp=sigmaYp, sigmaT=sigmaT, sigmaE=sigmaE, energy=energy)
    componentdict = dict(wireDiameter=wireDiameter, wireLength=wireLength, material=material, wireOffsetX=float(wireOffsetX))
    optiondict = dict(printPhysicsProcesses=printPhysicsProcesses, checkOverlaps=checkOverlaps)
    GenerateOneGmadFile(extendedtag+".gmad", "T20_for_wire_template.gmad", paramdict=dict(tag=extendedtag))
    GenerateOneGmadFile(extendedtag+"_beam.gmad", "T20_for_wire_beam_template.gmad", paramdict=beamdict)
    GenerateOneGmadFile(extendedtag+"_components.gmad", "T20_for_wire_components_template.gmad", paramdict=componentdict)
    GenerateOneGmadFile(extendedtag+"_material.gmad", "T20_for_wire_material_template.gmad", paramdict=dict(T=T, density=density))
    GenerateOneGmadFile(extendedtag+"_objects.gmad", "T20_for_wire_objects_template.gmad", paramdict=dict(xsecfact=float(xsecfact)))
    GenerateOneGmadFile(extendedtag+"_options.gmad", "T20_for_wire_options_template.gmad", paramdict=optiondict)
    GenerateOneGmadFile(extendedtag+"_sequence.gmad", "T20_for_wire_sequence_template.gmad", paramdict=dict(line=line))

    return extendedtag


def GenerateAllGmadFilesAndList(tag="T20_for_wire", valuetoscan='wireOffsetX',
                                valuelist=['-0.50', '-0.40', '-0.30', '-0.20', '-0.10', '+0.00', '+0.10', '+0.20', '+0.30', '+0.40', '+0.50']):
    tagfilelistwire = open("tagfilelistwire", "w")
    for val in valuelist:
        paramdict = {valuetoscan: val}
        tagfilelistwire.write(GenerateSetGmadFiles(tag=tag, **paramdict)+'\n')


def runOneOffset(inputfilename, outputfilename=None, npart=100, seed=None, silent=False):
    if outputfilename is None:
        outputfilename = inputfilename.replace("../03_bdsimModel/", "../04_dataLocal/{}_part_".format(npart)).replace(".gmad", "")
    if seed is not None:
        _bd.Run.Bdsim(inputfilename, outputfilename, ngenerate=npart, options="--seed={}".format(seed), silent=silent)
    else:
        _bd.Run.Bdsim(inputfilename, outputfilename, ngenerate=npart, silent=silent)


def runScanOffset(tagfilelist="tagfilelistwire", npart=100, seed=None, silent=False):
    taglist = open(tagfilelist)
    nbpts = len(taglist.readlines())
    taglist.close()
    taglist = open(tagfilelist)
    for i, tag in enumerate(taglist):
        file = "../03_bdsimModel/" + tag.replace('\n', '.gmad')
        _printProgressBar(i, nbpts, prefix='Loading file {}. Run {} particles:'.format(file, npart), suffix='Complete', length=50)
        runOneOffset(file, npart=npart, seed=seed, silent=silent)
    taglist.close()
    _printProgressBar(nbpts, nbpts, prefix='Loading of {} files done. Run {} particles:'.format(nbpts, npart), suffix='Complete', length=50)


def analysisFilelist(tagfilelistwire):
    taglist = open(tagfilelistwire)
    for tag in taglist:
        analysis(_gl.glob('../04_dataLocal/*'+tag.replace('\n', '')+'*.root')[0])
        farmfilelist = _gl.glob('../05_dataFarm/*'+tag.replace('\n', '')+'*.root')
        for file in farmfilelist:
            analysis(file)
    taglist.close()


def analysis(inputfilename, nbins=50):
    if type(inputfilename) == list:
        for file in inputfilename:
            analysis(file)
        return 0

    tag = inputfilename.split('/')[-1].split('.root')[0].split('part_')[-1]

    data = _bd.Data.Load(inputfilename)
    e = data.GetEvent()
    et = data.GetEventTree()
    npart = et.GetEntries()
    sampler_data = e.GetSampler("DRIFT.")

    print("File :", inputfilename, " / Nb of entries = ", et.GetEntries())

    HIST_DICT = {}

    HIST_DICT['PHOTONS_X'] =     _rt.TH1D('PHOTONS_X',      "{} Photons wrt x at sampler".format(tag), nbins, -1e-3, 1e-3)
    HIST_DICT['PHOTONS_Y'] =     _rt.TH1D('PHOTONS_Y',      "{} Photons wrt y at sampler".format(tag), nbins, -1e-3, 1e-3)

    HIST_DICT['PHOTONS_R'] =     _rt.TH1D('PHOTONS_R',      "{} Photons wrt R at sampler".format(tag), nbins, 0, 1e-3)
    HIST_DICT['PHOTONS_Theta'] = _rt.TH1D('PHOTONS_Theta',  "{} Photons wrt theta at sampler".format(tag), nbins, 0, 1e-3)
    HIST_DICT['PHOTONS_E'] =     _rt.TH1D('PHOTONS_E',      "{} Photons wrt energy at sampler".format(tag), nbins, 0, 14)

    HIST_DICT['PHOTONS_R_cut']     = _rt.TH1D('PHOTONS_R_cut',      "{} Photons wrt R at sampler cutted".format(tag), nbins, 0.1, 0.5)
    HIST_DICT['PHOTONS_Theta_cut'] = _rt.TH1D('PHOTONS_Theta_cut',  "{} Photons wrt theta at sampler cutted".format(tag), nbins, 0.1, 0.5)
    HIST_DICT['PHOTONS_E_cut']     = _rt.TH1D('PHOTONS_E_cut',      "{} Photons wrt energy at sampler cutted".format(tag), nbins, 2, 14)

    HIST_DICT['PHOTONS_CUT'] = _rt.TH1D('PHOTONS_CUT', "{} Photons wrt theta at sampler cutted with enegy cut".format(tag), nbins, 0.1, 0.5)

    for evt in et:
        if len(sampler_data.weight) != 0:
            for i, partID in enumerate(sampler_data.partID):
                if partID == 22:
                    HIST_DICT['PHOTONS_X'].Fill(sampler_data.x[i], sampler_data.weight[i])
                    HIST_DICT['PHOTONS_Y'].Fill(sampler_data.y[i], sampler_data.weight[i])

                    R = _np.sqrt(sampler_data.x[i]**2 + sampler_data.y[i]**2)
                    theta = _np.arcsin(R)

                    HIST_DICT['PHOTONS_R'].Fill(R, sampler_data.weight[i])
                    HIST_DICT['PHOTONS_Theta'].Fill(theta, sampler_data.weight[i])
                    HIST_DICT['PHOTONS_E'].Fill(sampler_data.energy[i], sampler_data.weight[i])
                    if R >= 0.1:
                        HIST_DICT['PHOTONS_R_cut'].Fill(R, sampler_data.weight[i])
                        HIST_DICT['PHOTONS_Theta_cut'].Fill(theta, sampler_data.weight[i])
                    if sampler_data.energy[i] >= 2:
                        HIST_DICT['PHOTONS_E_cut'].Fill(sampler_data.energy[i], sampler_data.weight[i])
                    if R >= 0.1 and sampler_data.energy[i] >= 2:
                        HIST_DICT['PHOTONS_CUT'].Fill(sampler_data.energy[i], sampler_data.weight[i])

    for hist in HIST_DICT:
        HIST_DICT[hist].Scale(ELECTRONS_PER_BUNCH/npart)

    outputfilename = inputfilename.replace('04_dataLocal', '06_analysis').replace('05_dataFarm', '06_analysis').replace('.root', '')
    outfile = _bd.Data.CreateEmptyRebdsimFile('{}_hist.root'.format(outputfilename), data.header.nOriginalEvents)
    _bd.Data.WriteROOTHistogramsToDirectory(outfile, "Event/MergedHistograms", list(HIST_DICT.values()))
    outfile.Close()


def combineHistFiles(tag):
    globstring = "../06_analysis/*" + tag + "*T20_for_wire_hist.root"
    filelist = _gl.glob(globstring)
    if not filelist:
        raise FileNotFoundError("Glob did not find any files")
    npart = 0
    for filename in filelist:
        npart += int(filename.split('/')[-1].split('_part')[0].split('_')[-1])
    outputfile = "{}_part".format(npart) + filelist[0].split('_part')[-1].replace('hist', 'merged_hist')
    _sub.call('rebdsimCombine ' + outputfile + ' ' + globstring, shell=True)


def combineAllHistFiles(tagfilelist):
    taglist = open(tagfilelist)
    for tag in taglist:
        combineHistFiles(tag)


def countPhotons(inputfilename):
    data = _bd.Data.Load(inputfilename)
    e = data.GetEvent()
    et = data.GetEventTree()
    npart = et.GetEntries()
    sampler_data = e.GetSampler("DRIFT.")

    nbphotons = 0
    for evt in et:
        for i, partID in enumerate(sampler_data.partID):
            if partID == 22:
                nbphotons += sampler_data.weight[i]
    return nbphotons * (ELECTRONS_PER_BUNCH/npart)


def countPhotonsAllFiles(tag):
    filelist = _gl.glob('../04_dataLocal/*' + tag + '*.root')
    OFFSETS = []
    NPHOTONS = []
    for file in filelist:
        OFFSETS.append(float(file.replace('.root', '').replace('../04_dataLocal/{}_'.format(tag), '')))
        NPHOTONS.append(countPhotons(file))

    OFFSETS_sorted = [x for x, _ in sorted(zip(OFFSETS, NPHOTONS))]
    NPHOTONS_sorted = [y for _, y in sorted(zip(OFFSETS, NPHOTONS))]
    return OFFSETS_sorted, NPHOTONS_sorted


def countPhotonsInHist(inputfilename, histname):
    f = _rt.TFile(inputfilename)
    test_bd_load = _bd.Data.Load(inputfilename)
    npart = test_bd_load.header.nOriginalEvents
    root_hist = f.Get("Event/MergedHistograms/" + histname)
    python_hist = _bd.Data.TH1(root_hist)

    nbphotons = sum(python_hist.contents)
    error = _np.sqrt(sum(python_hist.errors**2))
    return nbphotons, error


def countPhotonsInHistAllFiles(tag, histname):
    filelist = _gl.glob(tag)
    OFFSETS = []
    NPHOTONS = []
    ERRORS = []
    for file in filelist:
        OFFSETS.append(float(file.split('_offset_')[-1].split('_')[0]))
        nphotons, error = countPhotonsInHist(file, histname)
        NPHOTONS.append(nphotons)
        ERRORS.append(error)

    OFFSETS_sorted =    [x for x, _, _ in sorted(zip(OFFSETS, NPHOTONS, ERRORS))]
    NPHOTONS_sorted =   [y for _, y, _ in sorted(zip(OFFSETS, NPHOTONS, ERRORS))]
    ERRORS_sorted =     [z for _, _, z in sorted(zip(OFFSETS, NPHOTONS, ERRORS))]
    return OFFSETS_sorted, NPHOTONS_sorted, ERRORS_sorted


def PlotConvolutionExample():
    _plt.rcParams['font.size'] = 17
    fig, ax = _plt.subplots(1, 1, figsize=(9, 4))
    fig.tight_layout()

    X = _np.linspace(-0.5e-3, 0.5e-3, 500)
    _plt.plot(X, gaus(X, a=1, sigma=87e-6, mu=0), label='beam')
    _plt.plot(X, semicircle(X, b=1, R=250e-6), label='wire')
    _plt.plot(X, func_conv(X, A=1, sigma=87e-6, mu=0, R=250e-6), ls='-', label='convolution')
    _plt.legend()


def PlotConvolution(OFFSETS, NPHOTONS, ERRORS, A=None, sigma=50e-3, mu=0, wireRadius=250e-3, manualFit=False):
    _plt.rcParams['font.size'] = 17
    fig, ax = _plt.subplots(1, 1, figsize=(9, 6))
    fig.tight_layout()

    _plt.errorbar(OFFSETS, NPHOTONS, yerr=ERRORS, fmt="k", elinewidth=1, capsize=3, label='data')

    X = _np.linspace(-0.5, 0.5, 500)

    if A is None:
        A = max(NPHOTONS)/3
    if manualFit:
        _plt.plot(X, func_conv(X, A=A, sigma=sigma, mu=mu, R=wireRadius), '-', color="C0",
                  label='conv manual: $\sigma$ = {:1.2e} / $R$ = {:1.2e}'.format(sigma, wireRadius))

    popt, pcov = curve_fit(func_conv, OFFSETS, NPHOTONS, p0=[A, sigma, mu, wireRadius])
    _plt.plot(X, func_conv(X, A=popt[0], sigma=popt[1], mu=popt[2], R=popt[3]), '-', color="C2",
              label='conv fit : $\sigma$ = {:1.2e} / $R$ = {:1.2e}'.format(popt[1], popt[3]))

    popt, pcov = curve_fit(lambda x, _A, _sigma, _mu: func_conv(x, _A, _sigma, _mu, R=wireRadius), OFFSETS, NPHOTONS, p0=[A, sigma, mu])
    _plt.plot(X, func_conv(X, A=popt[0], sigma=popt[1], mu=popt[2], R=wireRadius), '-', color="C3",
              label='conv fit (fixed R): $\sigma$ = {:1.2e} / $R$ = {:1.2e}'.format(popt[1], wireRadius))

    _plt.xlabel('Offset from center of pipe [mm]')
    _plt.ylabel('Number of photons')
    _plt.legend(fontsize="15", loc=9)


def plot_hist(inputfilename, histname, errorbars=False, steps=True, fitFunction=None, fitRange=None,
              yLogScale=False, color=None, printLegend=True):
    f = _rt.TFile(inputfilename)
    test_bd_load = _bd.Data.Load(inputfilename)
    npart = test_bd_load.header.nOriginalEvents
    root_hist = f.Get("Event/MergedHistograms/"+histname)
    python_hist = _bd.Data.TH1(root_hist)

    title = python_hist.hist.GetTitle()
    centres = python_hist.xcentres
    contents = python_hist.contents
    errors = python_hist.errors
    widths = python_hist.xwidths

    _plt.rcParams['font.size'] = 17
    fig, ax = _plt.subplots(1, 1, figsize=(9, 6))
    fig.tight_layout()

    if errorbars:
        _plt.errorbar(centres, contents, yerr=errors, xerr=widths * 0.5, ls='', marker='+', color=color, label=title)
    if steps:
        _plt.step(centres, contents, where='mid', color=color, label=title)

    #empty_bins = []
    #for i, val in enumerate(contents):
    #    if val == 0:
    #        empty_bins.append(i)
    #centres = _np.delete(centres, empty_bins)
    #contents = _np.delete(contents, empty_bins)
    #errors = _np.delete(errors, empty_bins)

    if fitRange:
        centres = centres[fitRange[0]:fitRange[1]]
        contents = contents[fitRange[0]:fitRange[1]]
        errors = errors[fitRange[0]:fitRange[1]]

    if fitFunction is not None:
        popt, pcov = curve_fit(fitFunction, centres, contents, sigma=errors, absolute_sigma=True)
        _plt.plot(centres, fitFunction(centres, *popt), ls='--', color=color, label='Fit')

    if yLogScale:
        _plt.yscale("log")
    if printLegend:
        _plt.legend()

    _plt.xlabel(histname.split('_')[1])
    _plt.ylabel(histname.split('_')[0])
