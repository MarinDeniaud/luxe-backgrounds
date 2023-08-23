import sys
import os
import pybdsim as _bd
import ROOT as _rt
import numpy as _np
import matplotlib.pyplot as _plt
import glob as _gl
import sympy as _sp
import jinja2 as _jj
import subprocess as _sub
from scipy.optimize import curve_fit
from scipy.integrate import simps
from scipy.interpolate import interp1d
from scipy import integrate


def plotOptions(figsize=[9, 6], rows_colums=[1, 1], font_size=17):
    _plt.rcParams['font.size'] = font_size
    fig, ax = _plt.subplots(rows_colums[0], rows_colums[1], figsize=(figsize[0], figsize[1]))
    fig.tight_layout()


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


def GenerateLinearListValuesStr(minValue=-0.1, maxValue=0.1, nbpts=11, nbDecimals=1, exponant=None):
    floatlist = _np.linspace(minValue, maxValue, nbpts)
    strlist = []
    for elem in floatlist:
        elem = round(elem, nbDecimals)
        strlist.append(FormatStrExponant(FormatStrDecimals(FormatStrSign(str(elem)), nbDecimals), exponant))
    return strlist


def FormatStrSign(string):
    if float(string) >= 0:
        return '+' + string
    return string


def FormatStrDecimals(string, nbDecimals=2):
    integer, decimals = string.split('.')
    if len(decimals) < nbDecimals:
        return string + '0' * (nbDecimals - len(decimals))
    return integer + '.' + decimals[:nbDecimals]


def FormatStrExponant(string, exponant=None):
    if exponant is not None:
        return string + 'e' + str(exponant)
    return string


def GenerateLogListValuesStr(minExponant=0, maxExponant=5, nbpts=6, nbDecimals=1, negative=False):
    floatlist = _np.logspace(minExponant, maxExponant, nbpts)
    strlist = []
    if negative:
        floatlist = _np.negative(floatlist)
    for elem in floatlist:
        strformat = '{:1.' + str(nbDecimals) + 'e}'
        strlist.append(FormatStrSign(strformat.format(elem)))
    return strlist


def linear(x, a, b):
    return a * x + b


def gauss(x, a, sigma, mu):
    return a / (_np.sqrt(2 * _np.pi) * sigma) * _np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def needleprofile(x, c, R1, R2, length):
    x1 = 0
    x2 = length
    slope = (R2-R1)/(x2-x1)
    intercept = R1 - slope*x1
    surface = (R1 + 0.5*_np.abs(R2-R1))*_np.abs(x1-x2)
    return _np.piecewise(x, [x < x1, x > x1, x > x2], [0, lambda xx: c / surface * (slope * xx + intercept), 0])


def func_conv_needle(X, A=1, sigma=3e-5, mu=0, R1=50e-6, R2=200e-6, length=10e-3):
    y_gaus = gauss(X, 1, sigma, mu)
    y_needle = needleprofile(X, 1, R1, R2, length)

    conv = _np.convolve(y_gaus, y_needle, "same")
    conv = conv / simps(conv, X)

    func = interp1d(X, conv, fill_value='extrapolate')
    return A * func(X)


def gauss2D(x, y, a, sigmax, sigmay, mux, muy):
    return a / (_np.sqrt(2 * _np.pi) * sigmax * sigmay) * _np.exp(-(x - mux) ** 2 / (2 * sigmax ** 2)) * _np.exp(-(y - muy) ** 2 / (2 * sigmay ** 2))


def needle2D(x, y, b, length, R1, R2):
    r = (R1 - R2)/length * y + R2
    n = 2*_np.sqrt(r**2 - x**2)
    try:
        n[y > length] = 0
        n[y < 0] = 0
        _np.nan_to_num(n, False, 0.0)
    except:
        if y > length or y < 0 or _np.isnan(n):
            n = 0
    return b*n


def gaussNeedleProduct(x, y, a=1, sigmax=1, sigmay=1, mux=0, muy=0, b=1, length=6, R1=1, R2=0):
    return gauss2D(x, y, a, sigmax, sigmay, mux, muy) * needle2D(x, y, b, length, R1, R2)


def integrate_needle_beam_2D(a=1, sigmax=1, sigmay=1, mux=0, muy=0, b=1, length=6, R1=1, R2=0):
    i, e = integrate.dblquad(gaussNeedleProduct, -length, length, -length, length, args=(a, sigmax, sigmay, mux, muy, b, length, R1, R2))
    return i, e


def integrate_needle_beam_2D_by_X(mux, a=1, sigmax=1, sigmay=1, muy=0, b=1, length=6, R1=1, R2=0):
    I = []
    for mu in mux:
        I.append(integrate.dblquad(gaussNeedleProduct, -length, length, -length, length, args=(a, sigmax, sigmay, mu, muy, b, length, R1, R2))[0])
    func = interp1d(mux, I, fill_value='extrapolate')
    return func(mux)


def integrate_needle_beam_2D_by_Y(muy, a=1, sigmax=1, sigmay=1, mux=0, b=1, length=6, R1=1, R2=0):
    I = []
    for mu in muy:
        I.append(integrate.dblquad(gaussNeedleProduct, -length, length, -length, length, args=(a, sigmax, sigmay, mux, mu, b, length, R1, R2))[0])
    func = interp1d(muy, I, fill_value='extrapolate')
    return func(muy)


def numerical_needle_scan(param='mux', scan_min=-3, scan_max=3, a=1, sigmax=1, sigmay=1, mux=0, muy=0, b=1, length=6, R1=1, R2=0, nbpts=20):
    MU = _np.linspace(scan_min, scan_max, nbpts)
    Integrals = []
    Errors = []
    for mu in MU:
        if param == 'mux':
            i, e = integrate_needle_beam_2D(a, sigmax, sigmay, mu, muy, b, length, R1, R2)
        elif param == 'muy':
            i, e = integrate_needle_beam_2D(a, sigmax, sigmay, mux, mu, b, length, R1, R2)
        else:
            raise ValueError("Parameter to scan should be 'mux' or 'muy'")
        Integrals.append(i)
        Errors.append(e)
    return MU, Integrals, Errors


def GenerateOneGmadFile(gmadfilename, templatefilename, templatefolder="../03_bdsimModel/", paramdict=None):
    env = _jj.Environment(loader=_jj.FileSystemLoader(templatefolder))
    template = env.get_template(templatefilename)
    if paramdict is None:
        raise ValueError("No dictionary is provided to set the different parameters in file {}".format(templatefolder+templatefilename))
    f = open(templatefolder+gmadfilename, 'w')
    f.write(template.render(paramdict))
    f.close()


def GenerateSetGmadFiles(tag="T20_needle", X0=0, Xp0=0, Y0=0, Yp0=0, distrType='gausstwiss',
                         alfx=0, alfy=0, betx=0.2, bety=3, dispx=0, dispxp=0, dispy=0, dispyp=0, emitx=3.58e-11, emity=3.58e-11,
                         sigmaX=10e-6, sigmaXp=10e-6, sigmaY=10e-6, sigmaYp=10e-6, sigmaT=100e-15, sigmaE=1e-6, energy=14,
                         needleOffsetX='+0.00', needleOffsetY='+0.00',
                         T=300, density=1e-12, xsecfact='1e0', printPhysicsProcesses=0, checkOverlaps=0, line='l0, l1, l2, l3, l4, l5, l6, l7'):
    extendedtag = tag + "_X_" + needleOffsetX + "_Y_" + needleOffsetY
    template_tag = "T20_needle_template"
    beamdict = dict(X0=X0, Xp0=Xp0, Y0=Y0, Yp0=Yp0, distrType=distrType,
                    alfx=alfx, alfy=alfy, betx=betx, bety=bety, dispx=dispx, dispxp=dispxp, dispy=dispy, dispyp=dispyp, emitx=emitx, emity=emity,
                    sigmaX=sigmaX, sigmaXp=sigmaXp, sigmaY=sigmaY, sigmaYp=sigmaYp, sigmaT=sigmaT, sigmaE=sigmaE, energy=energy)
    componentdict = dict(needleOffsetX=float(needleOffsetX), needleOffsetY=float(needleOffsetY))
    optiondict = dict(printPhysicsProcesses=printPhysicsProcesses, checkOverlaps=checkOverlaps)
    GenerateOneGmadFile(extendedtag+".gmad",            template_tag+".gmad",               paramdict=dict(tag=extendedtag))
    GenerateOneGmadFile(extendedtag+"_beam.gmad",       template_tag+"_beam.gmad",          paramdict=beamdict)
    GenerateOneGmadFile(extendedtag+"_components.gmad", template_tag+"_components.gmad",    paramdict=componentdict)
    GenerateOneGmadFile(extendedtag+"_material.gmad",   template_tag+"_material.gmad",      paramdict=dict(T=T, density=density))
    GenerateOneGmadFile(extendedtag+"_objects.gmad",    template_tag+"_objects.gmad",       paramdict=dict(xsecfact=float(xsecfact)))
    GenerateOneGmadFile(extendedtag+"_options.gmad",    template_tag+"_options.gmad",       paramdict=optiondict)
    GenerateOneGmadFile(extendedtag+"_sequence.gmad",   template_tag+"_sequence.gmad",      paramdict=dict(line=line))

    return extendedtag


def GenerateAllGmadFilesAndList(tag="T20_wire", tagfilename='tagfilelistneedle', valuetoscan='wireOffsetX',
                                valuelist=['-0.50', '-0.40', '-0.30', '-0.20', '-0.10', '+0.00', '+0.10', '+0.20', '+0.30', '+0.40', '+0.50'],
                                **otherargs):
    taglist = open(tagfilename, "w")

    for val in valuelist:
        paramdict = {valuetoscan: val}
        for arg in otherargs:
            paramdict[arg] = otherargs[arg]
        taglist.write(GenerateSetGmadFiles(tag=tag, **paramdict)+'\n')
    taglist.close()

    print("File names written in {}".format(tagfilename))


def runOneOffset(inputfilename, outputfilename=None, npart=100, seed=None, silent=False):
    if outputfilename is None:
        outputfilename = inputfilename.replace("../03_bdsimModel/", "../04_dataLocal/{}_part_".format(npart)).replace(".gmad", "")
    if seed is not None:
        _bd.Run.Bdsim(inputfilename, outputfilename, ngenerate=npart, options="--seed={}".format(seed), silent=silent)
    else:
        _bd.Run.Bdsim(inputfilename, outputfilename, ngenerate=npart, silent=silent)


def runScanOffset(tagfilename="tagfilelistneedle", npart=100, seed=None, silent=False):
    taglist = open(tagfilename)
    nbpts = len(taglist.readlines())
    taglist.close()
    taglist = open(tagfilename)
    for i, tag in enumerate(taglist):
        file = "../03_bdsimModel/" + tag.replace('\n', '.gmad')
        _printProgressBar(i, nbpts, prefix='Run BDSIM on file {} with {} particles:'.format(file, npart), suffix='Complete', length=50)
        runOneOffset(file, npart=npart, seed=seed, silent=silent)
    taglist.close()
    _printProgressBar(nbpts, nbpts, prefix='Run BDSIM on file {} with {} particles:'.format(file, npart), suffix='Complete', length=50)
    print("Succesfull BDSIM run for {} files with {} particles".format(nbpts, npart))


def analysisFilelist(tagfilelist):
    taglist = open(tagfilelist)
    for tag in taglist:
        analysis(_gl.glob('../04_dataLocal/*'+tag.replace('\n', '')+'*.root')[0])
        farmfilelist = _gl.glob('../05_dataFarm/*'+tag.replace('\n', '')+'*.root')
        for file in farmfilelist:
            analysis(file)
    taglist.close()


def analysis(inputfilename, nbins=50, ELECTRONS_PER_BUNCH = 2e9):
    if type(inputfilename) == list:
        nb_files = len(inputfilename)
        for i, file in enumerate(inputfilename):
            _printProgressBar(i, nb_files, prefix='Run REBDSIM analysis on file {}:'.format(file), suffix='Complete', length=50)
            save_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            analysis(file)
            sys.stdout = save_stdout
        _printProgressBar(nb_files, nb_files, prefix='Run REBDSIM analysis on file {}:'.format(file), suffix='Complete', length=50)
        print("Succesfull REBDSIM analysis for {} files".format(nb_files))
        return 0

    tag = inputfilename.split('/')[-1].split('.root')[0].split('part_')[-1]

    data = _bd.Data.Load(inputfilename)
    e = data.GetEvent()
    et = data.GetEventTree()
    npart = et.GetEntries()
    sampler_data = e.GetSampler("DRIFT.")

    print("File :", inputfilename, " / Nb of entries = ", et.GetEntries())

    HIST_DICT = {}

    HIST_DICT['PHOTONS_X']          = _rt.TH1D('PHOTONS_X',      "{} Photons wrt x at sampler".format(tag), nbins, -1e-3, 1e-3)
    HIST_DICT['PHOTONS_Y']          = _rt.TH1D('PHOTONS_Y',      "{} Photons wrt y at sampler".format(tag), nbins, -1e-3, 1e-3)

    HIST_DICT['PHOTONS_X_Y'] = _rt.TH2D('PHOTONS_X_Y', r"{} X-Y photons correl at sampler".format(tag),
                                        nbins, -1e-3, 1e-3,
                                        nbins, -1e-3, 1e-3)

    HIST_DICT['PHOTONS_R']          = _rt.TH1D('PHOTONS_R',      "{} Photons wrt R at sampler".format(tag), nbins, 0, 1e-3)
    HIST_DICT['PHOTONS_Theta']      = _rt.TH1D('PHOTONS_Theta',  "{} Photons wrt theta at sampler".format(tag), nbins, 0, 1e-3)
    HIST_DICT['PHOTONS_E']          = _rt.TH1D('PHOTONS_E',      "{} Photons wrt energy at sampler".format(tag), nbins, 0, 14)

    HIST_DICT['PHOTONS_E_Theta'] = _rt.TH2D('PHOTONS_E_Theta', r"{} E-$\theta$ photons correl at sampler".format(tag),
                                            nbins, 0, 14,
                                            nbins, 0, 1e-3)

    HIST_DICT['PHOTONS_E_Theta_log']    = _rt.TH2D('PHOTONS_E_Theta_log', r"{} E-$\theta$ photons correl at sampler".format(tag),
                                                   nbins, _np.logspace(-4, 2, nbins+1),
                                                   nbins, _np.logspace(-6, -2, nbins+1))

    HIST_DICT['PHOTONS_R_cut']      = _rt.TH1D('PHOTONS_R_cut',      "{} Photons wrt R at sampler cutted".format(tag), nbins, 0.1, 0.5)
    HIST_DICT['PHOTONS_Theta_cut']  = _rt.TH1D('PHOTONS_Theta_cut',  "{} Photons wrt theta at sampler cutted".format(tag), nbins, 0.1, 0.5)
    HIST_DICT['PHOTONS_E_cut']      = _rt.TH1D('PHOTONS_E_cut',      "{} Photons wrt energy at sampler cutted".format(tag), nbins, 2, 14)

    HIST_DICT['PHOTONS_CUT']        = _rt.TH1D('PHOTONS_CUT', "{} Photons wrt theta at sampler cutted with enegy cut".format(tag), nbins, 0.1, 0.5)

    for evt in et:
        if len(sampler_data.weight) != 0:
            for i, partID in enumerate(sampler_data.partID):
                if partID == 22:
                    HIST_DICT['PHOTONS_X'].Fill(sampler_data.x[i], sampler_data.weight[i])
                    HIST_DICT['PHOTONS_Y'].Fill(sampler_data.y[i], sampler_data.weight[i])

                    HIST_DICT['PHOTONS_X_Y'].Fill(sampler_data.x[i], sampler_data.y[i], sampler_data.weight[i])

                    R = _np.sqrt(sampler_data.x[i]**2 + sampler_data.y[i]**2)
                    theta = _np.arcsin(R)

                    HIST_DICT['PHOTONS_R'].Fill(R, sampler_data.weight[i])
                    HIST_DICT['PHOTONS_Theta'].Fill(theta, sampler_data.weight[i])
                    HIST_DICT['PHOTONS_E'].Fill(sampler_data.energy[i], sampler_data.weight[i])

                    HIST_DICT['PHOTONS_E_Theta'].Fill(sampler_data.energy[i], theta, sampler_data.weight[i])
                    HIST_DICT['PHOTONS_E_Theta_log'].Fill(sampler_data.energy[i], theta, sampler_data.weight[i])

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
    globstring = "../06_analysis/*" + tag + "*_hist.root"
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


def countPhotons(inputfilename, ELECTRONS_PER_BUNCH = 2e9):
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


def countPhotonsInHistAllFiles(tag, histname, coord):
    filelist = _gl.glob(tag)
    OFFSETS = []
    NPHOTONS = []
    ERRORS = []
    nb_files = len(filelist)
    for i, file in enumerate(filelist):
        save_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        if coord == "X" or coord == "Y":
            OFFSETS.append(float(file.split('_{}_'.format(coord))[-1].split('_')[0]))
        else:
            raise ValueError("Unknown value {} for parameter 'coord'".format(coord))
        nphotons, error = countPhotonsInHist(file, histname)
        NPHOTONS.append(nphotons)
        ERRORS.append(error)
        sys.stdout = save_stdout

    OFFSETS_sorted =    [x for x, _, _ in sorted(zip(OFFSETS, NPHOTONS, ERRORS))]
    NPHOTONS_sorted =   [y for _, y, _ in sorted(zip(OFFSETS, NPHOTONS, ERRORS))]
    ERRORS_sorted =     [z for _, _, z in sorted(zip(OFFSETS, NPHOTONS, ERRORS))]
    return _np.array(OFFSETS_sorted), _np.array(NPHOTONS_sorted), _np.array(ERRORS_sorted)


def CalcVariance(inputfilename, histname):
    f = _rt.TFile(inputfilename)
    test_bd_load = _bd.Data.Load(inputfilename)
    npart = test_bd_load.header.nOriginalEvents
    root_hist = f.Get("Event/MergedHistograms/" + histname)
    python_hist = _bd.Data.TH1(root_hist)

    contents = python_hist.contents
    variance = contents.std() / contents.mean() * 100

    errors = python_hist.errors
    error_mean = 1 / python_hist.entries * _np.sqrt(sum(errors ** 2))
    error_std = contents.std() / _np.sqrt(python_hist.entries)
    variance_error = variance * _np.sqrt((error_std / contents.std()) ** 2 + (error_mean / contents.mean()) ** 2)

    return variance, variance_error


def CalcAllVariance(tag, histname):
    filelist = _gl.glob(tag)
    BIAS = []
    VAR = []
    ERR = []
    for file in filelist:
        BIAS.append(float(file.split('_bias_')[-1].split('_')[0]))
        variance, variance_error = CalcVariance(file, histname)
        VAR.append(variance)
        ERR.append(variance_error)

    BIAS_sorted = [x for x, _, _ in sorted(zip(BIAS, VAR, ERR))]
    VAR_sorted = [y for _, y, _ in sorted(zip(BIAS, VAR, ERR))]
    ERR_sorted = [z for _, _, z in sorted(zip(BIAS, VAR, ERR))]
    return BIAS_sorted, VAR_sorted, ERR_sorted


def PlotConvolutionExampleNeedle(a=1, sigma=10e-6, mu=0, c=1, R1=20e-6, R2=200e-6, length=10e-3, A=1, xmin=4e-3, xmax=6e-3):
    plotOptions()

    # Y = _np.linspace(xmin, xmax, 100)
    Y = _np.append(-_np.logspace(-3, -7, 100), _np.logspace(-7, -3, 100))
    _plt.plot(Y, gauss(Y, a=a, sigma=sigma, mu=mu), label='beam')
    _plt.plot(Y, needleprofile(Y, c=c, R1=R1, R2=R2, length=length), label='needle')
    _plt.plot(Y, func_conv_needle(Y, A=A, sigma=sigma, mu=mu, R1=R1, R2=R2, length=length), ls='-', label='convolution')
    _plt.legend()


def PlotIntegralNeedle(filename, histname, coord, a=None, sigmax=10e-6, sigmay=10e-6, mux=0, muy=0, b=None, length=1e-3, R1=7e-6, R2=125e-6,
                       manualFit=False, autoFit=False, autoFitFixed=False, linFit=False, fitRange=[-_np.inf, _np.inf]):
    plotOptions()

    OFFSETS, NPHOTONS, ERRORS = countPhotonsInHistAllFiles(filename, histname, coord)

    if coord == "X":
        integral_function = integrate_needle_beam_2D_by_X
    elif coord == "Y":
        integral_function = integrate_needle_beam_2D_by_Y
    else:
        raise ValueError("Unknown value {} for parameter coord".format(coord))

    if a is None:
        a = max(NPHOTONS)/3
    if b is None:
        b = max(NPHOTONS)/3

    OFFSETS_fit = OFFSETS[[fitRange[0] <= offset <= fitRange[1] for offset in OFFSETS]]
    NPHOTONS_fit = NPHOTONS[[fitRange[0] <= offset <= fitRange[1] for offset in OFFSETS]]
    MU = _np.linspace(min(OFFSETS_fit), max(OFFSETS_fit), 200)

    if linFit:
        popt, pcov = curve_fit(linear, OFFSETS_fit, NPHOTONS_fit, p0=[a, b])
        chi2 = sum((NPHOTONS_fit - a * OFFSETS_fit - b) ** 2/(a * OFFSETS_fit + b))
        _plt.plot(MU, linear(MU, a=popt[0], b=popt[1]), '-', color="C3",
                  label=r'fit (linear): a={:1.2e}/b={:1.2e}/$\chi^2$={:1.2e}'.format(popt[0], popt[1], chi2))

    if manualFit:
        _plt.plot(MU, integral_function(MU, a=a, sigmax=sigmax, sigmay=sigmay, mux=MU, muy=muy, b=b, length=length, R1=R1, R2=R2), '-', color="C0",
                  label=r'fit (manual): $\sigma_x$={:1.2e}/$\sigma_y$={:1.2e}/$R1$={:1.2e}/$R2$={:1.2e}'.format(sigmax, sigmay, R1, R2))

    if autoFit:
        popt, pcov = curve_fit(integral_function, OFFSETS_fit, NPHOTONS_fit, p0=[a, sigmax, sigmay, muy, b, length, R1, R2])
        _plt.plot(MU, integral_function(MU, a=popt[0], sigmax=popt[1], sigmay=popt[2], muy=popt[3], b=popt[4], length=popt[5], R1=popt[6], R2=popt[7]),
                  '-', color="C2", label=r'fit: $\sigma_x$={:1.2e}/$\sigma_y$={:1.2e}/$R1$={:1.2e}/$R2$={:1.2e}'.format(popt[1], popt[2], popt[6], popt[7]))

    if autoFitFixed:
        popt, pcov = curve_fit(lambda x, _a, _sigmax, _sigmay, _muy, _b: integral_function(x, _a, _sigmax, _sigmay, _muy, _b, length=length, R1=R1, R2=R2),
                               OFFSETS_fit, NPHOTONS_fit, p0=[a, sigmax, sigmay, muy, b])
        _plt.plot(MU, integral_function(MU, a=popt[0], sigmax=popt[1], sigmay=popt[2], muy=popt[3],b=popt[4], length=length, R1=R1, R2=R2),
                  '-', color="C3", label=r'fit (fixed needle): $\sigma_x$={:1.2e}/$\sigma_y$={:1.2e}'.format(popt[1], popt[2]))

    _plt.errorbar(OFFSETS, NPHOTONS, yerr=ERRORS, fmt="k", elinewidth=2, capsize=4, label='data')

    _plt.xlabel('Offset from center of pipe [m]')
    _plt.ylabel(r"$N_{photons}$")
    _plt.legend(fontsize="15", loc=9)


def plot_gauss_test():
    X = _np.linspace(-3, 3, 20)
    Y = _np.linspace(-3, 3, 20)
    XX, YY = _np.meshgrid(X, _np.flip(Y))
    G = gauss2D(XX, YY, 1, 1, 1, 0, 0)
    _plt.imshow(G, cmap='Blues')


def plot_needle_test(b=1, length=6, R1=1, R2=0, xmin=-3, xmax=3, ymin=-3, ymax=3, nbins=100):
    X = _np.linspace(xmin, xmax, nbins)
    Y = _np.linspace(ymin, ymax, nbins)
    XX, YY = _np.meshgrid(X, _np.flip(Y))
    N = needle2D(XX, YY, b, length, R1, R2)
    _plt.imshow(N, cmap='Reds')


def plot_gauss_needle_integral(a=1, sigmax=1, sigmay=1, mux=0, muy=0, b=1, length=6, R1=1, R2=0, xmin=-3, xmax=3, ymin=-3, ymax=3, nbins=100):
    i, e = integrate_needle_beam_2D(a, sigmax, sigmay, mux, muy, b, length, R1, R2)

    X = _np.linspace(xmin, xmax, nbins)
    Y = _np.linspace(ymin, ymax, nbins)
    XX, YY = _np.meshgrid(X, _np.flip(Y))
    G = gauss2D(XX, YY, a, sigmax, sigmay, mux, muy)
    N = needle2D(XX, YY, b, length, R1, R2)
    bounds = [xmin, xmax, ymin, ymax]

    _plt.imshow(G, cmap='Blues', extent=bounds)
    _plt.imshow(N, cmap='Reds', extent=bounds, alpha=_np.piecewise(N, [N > 0, N == 0], [1, 0]))

    _plt.xlabel('X [m]')
    _plt.ylabel('Y [m]')
    _plt.title(r"Integral : {:1.2e} $\pm$ {:1.2e}".format(i, e))


def plot_numerical_needle_scan(param='mux', scan_min=-3, scan_max=3, a=1, sigmax=1, sigmay=1, mux=0, muy=0, b=1, length=6, R1=1, R2=0, nbpts=20, color='C0',
                               label='needle_scan'):
    MU, I, E = numerical_needle_scan(param, scan_min, scan_max, a, sigmax, sigmay, mux, muy, b, length, R1, R2, nbpts)

    _plt.errorbar(MU, I, yerr=E, fmt=color, elinewidth=2, capsize=4, label=label)

    _plt.xlabel(param)
    _plt.ylabel('Integral value')
    _plt.legend()


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

    plotOptions()

    if errorbars:
        _plt.errorbar(centres, contents, yerr=errors, ls='', marker='+', elinewidth=2, capsize=4, color=color, label=title)
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


def plot_hist_2d(inputfilename, histname, xLogScale=False, yLogScale=False, zLogScale=False, xlabel=None, ylabel=None, zlabel=None):
    f = _rt.TFile(inputfilename)
    test_bd_load = _bd.Data.Load(inputfilename)
    npart = test_bd_load.header.nOriginalEvents
    root_hist = f.Get("Event/MergedHistograms/" + histname)
    python_hist = _bd.Data.TH2(root_hist)

    _plt.rcParams['font.size'] = 17
    fig = _bd.Plot.Histogram2D(python_hist, xLogScale=xLogScale, yLogScale=yLogScale, logNorm=zLogScale,
                               xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, title=str(npart), figsize=(9, 6))
    fig.tight_layout()


def plot_Theta_E(inputfilename, histname="PHOTONS_E_Theta", xLogScale=False, yLogScale=False, printLegend=True):
    f = _rt.TFile(inputfilename)
    test_bd_load = _bd.Data.Load(inputfilename)
    npart = test_bd_load.header.nOriginalEvents
    root_hist = f.Get("Event/MergedHistograms/" + histname)
    python_hist = _bd.Data.TH2(root_hist)

    title = python_hist.hist.GetTitle()
    xcentres = python_hist.xcentres
    ycentres = python_hist.ycentres
    contents = python_hist.contents
    errors = python_hist.errors

    E = xcentres
    Theta = []
    Errors = []
    for i in range(len(xcentres)):
        Theta_temp = 0
        Errors_temp = 0
        nb = 0
        for j in range(len(ycentres)):
            if contents[i][j] != 0:
                Theta_temp += (ycentres[j] * contents[i][j])
                nb += contents[i][j]
                Errors_temp += (ycentres[j] * errors[i][j])**2
        if nb == 0:
            nb = 1
        Theta.append(Theta_temp/nb)
        Errors.append(_np.sqrt(Errors_temp)/nb)

    plotOptions()

    _plt.step(E, Theta, where='mid', color="C0")
    _plt.errorbar(E, Theta, yerr=Errors, ls='', marker='+', color="C0", elinewidth=2, capsize=4, label=title)

    if xLogScale:
        _plt.xscale("log")
    if yLogScale:
        _plt.yscale("log")
    if printLegend:
        _plt.legend()


def plot_var(tag, histname, errorbars=False, steps=True, xLogScale=False, yLogScale=False, color=None, printLegend=True):
    BIAS, VAR, ERR = CalcAllVariance(tag, histname)

    plotOptions()

    if errorbars:
        _plt.errorbar(BIAS, VAR, yerr=ERR, ls='', marker='+', color=color, elinewidth=2, capsize=4, label=tag)
    if steps:
        _plt.step(BIAS, VAR, where='mid', color=color, label=tag)

    if xLogScale:
        _plt.xscale("log")
    if yLogScale:
        _plt.yscale("log")
    if printLegend:
        _plt.legend()
