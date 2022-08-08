import pybdsim as _bd
import ROOT as _rt
import numpy as _np
import matplotlib.pyplot as _plt
from scipy.optimize import curve_fit

ELECTRONS_PER_BUNCH = 2e9

FILES_DICT = {'Bias_0.025': {'filename': 'T20_bias_0.025_output.root', 'histname': 'T20_bias_0.025_output_hist.root', 'tag': 'T20_bias_0.025', 'factor': 0.025},
              'Bias_0.05':  {'filename': 'T20_bias_0.05_output.root',  'histname': 'T20_bias_0.05_output_hist.root',  'tag': 'T20_bias_0.05',  'factor': 0.05},
              'Bias_0.075': {'filename': 'T20_bias_0.075_output.root', 'histname': 'T20_bias_0.075_output_hist.root', 'tag': 'T20_bias_0.075', 'factor': 0.075},
              'Bias_0.1':   {'filename': 'T20_bias_0.1_output.root',   'histname': 'T20_bias_0.1_output_hist.root',   'tag': 'T20_bias_0.1',   'factor': 0.1},
              'Bias_0.2':   {'filename': 'T20_bias_0.2_output.root',   'histname': 'T20_bias_0.2_output_hist.root',   'tag': 'T20_bias_0.2',   'factor': 0.2},
              'Bias_0.3':   {'filename': 'T20_bias_0.3_output.root',   'histname': 'T20_bias_0.3_output_hist.root',   'tag': 'T20_bias_0.3',   'factor': 0.3},
              'Bias_0.4':   {'filename': 'T20_bias_0.4_output.root',   'histname': 'T20_bias_0.4_output_hist.root',   'tag': 'T20_bias_0.4',   'factor': 0.4},
              'Bias_0.5':   {'filename': 'T20_bias_0.5_output.root',   'histname': 'T20_bias_0.5_output_hist.root',   'tag': 'T20_bias_0.5',   'factor': 0.5},
              'Bias_0.6':   {'filename': 'T20_bias_0.6_output.root',   'histname': 'T20_bias_0.6_output_hist.root',   'tag': 'T20_bias_0.6',   'factor': 0.6},
              'Bias_0.7':   {'filename': 'T20_bias_0.7_output.root',   'histname': 'T20_bias_0.7_output_hist.root',   'tag': 'T20_bias_0.7',   'factor': 0.7},
              'Bias_0.8':   {'filename': 'T20_bias_0.8_output.root',   'histname': 'T20_bias_0.8_output_hist.root',   'tag': 'T20_bias_0.8',   'factor': 0.8},
              'Bias_1':     {'filename': 'T20_bias_1_output.root',     'histname': 'T20_bias_1_output_hist.root',     'tag': 'T20_bias_1',     'factor': 1},
              'Bias_2':     {'filename': 'T20_bias_2_output.root',     'histname': 'T20_bias_2_output_hist.root',     'tag': 'T20_bias_2',     'factor': 2},
              'Bias_3':     {'filename': 'T20_bias_3_output.root',     'histname': 'T20_bias_3_output_hist.root',     'tag': 'T20_bias_3',     'factor': 3},
              'Bias_4':     {'filename': 'T20_bias_4_output.root',     'histname': 'T20_bias_4_output_hist.root',     'tag': 'T20_bias_4',     'factor': 4},

              'Bias_0.01_eBrem': {'filename': 'T20_bias_0.01_eBrem_output.root', 'histname': 'T20_bias_0.01_eBrem_output_hist.root', 'tag': 'T20_bias_0.5'},
              'Bias_0.25_eBrem': {'filename': 'T20_bias_0.25_eBrem_output.root', 'histname': 'T20_bias_0.25_eBrem_output_hist.root', 'tag': 'T20_bias_0.5'},
              'Bias_0.5_eBrem':  {'filename': 'T20_bias_0.5_eBrem_output.root',  'histname': 'T20_bias_0.5_eBrem_output_hist.root',  'tag': 'T20_bias_0.5'}}

SCENAR_DICT = {'Primary First Hits all processes':               {'biaslist': ['Bias_0.2', 'Bias_0.5', 'Bias_1', 'Bias_2'], 'histlist': ['h_PFH_S'],
                                                                  'xlabel': 'S [m]', 'ylabel': 'Number of events', 'linFit': True, 'expFit': False, 'logScale': True},
               'Primary First Hits all processes (unweighted)':  {'biaslist': ['Bias_0.2', 'Bias_0.5', 'Bias_1', 'Bias_2'], 'histlist': ['h_PFH_S_unweight'],
                                                                  'xlabel': 'S [m]', 'ylabel': 'Number of events', 'linFit': False, 'expFit': True, 'logScale': True},
               'Primary First Hits each processes':              {'biaslist': ['Bias_1'], 'histlist': ['h_PFH_S_eBrem', 'h_PFH_S_Coulomb', 'h_PFH_S_elecNuc'],
                                                                  'xlabel': 'S [m]', 'ylabel': 'Number of events', 'linFit': True, 'expFit': False, 'logScale': True},
               'Primary First Hits each processes (unweighted)': {'biaslist': ['Bias_1'],
                                                                  'histlist': ['h_PFH_S_eBrem_unweight', 'h_PFH_S_Coulomb_unweight', 'h_PFH_S_elecNuc_unweight'],
                                                                  'xlabel': 'S [m]', 'ylabel': 'Number of events', 'linFit': True, 'expFit': False, 'logScale': False},
               'Primary First Hits eBrem (unweighted)':          {'biaslist': ['Bias_1', 'Bias_0.5_eBrem', 'Bias_0.25_eBrem', 'Bias_0.01_eBrem'],
                                                                  'histlist': ['h_PFH_S_eBrem_unweight'],
                                                                  'xlabel': 'S [m]', 'ylabel': 'Number of events', 'linFit': True, 'expFit': False, 'logScale': True},
               # 'Beam profile in x for 3 samplers':               {'biaslist': ['Bias_0.5'], 'histlist': ['h_StartSampler_x', 'h_MidSampler_x', 'h_EndSampler_x'],
               #                                                    'xlabel': 'X [m]', 'ylabel': 'Number of events', 'linFit': False, 'expFit': False, 'logScale': True},
               # 'Beam profile in xp for 3 samplers':              {'biaslist': ['Bias_0.5'], 'histlist': ['h_StartSampler_xp', 'h_MidSampler_xp', 'h_EndSampler_xp'],
               #                                                    'xlabel': 'XP [rad]', 'ylabel': 'Number of events', 'linFit': False, 'expFit': False, 'logScale': True},
               # 'Beam profile in y for 3 samplers':               {'biaslist': ['Bias_0.5'], 'histlist': ['h_StartSampler_y', 'h_MidSampler_y', 'h_EndSampler_y'],
               #                                                    'xlabel': 'Y [m]', 'ylabel': 'Number of events', 'linFit': False, 'expFit': False, 'logScale': True},
               # 'Beam profile in yp for 3 samplers':              {'biaslist': ['Bias_0.5'], 'histlist': ['h_StartSampler_yp', 'h_MidSampler_yp', 'h_EndSampler_yp'],
               #                                                    'xlabel': 'Y [rad]', 'ylabel': 'Number of events', 'linFit': False, 'expFit': False, 'logScale': True},
               # 'Beam energy profile for 3 samplers':             {'biaslist': ['Bias_0.5'],
               #                                                    'histlist': ['h_StartSampler_energy', 'h_MidSampler_energy', 'h_EndSampler_energy'],
               #                                                    'xlabel': 'E [GeV]', 'ylabel': 'Number of events', 'linFit': False, 'expFit': False, 'logScale': True},
               }


def linear(x, a, b):
    return a * x + b


def exponential(x, a, b):
    return a * _np.exp(-b * x)


def poly2(x, a, b, c):
    X = _np.log(x)
    return a * X * X + b * X + c


def analysis(inputfilename, nbins=50):
    tag = inputfilename.split('.root')[0]

    root_data = _bd.Data.Load('../04_dataLocal/'+inputfilename)
    # e = root_data.GetEvent()
    t = root_data.GetEventTree()

    print("File :", tag, " / Nb of entries = ", t.GetEntries())

    h_PrimaryFirstHit_S_unweight         = _rt.TH1D("h_PFH_S_unweight",         "{} PFH wrt S all processes (unweighted)".format(tag), nbins, 0, 300)
    h_PrimaryFirstHit_S                  = _rt.TH1D("h_PFH_S",                  "{} PFH wrt S all processes".format(tag),              nbins, 0, 300)
    h_PrimaryFirstHit_S_eBrem_unweight   = _rt.TH1D("h_PFH_S_eBrem_unweight",   "{} PFH wrt S eBrem".format(tag),                      nbins, 0, 300)
    h_PrimaryFirstHit_S_eBrem            = _rt.TH1D("h_PFH_S_eBrem",            "{} PFH wrt S eBrem".format(tag),                      nbins, 0, 300)
    h_PrimaryFirstHit_S_Coulomb_unweight = _rt.TH1D("h_PFH_S_Coulomb_unweight", "{} PFH wrt S Coulomb".format(tag),                    nbins, 0, 300)
    h_PrimaryFirstHit_S_Coulomb          = _rt.TH1D("h_PFH_S_Coulomb",          "{} PFH wrt S Coulomb".format(tag),                    nbins, 0, 300)
    h_PrimaryFirstHit_S_elecNuc_unweight = _rt.TH1D("h_PFH_S_elecNuc_unweight", "{} PFH wrt S elecNuc".format(tag),                    nbins, 0, 300)
    h_PrimaryFirstHit_S_elecNuc          = _rt.TH1D("h_PFH_S_elecNuc",          "{} PFH wrt S elecNuc".format(tag),                    nbins, 0, 300)

    h_PrimaryFirstHit_x          = _rt.TH1D("h_PFH_x",               "{} PFH wrt x all processes".format(tag),              nbins, -2e-4, 2e-4)
    h_PrimaryFirstHit_y          = _rt.TH1D("h_PFH_y",               "{} PFH wrt y all processes".format(tag),              nbins, -2e-4, 2e-4)
    h_PrimaryFirstHit_z          = _rt.TH1D("h_PFH_z",               "{} PFH wrt z all processes".format(tag),              nbins, -10, 10)
    h_PrimaryFirstHit_energy     = _rt.TH1D("h_PFH_energy",          "{} PFH wrt energy all processes".format(tag),         nbins, 0, 2e-4)

    h_StartSampler_x             = _rt.TH1D("h_StartSampler_x",      "{} Beam profile in x at start sampler".format(tag),   nbins, -3, 3)
    h_StartSampler_xp            = _rt.TH1D("h_StartSampler_xp",     "{} Beam profile in xp at start sampler".format(tag),  nbins, -1.5, 1.5)
    h_StartSampler_y             = _rt.TH1D("h_StartSampler_y",      "{} Beam profile in y at start sampler".format(tag),   nbins, -3, 3)
    h_StartSampler_yp            = _rt.TH1D("h_StartSampler_yp",     "{} Beam profile in yp at start sampler".format(tag),  nbins, -1.5, 1.5)
    h_StartSampler_energy        = _rt.TH1D("h_StartSampler_energy", "{} Beam energy profile at start sampler".format(tag), nbins, 0, 14)

    h_MidSampler_x               = _rt.TH1D("h_MidSampler_x",        "{} Beam profile in x at mid sampler".format(tag),     nbins, -3, 3)
    h_MidSampler_xp              = _rt.TH1D("h_MidSampler_xp",       "{} Beam profile in xp at mid sampler".format(tag),    nbins, -1.5, 1.5)
    h_MidSampler_y               = _rt.TH1D("h_MidSampler_y",        "{} Beam profile in y at mid sampler".format(tag),     nbins, -3, 3)
    h_MidSampler_yp              = _rt.TH1D("h_MidSampler_yp",       "{} Beam profile in yp at mid sampler".format(tag),    nbins, -1.5, 1.5)
    h_MidSampler_energy          = _rt.TH1D("h_MidSampler_energy",   "{} Beam energy profile at mid sampler".format(tag),   nbins, 0, 14)

    h_EndSampler_x               = _rt.TH1D("h_EndSampler_x",        "{} Beam profile in x at end sampler".format(tag),     nbins, -3, 3)
    h_EndSampler_xp              = _rt.TH1D("h_EndSampler_xp",       "{} Beam profile in xp at end sampler".format(tag),    nbins, -1.5, 1.5)
    h_EndSampler_y               = _rt.TH1D("h_EndSampler_y",        "{} Beam profile in y at end sampler".format(tag),     nbins, -3, 3)
    h_EndSampler_yp              = _rt.TH1D("h_EndSampler_yp",       "{} Beam profile in yp at end sampler".format(tag),    nbins, -1.5, 1.5)
    h_EndSampler_energy          = _rt.TH1D("h_EndSampler_energy",   "{} Beam energy profile at end sampler".format(tag),   nbins, 0, 14)

    h_EndSampler_x_electrons      = _rt.TH1D("h_EndSampler_x_electrons",      "{} Beam profile in x at end sampler for electrons".format(tag),      nbins, -3, 3)
    h_EndSampler_xp_electrons     = _rt.TH1D("h_EndSampler_xp_electrons",     "{} Beam profile in xp at end sampler for electrons".format(tag),     nbins, -1.5, 1.5)
    h_EndSampler_y_electrons      = _rt.TH1D("h_EndSampler_y_electrons",      "{} Beam profile in y at end sampler for electrons".format(tag),      nbins, -3, 3)
    h_EndSampler_yp_electrons     = _rt.TH1D("h_EndSampler_yp_electrons",     "{} Beam profile in yp at end sampler for electrons".format(tag),     nbins, -1.5, 1.5)
    h_EndSampler_energy_electrons = _rt.TH1D("h_EndSampler_energy_electrons", "{} Beam energy profile at end sampler for electrons".format(tag),    nbins, 0, 14)

    h_EndSampler_x_positrons      = _rt.TH1D("h_EndSampler_x_positrons",      "{} Beam profile in x at end sampler for positrons".format(tag),      nbins, -3, 3)
    h_EndSampler_xp_positrons     = _rt.TH1D("h_EndSampler_xp_positrons",     "{} Beam profile in xp at end sampler for positrons".format(tag),     nbins, -1.5, 1.5)
    h_EndSampler_y_positrons      = _rt.TH1D("h_EndSampler_y_positrons",      "{} Beam profile in y at end sampler for positrons".format(tag),      nbins, -3, 3)
    h_EndSampler_yp_positrons     = _rt.TH1D("h_EndSampler_yp_positrons",     "{} Beam profile in yp at end sampler for positrons".format(tag),     nbins, -1.5, 1.5)
    h_EndSampler_energy_positrons = _rt.TH1D("h_EndSampler_energy_positrons", "{} Beam energy profile at end sampler for positrons".format(tag),    nbins, 0, 14)

    h_EndSampler_x_photons        = _rt.TH1D("h_EndSampler_x_photons",        "{} Beam profile in x at end sampler for photons".format(tag),        nbins, -3, 3)
    h_EndSampler_xp_photons       = _rt.TH1D("h_EndSampler_xp_photons",       "{} Beam profile in xp at end sampler for photons".format(tag),       nbins, -1.5, 1.5)
    h_EndSampler_y_photons        = _rt.TH1D("h_EndSampler_y_photons",        "{} Beam profile in y at end sampler for photons".format(tag),        nbins, -3, 3)
    h_EndSampler_yp_photons       = _rt.TH1D("h_EndSampler_yp_photons",       "{} Beam profile in yp at end sampler for photons".format(tag),       nbins, -1.5, 1.5)
    h_EndSampler_energy_photons   = _rt.TH1D("h_EndSampler_energy_photons",   "{} Beam energy profile at end sampler for photons".format(tag),      nbins, 0, 14)

    for i, evt in enumerate(t):
        h_PrimaryFirstHit_S_unweight.Fill(evt.PrimaryFirstHit.S[0])
        h_PrimaryFirstHit_S.Fill(evt.PrimaryFirstHit.S[0], evt.PrimaryFirstHit.weight[0])
        if evt.PrimaryFirstHit.postStepProcessSubType[0] == 1:
            h_PrimaryFirstHit_S_Coulomb_unweight.Fill(evt.PrimaryFirstHit.S[0])
            h_PrimaryFirstHit_S_Coulomb.Fill(evt.PrimaryFirstHit.S[0], evt.PrimaryFirstHit.weight[0])
        if evt.PrimaryFirstHit.postStepProcessSubType[0] == 3:
            h_PrimaryFirstHit_S_eBrem_unweight.Fill(evt.PrimaryFirstHit.S[0])
            h_PrimaryFirstHit_S_eBrem.Fill(evt.PrimaryFirstHit.S[0], evt.PrimaryFirstHit.weight[0])
        if evt.PrimaryFirstHit.postStepProcessSubType[0] == 121:
            h_PrimaryFirstHit_S_elecNuc_unweight.Fill(evt.PrimaryFirstHit.S[0])
            h_PrimaryFirstHit_S_elecNuc.Fill(evt.PrimaryFirstHit.S[0], evt.PrimaryFirstHit.weight[0])

        h_PrimaryFirstHit_x.Fill(evt.PrimaryFirstHit.x[0], evt.PrimaryFirstHit.weight[0])
        h_PrimaryFirstHit_y.Fill(evt.PrimaryFirstHit.y[0], evt.PrimaryFirstHit.weight[0])
        h_PrimaryFirstHit_z.Fill(evt.PrimaryFirstHit.z[0], evt.PrimaryFirstHit.weight[0])
        h_PrimaryFirstHit_energy.Fill(evt.PrimaryFirstHit.energy[0], evt.PrimaryFirstHit.weight[0])

        if len(evt.QFH41CL.weight) != 0:
            h_StartSampler_x.Fill(evt.QFH41CL.x[0], evt.QFH41CL.weight[0])
            h_StartSampler_xp.Fill(evt.QFH41CL.xp[0], evt.QFH41CL.weight[0])
            h_StartSampler_y.Fill(evt.QFH41CL.y[0], evt.QFH41CL.weight[0])
            h_StartSampler_yp.Fill(evt.QFH41CL.yp[0], evt.QFH41CL.weight[0])
            h_StartSampler_energy.Fill(evt.QFH41CL.energy[0], evt.QFH41CL.weight[0])

        if len(evt.KL2TL.weight) != 0:
            h_MidSampler_x.Fill(evt.KL2TL.x[0], evt.KL2TL.weight[0])
            h_MidSampler_xp.Fill(evt.KL2TL.xp[0], evt.KL2TL.weight[0])
            h_MidSampler_y.Fill(evt.KL2TL.y[0], evt.KL2TL.weight[0])
            h_MidSampler_yp.Fill(evt.KL2TL.yp[0], evt.KL2TL.weight[0])
            h_MidSampler_energy.Fill(evt.KL2TL.energy[0], evt.KL2TL.weight[0])

        if len(evt.D70899L.weight) != 0:
            h_EndSampler_x.Fill(evt.D70899L.x[0], evt.D70899L.weight[0])
            h_EndSampler_xp.Fill(evt.D70899L.xp[0], evt.D70899L.weight[0])
            h_EndSampler_y.Fill(evt.D70899L.y[0], evt.D70899L.weight[0])
            h_EndSampler_yp.Fill(evt.D70899L.yp[0], evt.D70899L.weight[0])
            h_EndSampler_energy.Fill(evt.D70899L.energy[0], evt.D70899L.weight[0])

            if evt.QFH41CL.partID[0] == 11:
                h_EndSampler_x_electrons.Fill(evt.D70899L.x[0], evt.D70899L.weight[0])
                h_EndSampler_xp_electrons.Fill(evt.D70899L.xp[0], evt.D70899L.weight[0])
                h_EndSampler_y_electrons.Fill(evt.D70899L.y[0], evt.D70899L.weight[0])
                h_EndSampler_yp_electrons.Fill(evt.D70899L.yp[0], evt.D70899L.weight[0])
                h_EndSampler_energy_electrons.Fill(evt.D70899L.energy[0], evt.D70899L.weight[0])
            if evt.QFH41CL.partID[0] == -11:
                h_EndSampler_x_positrons.Fill(evt.D70899L.x[0], evt.D70899L.weight[0])
                h_EndSampler_xp_positrons.Fill(evt.D70899L.xp[0], evt.D70899L.weight[0])
                h_EndSampler_y_positrons.Fill(evt.D70899L.y[0], evt.D70899L.weight[0])
                h_EndSampler_yp_positrons.Fill(evt.D70899L.yp[0], evt.D70899L.weight[0])
                h_EndSampler_energy_positrons.Fill(evt.D70899L.energy[0], evt.D70899L.weight[0])
            if evt.QFH41CL.partID[0] == 22:
                h_EndSampler_x_photons.Fill(evt.D70899L.x[0], evt.D70899L.weight[0])
                h_EndSampler_xp_photons.Fill(evt.D70899L.xp[0], evt.D70899L.weight[0])
                h_EndSampler_y_photons.Fill(evt.D70899L.y[0], evt.D70899L.weight[0])
                h_EndSampler_yp_photons.Fill(evt.D70899L.yp[0], evt.D70899L.weight[0])
                h_EndSampler_energy_photons.Fill(evt.D70899L.energy[0], evt.D70899L.weight[0])

    h_PrimaryFirstHit_S_unweight.Scale(ELECTRONS_PER_BUNCH/t.GetEntries())
    h_PrimaryFirstHit_S.Scale(ELECTRONS_PER_BUNCH/t.GetEntries())
    h_PrimaryFirstHit_S_eBrem_unweight.Scale(ELECTRONS_PER_BUNCH / t.GetEntries())
    h_PrimaryFirstHit_S_eBrem.Scale(ELECTRONS_PER_BUNCH/t.GetEntries())
    h_PrimaryFirstHit_S_Coulomb_unweight.Scale(ELECTRONS_PER_BUNCH / t.GetEntries())
    h_PrimaryFirstHit_S_Coulomb.Scale(ELECTRONS_PER_BUNCH/t.GetEntries())
    h_PrimaryFirstHit_S_elecNuc_unweight.Scale(ELECTRONS_PER_BUNCH / t.GetEntries())
    h_PrimaryFirstHit_S_elecNuc.Scale(ELECTRONS_PER_BUNCH/t.GetEntries())

    h_PrimaryFirstHit_x.Scale(ELECTRONS_PER_BUNCH / t.GetEntries())
    h_PrimaryFirstHit_y.Scale(ELECTRONS_PER_BUNCH / t.GetEntries())
    h_PrimaryFirstHit_z.Scale(ELECTRONS_PER_BUNCH / t.GetEntries())
    h_PrimaryFirstHit_energy.Scale(ELECTRONS_PER_BUNCH / t.GetEntries())

    h_StartSampler_x.Scale(ELECTRONS_PER_BUNCH / t.GetEntries())
    h_StartSampler_xp.Scale(ELECTRONS_PER_BUNCH / t.GetEntries())
    h_StartSampler_y.Scale(ELECTRONS_PER_BUNCH / t.GetEntries())
    h_StartSampler_yp.Scale(ELECTRONS_PER_BUNCH / t.GetEntries())
    h_StartSampler_energy.Scale(ELECTRONS_PER_BUNCH / t.GetEntries())

    h_MidSampler_x.Scale(ELECTRONS_PER_BUNCH / t.GetEntries())
    h_MidSampler_xp.Scale(ELECTRONS_PER_BUNCH / t.GetEntries())
    h_MidSampler_y.Scale(ELECTRONS_PER_BUNCH / t.GetEntries())
    h_MidSampler_yp.Scale(ELECTRONS_PER_BUNCH / t.GetEntries())
    h_MidSampler_energy.Scale(ELECTRONS_PER_BUNCH / t.GetEntries())

    h_EndSampler_x.Scale(ELECTRONS_PER_BUNCH / t.GetEntries())
    h_EndSampler_xp.Scale(ELECTRONS_PER_BUNCH / t.GetEntries())
    h_EndSampler_y.Scale(ELECTRONS_PER_BUNCH / t.GetEntries())
    h_EndSampler_yp.Scale(ELECTRONS_PER_BUNCH / t.GetEntries())
    h_EndSampler_energy.Scale(ELECTRONS_PER_BUNCH / t.GetEntries())

    h_EndSampler_x_electrons.Scale(ELECTRONS_PER_BUNCH / t.GetEntries())
    h_EndSampler_xp_electrons.Scale(ELECTRONS_PER_BUNCH / t.GetEntries())
    h_EndSampler_y_electrons.Scale(ELECTRONS_PER_BUNCH / t.GetEntries())
    h_EndSampler_yp_electrons.Scale(ELECTRONS_PER_BUNCH / t.GetEntries())
    h_EndSampler_energy_electrons.Scale(ELECTRONS_PER_BUNCH / t.GetEntries())

    h_EndSampler_x_positrons.Scale(ELECTRONS_PER_BUNCH / t.GetEntries())
    h_EndSampler_xp_positrons.Scale(ELECTRONS_PER_BUNCH / t.GetEntries())
    h_EndSampler_y_positrons.Scale(ELECTRONS_PER_BUNCH / t.GetEntries())
    h_EndSampler_yp_positrons.Scale(ELECTRONS_PER_BUNCH / t.GetEntries())
    h_EndSampler_energy_positrons.Scale(ELECTRONS_PER_BUNCH / t.GetEntries())

    h_EndSampler_x_photons.Scale(ELECTRONS_PER_BUNCH / t.GetEntries())
    h_EndSampler_xp_photons.Scale(ELECTRONS_PER_BUNCH / t.GetEntries())
    h_EndSampler_y_photons.Scale(ELECTRONS_PER_BUNCH / t.GetEntries())
    h_EndSampler_yp_photons.Scale(ELECTRONS_PER_BUNCH / t.GetEntries())
    h_EndSampler_energy_photons.Scale(ELECTRONS_PER_BUNCH / t.GetEntries())

    f = _rt.TFile("{}_hist.root".format(tag), "recreate")
    h_PrimaryFirstHit_S_unweight.Write()
    h_PrimaryFirstHit_S.Write()
    h_PrimaryFirstHit_S_eBrem_unweight.Write()
    h_PrimaryFirstHit_S_eBrem.Write()
    h_PrimaryFirstHit_S_Coulomb_unweight.Write()
    h_PrimaryFirstHit_S_Coulomb.Write()
    h_PrimaryFirstHit_S_elecNuc_unweight.Write()
    h_PrimaryFirstHit_S_elecNuc.Write()

    h_PrimaryFirstHit_x.Write()
    h_PrimaryFirstHit_y.Write()
    h_PrimaryFirstHit_z.Write()
    h_PrimaryFirstHit_energy.Write()

    h_StartSampler_x.Write()
    h_StartSampler_xp.Write()
    h_StartSampler_y.Write()
    h_StartSampler_yp.Write()
    h_StartSampler_energy.Write()

    h_MidSampler_x.Write()
    h_MidSampler_xp.Write()
    h_MidSampler_y.Write()
    h_MidSampler_yp.Write()
    h_MidSampler_energy.Write()

    h_EndSampler_x.Write()
    h_EndSampler_xp.Write()
    h_EndSampler_y.Write()
    h_EndSampler_yp.Write()
    h_EndSampler_energy.Write()

    h_EndSampler_x_electrons.Write()
    h_EndSampler_xp_electrons.Write()
    h_EndSampler_y_electrons.Write()
    h_EndSampler_yp_electrons.Write()
    h_EndSampler_energy_electrons.Write()

    h_EndSampler_x_positrons.Write()
    h_EndSampler_xp_positrons.Write()
    h_EndSampler_y_positrons.Write()
    h_EndSampler_yp_positrons.Write()
    h_EndSampler_energy_positrons.Write()

    h_EndSampler_x_photons.Write()
    h_EndSampler_xp_photons.Write()
    h_EndSampler_y_photons.Write()
    h_EndSampler_yp_photons.Write()
    h_EndSampler_energy_photons.Write()

    f.Close()


def plot_var(biaslist, histname):
    X = _np.array([])
    Y = _np.array([])
    for name in biaslist:
        f = _rt.TFile(FILES_DICT[name]['histname'])
        root_hist = f.Get(histname)
        python_hist = _bd.Data.TH1(root_hist)
        X = _np.append(X, FILES_DICT[name]['factor'])
        contents = python_hist.contents
        Y = _np.append(Y, contents[0:-10].std()/contents[0:-10].mean()*100)

    _plt.plot(X, Y, ls='', marker='+', label='Variance data')
    popt, pcov = curve_fit(poly2, X, Y)
    _plt.plot(X, poly2(X, *popt), ls='-', label='Polynomial fit : min for factor = %5.3f' % _np.exp(-popt[1]/(2*popt[0])))

    _plt.xscale("log")
    _plt.legend()


def plot_hist(inputfilename, histname, linFit=False, expFit=False, fitRange=None, color=None, logScale=False):
    f = _rt.TFile(inputfilename)
    root_hist = f.Get(histname)
    python_hist = _bd.Data.TH1(root_hist)

    title = python_hist.hist.GetTitle()
    centres = python_hist.xcentres
    contents = python_hist.contents
    errors = python_hist.errors
    widths = python_hist.xwidths

    #_plt.errorbar(centres, contents, yerr=errors, xerr=widths * 0.5, ls='', marker='+', color=color)#, label=title)
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

    if logScale:
        _plt.yscale("log")
    _plt.legend()


if __name__ == "__main__":

    if False:
        for bias_name in FILES_DICT:
            analysis(FILES_DICT[bias_name]['filename'])
        print("Analysis Completed")

    for scenario in SCENAR_DICT:
        color = 0
        _plt.figure(figsize=(10, 7))
        for bias in SCENAR_DICT[scenario]['biaslist']:
            for hist in SCENAR_DICT[scenario]['histlist']:
                plot_hist(FILES_DICT[bias]['histname'], hist, linFit=SCENAR_DICT[scenario]['linFit'], expFit=SCENAR_DICT[scenario]['expFit'],
                          logScale=SCENAR_DICT[scenario]['logScale'], fitRange=[0, -1], color='C{}'.format(color))
                color += 1

        _plt.xlabel(SCENAR_DICT[scenario]['xlabel'])
        _plt.ylabel(SCENAR_DICT[scenario]['ylabel'])
        _plt.title(scenario)

    if True:
        _plt.figure(figsize=(10, 7))
        biaslist = ['Bias_0.025', 'Bias_0.05', 'Bias_0.075', 'Bias_0.1', 'Bias_0.2', 'Bias_0.3', 'Bias_0.4', 'Bias_0.5', 'Bias_0.6', 'Bias_0.7', 'Bias_0.8',
                    'Bias_1', 'Bias_2', 'Bias_3', 'Bias_4']
        plot_var(biaslist,"h_PFH_S")
        _plt.ylabel("%")
        _plt.xlabel("Biasing factor")
        _plt.title("Variance")

    _plt.show()
