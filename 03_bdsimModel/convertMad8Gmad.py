import pybdsim as _bd
import pymad8 as _m8

def testConvert():
    _bd.Convert.Mad82Gmad('../01_mad8/TWISS_CL_T20', 'T20',
                          beamparamsdict={'EX': 3.58 * 10 ** -11, 'EY': 3.58 * 10 ** -11, 'Esprd': 1 * 10 ** -6,
                                          'particletype': 'e-'},
                          # aperturedict           = {'FD0125' : {"APER_1": 0.2}},
                          # aperlocalpositions      = {1: [(0.0, {"APER_1": 0.4}),
                          #                                (0.1, {"APERTYPE": 'elliptical',"APER_1":0.2,"APER_2":0.3})]},
                          # collimdict             = {},
                          # userdict               = {'FD0125' : {'biasMaterial':'"biasDef1"'}},
                          )

def testApertConvert():
    _bd.Convert.Mad82Gmad('../01_mad8/TWISS_CL_T20', 'T20_test_apert',
                        beamparamsdict={'EX': 3.58 * 10 ** -11, 'EY': 3.58 * 10 ** -11, 'Esprd': 1 * 10 ** -6,'particletype': 'e-'},
                        aperturedict           = {'FD0125' : {"APER_1": 0.02}},
                        aperlocalpositions      = {1: [(0.0, {"APER_1": 0.04}),
                                                        (0.1, {"APERTYPE": 'elliptical',"APER_1":0.02,"APER_2":0.03})]},
                        )

def simpleConvert():
    _bd.Convert.Mad82Gmad('../01_mad8/TWISS_CL_T20', 'T20',
                          beamparamsdict={'EX': 3.58 * 10 ** -11, 'EY': 3.58 * 10 ** -11, 'Esprd': 1 * 10 ** -6,
                                          'particletype': 'e-'},
                          # aperturedict           = {'FD0125' : {"APER_1": 0.2}},
                          # aperlocalpositions      = {1: [(0.0, {"APER_1": 0.4}),
                          #                                (0.1, {"APERTYPE": 'elliptical',"APER_1":0.2,"APER_2":0.3})]},
                          # collimdict             = {},
                          # userdict               = {'FD0125' : {'biasMaterial':'"biasDef1"'}},
                          )

def biasConvert():
    X = _bd.Builder.XSecBias('test', particle='e-', proc='all', xsecfact=10, flag=3)
    _bd.Convert.Mad82Gmad('../01_mad8/TWISS_CL_T20', 'T20_Bias',
                          beamparamsdict={'EX': 3.58 * 10 ** -11, 'EY': 3.58 * 10 ** -11, 'Esprd': 1 * 10 ** -6,
                                          'particletype': 'e-'},
                          # aperturedict           = {'FD0125' : {"APER_1": 0.2}},
                          # aperlocalpositions      = {1: [(0.0, {"APER_1": 0.01}),
                          #                                (0.1, {"APERTYPE": 'elliptical',"APER_1":0.2,"APER_2":0.3})]},
                          # collimdict             = {},
                           userdict               = {'FD0125' : {'biasMaterial':'"test"'}},
                          )