import pybdsim as _bd
# import pymad8 as _m8


OUT_DIR = '../04_dataLocal/'


def testConvert():
    tag = 'T20_test'
    _bd.Convert.Mad8Twiss2Gmad('../01_mad8/TWISS_CL_T20', tag,
                          beamparamsdict={'EX': 3.58*10**-11, 'EY': 3.58*10**-11, 'Esprd': 1*10**-6, 'particletype': 'e-'},
                          # aperturedict       = {'FD0125' : {"APER_1": 0.2}},
                          # aperlocalpositions = {1: [(0.0, {"APER_1": 0.01}), (0.1, {"APER_1": 0.05})],
                          #                       2: (0.1, {"APERTYPE": 'elliptical', "APER_1": 0.02, "APER_2": 0.03})})
                          # collimdict         = {},
                          # userdict           = {'FD0125' : {'biasMaterial':'"biasDef1"'}},
                          )
    _bd.Run.Bdsim('{}.gmad'.format(tag), '{}_output'.format(OUT_DIR+tag), ngenerate=1000, batch=True)
    _bd.Run.RebdsimOptics('{}_output.root'.format(OUT_DIR+tag), '{}_optics.root'.format(OUT_DIR+tag))


def testApertConvert():
    tag = 'T20_apert_test'
    _bd.Convert.Mad8Twiss2Gmad('../01_mad8/TWISS_CL_T20', tag,
                          beamparamsdict={'EX': 3.58*10**-11, 'EY': 3.58*10**-11, 'Esprd': 1*10**-6, 'particletype': 'e-'},
                          # aperturedict={'FD0125': {"APER_1": 0.02}},
                          aperlocalpositions={1: [(0.0, {"APER_1": 0.01}), (0.1, {"APER_1": 0.05})],
                                              2: (0.1, {"APERTYPE": 'elliptical', "APER_1": 0.02, "APER_2": 0.03})})

    _bd.Run.Bdsim('{}.gmad'.format(tag), '{}_output'.format(OUT_DIR+tag), ngenerate=1000, batch=True)


def testCollimConvert():
    tag = 'T20_collim_test'
    _bd.Convert.Mad8Twiss2Gmad('../01_mad8/TWISS_CL_T20', tag,
                          beamparamsdict={'EX': 3.58*10**-11, 'EY': 3.5*10**-11, 'Esprd': 1*10**-6, 'particletype': 'e-'},
                          collimatordict={'D0490': {'material': '"graphite"', 'tilt': 0, 'xsize': 0.05, 'ysize': 0.1}})

    _bd.Run.Bdsim('{}.gmad'.format(tag), '{}_output'.format(OUT_DIR+tag), ngenerate=1000, batch=True)


def testBiasConvert():
    tag = 'T20_bias_test'
    bias1 = _bd.Builder.XSecBias('biasDef1', particle='e-', processes='all', xsecfactors='10', flags='3')
    bias2 = _bd.Builder.XSecBias('biasDef2', particle='e-', processes='eBrem eIoni', xsecfactors='9 5', flags='1 2')
    _bd.Convert.Mad8Twiss2Gmad('../01_mad8/TWISS_CL_T20', tag,
                          beamparamsdict={'EX': 3.58*10**-11, 'EY': 3.58*10**-11, 'Esprd': 1*10**-6, 'particletype': 'e-'},
                          biases=[bias1, bias2],
                          userdict={'FD0125': {'biasMaterial': '"biasDef1"'}, 'D0045': {'biasMaterial': '"biasDef2"'}},
                          optionsdict={'physicsList': '"em"'})

    _bd.Run.Bdsim('{}.gmad'.format(tag), '{}_output'.format(OUT_DIR+tag), ngenerate=1000, batch=True)


def testMaterialConvert():
    tag = 'T20_material_test'
    mat1 = _bd.Builder.Material('matDef1', density=1e-12, T=200, components=['"G4_H"', '"G4_O"'], componentsFractions={0.6, 0.4})
    mat2 = _bd.Builder.Material('matDef2', density=1e-10, T=50, components=['"G4_C"', '"G4_O"'], componentsFractions={0.3, 0.7})
    _bd.Convert.Mad8Twiss2Gmad('../01_mad8/TWISS_CL_T20', tag,
                          beamparamsdict={'EX': 3.58*10**-11, 'EY': 3.58*10**-11, 'Esprd': 1*10**-6, 'particletype': 'e-'},
                          materials=[mat1, mat2],
                          userdict={'FD0125': {'vacuumMaterial': '"matDef1"'}, 'D0045': {'vacuumMaterial': '"matDef2"'}})

    _bd.Run.Bdsim('{}.gmad'.format(tag), '{}_output'.format(OUT_DIR+tag), ngenerate=1000, batch=True)


def testRmatConvert():
    tag = 'T20_rmat_test'
    d = {'twiss': '../01_mad8/TWISS_CL_T20', 'rmat': '../01_mad8/RMAT_CL_T20'}
    _bd.Convert.Mad8Twiss2Gmad(d, tag,
                               beamparamsdict={'EX': 3.58 * 10 ** -11, 'EY': 3.58 * 10 ** -11, 'Esprd': 1 * 10 ** -6, 'particletype': 'e-'})
    _bd.Run.Bdsim('{}.gmad'.format(tag), '{}_output'.format(OUT_DIR + tag), ngenerate=1000, batch=True)


def simpleConvert():
    tag = 'T20_simple'
    _bd.Convert.Mad8Twiss2Gmad('../01_mad8/TWISS_CL_T20', tag,
                          beamparamsdict={'EX': 3.58*10**-11, 'EY': 3.58*10**-11, 'Esprd': 1*10**-6, 'particletype': 'e-'})

    _bd.Run.Bdsim('{}.gmad'.format(tag), '{}_output'.format(OUT_DIR+tag), ngenerate=10000, batch=True)
    _bd.Run.RebdsimOptics('{}_output.root'.format(OUT_DIR+tag), '{}_optics.root'.format(OUT_DIR+tag))
    _bd.Compare.Mad8VsBDSIM("../01_mad8/TWISS_CL_T20", "{}_optics.root".format(OUT_DIR+tag), energySpread=1e-6, ex=3.58e-11, ey=3.58e-11)


def simpleDispConvert():
    tag = 'T20_simple_disp'
    _bd.Convert.Mad8Twiss2Gmad('../01_mad8/TWISS_CL_T20', tag,
                          beamparamsdict={'EX': 3.58*10**-11, 'EY': 3.58*10**-11, 'Esprd': 1*10**-4, 'particletype': 'e-'})

    _bd.Run.Bdsim('{}.gmad'.format(tag), '{}_output'.format(OUT_DIR+tag), ngenerate=10000, batch=True)
    _bd.Run.RebdsimOptics('{}_output.root'.format(OUT_DIR+tag), '{}_optics.root'.format(OUT_DIR+tag))
    _bd.Compare.Mad8VsBDSIM("../01_mad8/TWISS_CL_T20", "{}_optics.root".format(OUT_DIR+tag), energySpread=1e-4, ex=3.58e-11, ey=3.58e-11)


def nonBiasConvert():
    tag = 'T20_non_bias'
    vacuum = _bd.Builder.Material('luxeVacuum', density=1e-12, T=300, components=['"G4_H"', '"G4_C"', '"G4_O"'], componentsFractions={0.482, 0.221, 0.297})
    _bd.Convert.Mad8Twiss2Gmad('../01_mad8/TWISS_CL_T20', tag,
                          beamparamsdict={'EX': 3.58*10**-11, 'EY': 3.58*10**-11, 'Esprd': 1*10**-6, 'particletype': 'e-'},
                          materials=vacuum,
                          allelementdict={'vacuumMaterial': '"luxeVacuum"'},
                          optionsdict={'physicsList': '"em em_extra qgsp_bert decay"'})

    _bd.Run.Bdsim('{}.gmad'.format(tag), '{}_output'.format(OUT_DIR+tag), ngenerate=10000, batch=True)


def biasConvert(tag, biasfact=[0.5, 0.5, 0.5], dens=1e-12):
    bias = _bd.Builder.XSecBias('biasElecBeamGas', particle='e-', proc='eBrem CoulombScat electronNuclear',
                                xsecfact=[5e7*biasfact[0], 2.1e12*biasfact[1], 7.4e11*biasfact[2]], flag=[2, 2, 2])
    vacuum = _bd.Builder.Material('luxeVacuum', density=dens, T=300, components=['"G4_H"', '"G4_C"', '"G4_O"'], componentsFractions={0.482, 0.221, 0.297})
    _bd.Convert.Mad8Twiss2Gmad('../01_mad8/TWISS_CL_T20', tag,
                          beamparamsdict={'EX': 3.58*10**-11, 'EY': 3.58*10**-11, 'Esprd': 1*10**-6, 'particletype': 'e-'},
                          biases=bias,
                          materials=vacuum,
                          allelementdict={'biasVacuum': '"biasElecBeamGas"', 'vacuumMaterial': '"luxeVacuum"'},
                          optionsdict={'physicsList': '"em em_extra qgsp_bert decay"',
                                       'useLENDGammaNuclear': '0',
                                       'useElectroNuclear': '1',
                                       'useMuonNuclear': '1',
                                       'useGammaToMuMu': '1',
                                       'usePositronToMuMu': '1',
                                       'usePositronToHadrons': '1',
                                       'printPhysicsProcesses': '1',
                                       'worldMaterial': '"vacuum"',
                                       'checkOverlaps': '1'})

    # _bd.Run.Bdsim('{}.gmad'.format(tag), '{}_10k'.format(OUT_DIR+tag), ngenerate=10000, batch=True)


def run_all_bias():
    biasConvert('T20_bias_0.025', fact=[0.025, 0.025, 0.025])
    biasConvert('T20_bias_0.050', fact=[0.050, 0.050, 0.050])
    biasConvert('T20_bias_0.075', fact=[0.075, 0.075, 0.075])
    biasConvert('T20_bias_0.100', fact=[0.100, 0.100, 0.100])
    biasConvert('T20_bias_0.200', fact=[0.200, 0.200, 0.200])
    biasConvert('T20_bias_0.300', fact=[0.300, 0.300, 0.300])
    biasConvert('T20_bias_0.400', fact=[0.400, 0.400, 0.400])
    biasConvert('T20_bias_0.500', fact=[0.500, 0.500, 0.500])
    biasConvert('T20_bias_0.600', fact=[0.600, 0.600, 0.600])
    biasConvert('T20_bias_0.700', fact=[0.700, 0.700, 0.700])
    biasConvert('T20_bias_0.800', fact=[0.800, 0.800, 0.800])
    biasConvert('T20_bias_1.000', fact=[1.000, 1.000, 1.000])
    biasConvert('T20_bias_2.000', fact=[2.000, 2.000, 2.000])
    biasConvert('T20_bias_3.000', fact=[3.000, 3.000, 3.000])
    biasConvert('T20_bias_4.000', fact=[4.000, 4.000, 4.000])


def run_all_dens():
    biasConvert('T20_dens_1e-12')
