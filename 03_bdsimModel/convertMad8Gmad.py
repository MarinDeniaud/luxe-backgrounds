import pybdsim as _bd
# import pymad8 as _m8


def testConvert():
    tag = 'T20_test'
    _bd.Convert.Mad82Gmad('../01_mad8/TWISS_CL_T20', tag,
                          beamparamsdict={'EX': 3.58*10**-11, 'EY': 3.58*10**-11, 'Esprd': 1*10**-6, 'particletype': 'e-'},
                          # aperturedict       = {'FD0125' : {"APER_1": 0.2}},
                          # aperlocalpositions = {1: [(0.0, {"APER_1": 0.01}), (0.1, {"APER_1": 0.05})],
                          #                       2: (0.1, {"APERTYPE": 'elliptical', "APER_1": 0.02, "APER_2": 0.03})})
                          # collimdict         = {},
                          # userdict           = {'FD0125' : {'biasMaterial':'"biasDef1"'}},
                          )
    _bd.Run.Bdsim('{}.gmad'.format(tag),'{}_output'.format(tag),ngenerate=1000,batch=True)
    _bd.Run.RebdsimOptics('{}_output.root'.format(tag),'{}_optics.root'.format(tag))


def testApertConvert():
    tag = 'T20_apert_test'
    _bd.Convert.Mad82Gmad('../01_mad8/TWISS_CL_T20', tag,
                          beamparamsdict={'EX': 3.58*10**-11, 'EY': 3.58*10**-11, 'Esprd': 1*10**-6, 'particletype': 'e-'},
                          # aperturedict={'FD0125': {"APER_1": 0.02}},
                          aperlocalpositions={1: [(0.0, {"APER_1": 0.01}), (0.1, {"APER_1": 0.05})],
                                              2: (0.1, {"APERTYPE": 'elliptical', "APER_1": 0.02, "APER_2": 0.03})})

    _bd.Run.Bdsim('{}.gmad'.format(tag), '{}_output'.format(tag), ngenerate=1000, batch=True)


def testCollimConvert():
    tag = 'T20_collim_test'
    _bd.Convert.Mad82Gmad('../01_mad8/TWISS_CL_T20', tag,
                          beamparamsdict={'EX': 3.58*10**-11, 'EY': 3.5*10**-11, 'Esprd': 1*10**-6, 'particletype': 'e-'},
                          collimatordict={'D0490': {'material': '"graphite"', 'tilt': 0, 'xsize': 0.05, 'ysize': 0.1}})

    _bd.Run.Bdsim('{}.gmad'.format(tag), '{}_output'.format(tag), ngenerate=1000, batch=True)


def testBiasConvert():
    tag = 'T20_bias_test'
    bias1 = _bd.Builder.XSecBias('biasDef1', particle='e-', processes='all', xsecfactors='10', flags='3')
    bias2 = _bd.Builder.XSecBias('biasDef2', particle='e-', processes='eBrem eIoni', xsecfactors='9 5', flags='1 2')
    _bd.Convert.Mad82Gmad('../01_mad8/TWISS_CL_T20', tag,
                          beamparamsdict={'EX': 3.58*10**-11, 'EY': 3.58*10**-11, 'Esprd': 1*10**-6, 'particletype': 'e-'},
                          biases=[bias1, bias2],
                          userdict={'FD0125': {'biasMaterial': '"biasDef1"'}, 'D0045': {'biasMaterial': '"biasDef2"'}},
                          optionsdict={'physicsList': '"em"'})

    _bd.Run.Bdsim('{}.gmad'.format(tag), '{}_output'.format(tag), ngenerate=1000, batch=True)


def testMaterialConvert():
    tag = 'T20_material_test'
    mat1 = _bd.Builder.Material('matDef1', density=1e-12, T=200, components=['"G4_H"', '"G4_O"'], componentsFractions={0.6, 0.4})
    mat2 = _bd.Builder.Material('matDef2', density=1e-10, T=50, components=['"G4_C"', '"G4_O"'], componentsFractions={0.3, 0.7})
    _bd.Convert.Mad82Gmad('../01_mad8/TWISS_CL_T20', tag,
                          beamparamsdict={'EX': 3.58*10**-11, 'EY': 3.58*10**-11, 'Esprd': 1*10**-6, 'particletype': 'e-'},
                          materials=[mat1, mat2],
                          userdict={'FD0125': {'vacuumMaterial': '"matDef1"'}, 'D0045': {'vacuumMaterial': '"matDef2"'}})

    _bd.Run.Bdsim('{}.gmad'.format(tag), '{}_output'.format(tag), ngenerate=1000, batch=True)


def simpleConvert():
    tag = 'T20_simple'
    _bd.Convert.Mad82Gmad('../01_mad8/TWISS_CL_T20', tag,
                          beamparamsdict={'EX': 3.58*10**-11, 'EY': 3.58*10**-11, 'Esprd': 1*10**-6, 'particletype': 'e-'})

    _bd.Run.Bdsim('{}.gmad'.format(tag), '{}_output'.format(tag), ngenerate=10000, batch=True)
    _bd.Run.RebdsimOptics('output_{}.root'.format(tag), 'optics_{}.root'.format(tag))
    _bd.Compare.Mad8VsBDSIMpandas("../01_mad8/TWISS_CL_T20", "optics_{}.root".format(tag), energySpread=1e-6, ex=3.58e-11, ey=3.58e-11)


def simpleDispConvert():
    tag = 'T20_simple_disp'
    _bd.Convert.Mad82Gmad('../01_mad8/TWISS_CL_T20', tag,
                          beamparamsdict={'EX': 3.58*10**-11, 'EY': 3.58*10**-11, 'Esprd': 1*10**-4, 'particletype': 'e-'})

    _bd.Run.Bdsim('{}.gmad'.format(tag), '{}_output'.format(tag), ngenerate=10000, batch=True)
    _bd.Run.RebdsimOptics('{}_output.root'.format(tag), '{}_optics.root'.format(tag))
    _bd.Compare.Mad8VsBDSIMpandas("../01_mad8/TWISS_CL_T20", "{}_optics.root".format(tag), energySpread=1e-4, ex=3.58e-11, ey=3.58e-11)


def nonBiasConvert():
    tag = 'T20_non_bias'
    vacuum = _bd.Builder.Material('luxeVacuum', density=1e-12, T=300, components=['"G4_H"', '"G4_C"', '"G4_O"'], componentsFractions={0.482, 0.221, 0.297})
    _bd.Convert.Mad82Gmad('../01_mad8/TWISS_CL_T20', tag,
                          beamparamsdict={'EX': 3.58*10**-11, 'EY': 3.58*10**-11, 'Esprd': 1*10**-6, 'particletype': 'e-'},
                          materials=vacuum,
                          allelementdict={'vacuumMaterial': '"luxeVacuum"'},
                          optionsdict={'physicsList': '"em em_extra qgsp_bert decay"'})

    _bd.Run.Bdsim('{}.gmad'.format(tag), '{}_output'.format(tag), ngenerate=10000, batch=True)


def biasConvert():
    tag = 'T20_bias'
    bias = _bd.Builder.XSecBias('biasElecBeamGas', particle='e-', proc='eBrem CoulombScat electronNuclear', xsecfact=[5e7, 2.3e12, 7.3e11], flag=[2, 2, 2])
    vacuum = _bd.Builder.Material('luxeVacuum', density=1e-12, T=300, components=['"G4_H"', '"G4_C"', '"G4_O"'], componentsFractions={0.482, 0.221, 0.297})
    _bd.Convert.Mad82Gmad('../01_mad8/TWISS_CL_T20', tag,
                          beamparamsdict={'EX': 3.58*10**-11, 'EY': 3.58*10**-11, 'Esprd': 1*10**-6, 'particletype': 'e-'},
                          biases=bias,
                          materials=vacuum,
                          allelementdict={'biasVacuum': '"biasElecBeamGas"', 'vacuumMaterial': '"luxeVacuum"'},
                          optionsdict={'physicsList': '"em em_extra qgsp_bert decay"'})

    _bd.Run.Bdsim('{}.gmad'.format(tag), '{}_output'.format(tag), ngenerate=10000, batch=True)


def bias2Convert():
    tag = 'T20_bias_2'
    bias = _bd.Builder.XSecBias('biasElecBeamGas', particle='e-', proc='eBrem CoulombScat electronNuclear', xsecfact=[5e7/2, 2.3e12/2, 7.3e11/2], flag=[2, 2, 2])
    vacuum = _bd.Builder.Material('luxeVacuum', density=1e-12, T=300, components=['"G4_H"', '"G4_C"', '"G4_O"'], componentsFractions={0.482, 0.221, 0.297})
    _bd.Convert.Mad82Gmad('../01_mad8/TWISS_CL_T20', tag,
                          beamparamsdict={'EX': 3.58*10**-11, 'EY': 3.58*10**-11, 'Esprd': 1*10**-6, 'particletype': 'e-'},
                          biases=bias,
                          materials=vacuum,
                          allelementdict={'biasVacuum': '"biasElecBeamGas"', 'vacuumMaterial': '"luxeVacuum"'},
                          optionsdict={'physicsList': '"em em_extra qgsp_bert decay"'})

    _bd.Run.Bdsim('{}.gmad'.format(tag), '{}_output'.format(tag), ngenerate=10000, batch=True)


def bias4Convert():
    tag = 'T20_bias_4'
    bias = _bd.Builder.XSecBias('biasElecBeamGas', particle='e-', proc='eBrem CoulombScat electronNuclear', xsecfact=[5e7/4, 2.3e12/4, 7.3e11/4], flag=[2, 2, 2])
    vacuum = _bd.Builder.Material('luxeVacuum', density=1e-12, T=300, components=['"G4_H"', '"G4_C"', '"G4_O"'], componentsFractions={0.482, 0.221, 0.297})
    _bd.Convert.Mad82Gmad('../01_mad8/TWISS_CL_T20', tag,
                          beamparamsdict={'EX': 3.58*10**-11, 'EY': 3.58*10**-11, 'Esprd': 1*10**-6, 'particletype': 'e-'},
                          biases=bias,
                          materials=vacuum,
                          allelementdict={'biasVacuum': '"biasElecBeamGas"', 'vacuumMaterial': '"luxeVacuum"'},
                          optionsdict={'physicsList': '"em em_extra qgsp_bert decay"'})

    _bd.Run.Bdsim('{}.gmad'.format(tag), '{}_output'.format(tag), ngenerate=10000, batch=True)

def bias8Convert():
    tag = 'T20_bias_8'
    bias = _bd.Builder.XSecBias('biasElecBeamGas', particle='e-', proc='eBrem CoulombScat electronNuclear', xsecfact=[5e7/8, 2.3e12/8, 7.3e11/8], flag=[2, 2, 2])
    vacuum = _bd.Builder.Material('luxeVacuum', density=1e-12, T=300, components=['"G4_H"', '"G4_C"', '"G4_O"'], componentsFractions={0.482, 0.221, 0.297})
    _bd.Convert.Mad82Gmad('../01_mad8/TWISS_CL_T20', tag,
                          beamparamsdict={'EX': 3.58*10**-11, 'EY': 3.58*10**-11, 'Esprd': 1*10**-6, 'particletype': 'e-'},
                          biases=bias,
                          materials=vacuum,
                          allelementdict={'biasVacuum': '"biasElecBeamGas"', 'vacuumMaterial': '"luxeVacuum"'},
                          optionsdict={'physicsList': '"em em_extra qgsp_bert decay"'})

    _bd.Run.Bdsim('{}.gmad'.format(tag), '{}_output'.format(tag), ngenerate=10000, batch=True)


def biasX2Convert():
    tag = 'T20_bias_x2'
    bias = _bd.Builder.XSecBias('biasElecBeamGas', particle='e-', proc='eBrem CoulombScat electronNuclear', xsecfact=[5e7*2, 2.3e12*2, 7.3e11*2], flag=[2, 2, 2])
    vacuum = _bd.Builder.Material('luxeVacuum', density=1e-12, T=300, components=['"G4_H"', '"G4_C"', '"G4_O"'], componentsFractions={0.482, 0.221, 0.297})
    _bd.Convert.Mad82Gmad('../01_mad8/TWISS_CL_T20', tag,
                          beamparamsdict={'EX': 3.58*10**-11, 'EY': 3.58*10**-11, 'Esprd': 1*10**-6, 'particletype': 'e-'},
                          biases=bias,
                          materials=vacuum,
                          allelementdict={'biasVacuum': '"biasElecBeamGas"', 'vacuumMaterial': '"luxeVacuum"'},
                          optionsdict={'physicsList': '"em em_extra qgsp_bert decay"'})

    _bd.Run.Bdsim('{}.gmad'.format(tag), '{}_output'.format(tag), ngenerate=10000, batch=True)


def biasX4Convert():
    tag = 'T20_bias_x4'
    bias = _bd.Builder.XSecBias('biasElecBeamGas', particle='e-', proc='eBrem CoulombScat electronNuclear', xsecfact=[5e7*4, 2.3e12*4, 7.3e11*4], flag=[2, 2, 2])
    vacuum = _bd.Builder.Material('luxeVacuum', density=1e-12, T=300, components=['"G4_H"', '"G4_C"', '"G4_O"'], componentsFractions={0.482, 0.221, 0.297})
    _bd.Convert.Mad82Gmad('../01_mad8/TWISS_CL_T20', tag,
                          beamparamsdict={'EX': 3.58*10**-11, 'EY': 3.58*10**-11, 'Esprd': 1*10**-6, 'particletype': 'e-'},
                          biases=bias,
                          materials=vacuum,
                          allelementdict={'biasVacuum': '"biasElecBeamGas"', 'vacuumMaterial': '"luxeVacuum"'},
                          optionsdict={'physicsList': '"em em_extra qgsp_bert decay"'})

    _bd.Run.Bdsim('{}.gmad'.format(tag), '{}_output'.format(tag), ngenerate=10000, batch=True)