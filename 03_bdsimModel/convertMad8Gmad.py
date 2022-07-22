import pybdsim as _bd
# import pymad8 as _m8


def testConvert():
    _bd.Convert.Mad82Gmad('../01_mad8/TWISS_CL_T20', 'T20_test',
                          beamparamsdict={'EX': 3.58*10**-11, 'EY': 3.58*10**-11, 'Esprd': 1*10**-6, 'particletype': 'e-'},
                          # aperturedict       = {'FD0125' : {"APER_1": 0.2}},
                          # aperlocalpositions = {1: [(0.0, {"APER_1": 0.01}), (0.1, {"APER_1": 0.05})],
                          #                       2: (0.1, {"APERTYPE": 'elliptical', "APER_1": 0.02, "APER_2": 0.03})})
                          # collimdict         = {},
                          # userdict           = {'FD0125' : {'biasMaterial':'"biasDef1"'}},
                          )


def testApertConvert():
    _bd.Convert.Mad82Gmad('../01_mad8/TWISS_CL_T20', 'T20_test_apert',
                          beamparamsdict={'EX': 3.58*10**-11, 'EY': 3.58*10**-11, 'Esprd': 1*10**-6, 'particletype': 'e-'},
                          # aperturedict={'FD0125': {"APER_1": 0.02}},
                          aperlocalpositions={1: [(0.0, {"APER_1": 0.01}), (0.1, {"APER_1": 0.05})],
                                              2: (0.1, {"APERTYPE": 'elliptical', "APER_1": 0.02, "APER_2": 0.03})})


def testCollimConvert():
    _bd.Convert.Mad82Gmad('../01_mad8/TWISS_CL_T20', 'T20_test_collim',
                          beamparamsdict={'EX': 3.58*10**-11, 'EY': 3.5*10**-11, 'Esprd': 1*10**-6, 'particletype': 'e-'},
                          collimatordict={'D0490': {'material': '"graphite"', 'tilt': 0, 'xsize': 0.05, 'ysize': 0.1}})


def testBiasConvert():
    bias1 = _bd.Builder.XSecBias('biasDef1', particle='e-', processes='all', xsecfactors='10', flags='3')
    bias2 = _bd.Builder.XSecBias('biasDef2', particle='e-', processes='eBrem eIoni', xsecfactors='9 5', flags='1 2')
    _bd.Convert.Mad82Gmad('../01_mad8/TWISS_CL_T20', 'T20_test_bias',
                          beamparamsdict={'EX': 3.58*10**-11, 'EY': 3.58*10**-11, 'Esprd': 1*10**-6, 'particletype': 'e-'},
                          biases=[bias1, bias2],
                          userdict={'FD0125': {'biasMaterial': '"biasDef1"'}, 'D0045': {'biasMaterial': '"biasDef2"'}},
                          optionsdict={'physicsList': '"em"'})


def testMaterialConvert():
    mat1 = _bd.Builder.Material('matDef1', density=1e-12, T=200, components=['"G4_H"', '"G4_O"'], componentsFractions={0.6, 0.4})
    mat2 = _bd.Builder.Material('matDef2', density=1e-10, T=50, components=['"G4_C"', '"G4_O"'], componentsFractions={0.3, 0.7})
    _bd.Convert.Mad82Gmad('../01_mad8/TWISS_CL_T20', 'T20_test_material',
                          beamparamsdict={'EX': 3.58*10**-11, 'EY': 3.58*10**-11, 'Esprd': 1*10**-6, 'particletype': 'e-'},
                          materials=[mat1, mat2],
                          userdict={'FD0125': {'vacuumMaterial': '"matDef1"'}, 'D0045': {'vacuumMaterial': '"matDef2"'}})


def simpleConvert():
    _bd.Convert.Mad82Gmad('../01_mad8/TWISS_CL_T20', 'T20_simple',
                          beamparamsdict={'EX': 3.58*10**-11, 'EY': 3.58*10**-11, 'Esprd': 1*10**-6, 'particletype': 'e-'})


def simpleDispConvert():
    _bd.Convert.Mad82Gmad('../01_mad8/TWISS_CL_T20', 'T20_simple_disp',
                          beamparamsdict={'EX': 3.58*10**-11, 'EY': 3.58*10**-11, 'Esprd': 1*10**-4, 'particletype': 'e-'})


def biasConvert():
    bias = _bd.Builder.XSecBias('biasElecBeamGas', particle='e-', proc='eBrem CoulombScat electronNuclear',
                          xsecfact=[5e7, 2.3e12, 7.3e11], flag=[2, 2, 2])
    vacuum = _bd.Builder.Material('luxeVacuum', density=1e-12, T=300, components=['"G4_H"', '"G4_C"', '"G4_O"'],
                          componentsFractions={0.482, 0.221, 0.297})
    _bd.Convert.Mad82Gmad('../01_mad8/TWISS_CL_T20', 'T20_bias',
                          beamparamsdict={'EX': 3.58*10**-11, 'EY': 3.58*10**-11, 'Esprd': 1*10**-6, 'particletype': 'e-'},
                          # aperturedict       = {'FD0125' : {"APER_1": 0.2}},
                          # aperlocalpositions = {1: [(0.0, {"APER_1": 0.01}),
                          #                            0.1, {"APERTYPE": 'elliptical',"APER_1":0.2,"APER_2":0.3})]},
                          # collimdict         = {},
                          biases=bias,
                          materials=vacuum,
                          allelementdict={'biasVacuum': '"biasElecBeamGas"', 'vacuumMaterial': '"luxeVacuum"'},
                          optionsdict={'physicsList': '"em em_extra qgsp_bert decay"'})


def nonBiasConvert():
    vacuum = _bd.Builder.Material('luxeVacuum', density=1e-12, T=300, components=['"G4_H"', '"G4_C"', '"G4_O"'],
                          componentsFractions={0.482, 0.221, 0.297})
    _bd.Convert.Mad82Gmad('../01_mad8/TWISS_CL_T20', 'T20_non_bias',
                          beamparamsdict={'EX': 3.58*10**-11, 'EY': 3.58*10**-11, 'Esprd': 1*10**-6, 'particletype': 'e-'},
                          # aperturedict       = {'FD0125' : {"APER_1": 0.2}},
                          # aperlocalpositions = {1: [(0.0, {"APER_1": 0.01}),
                          #                            0.1, {"APERTYPE": 'elliptical',"APER_1":0.2,"APER_2":0.3})]},
                          # collimdict         = {},
                          materials=vacuum,
                          allelementdict={'vacuumMaterial': '"luxeVacuum"'},
                          optionsdict={'physicsList': '"em em_extra qgsp_bert decay"'})