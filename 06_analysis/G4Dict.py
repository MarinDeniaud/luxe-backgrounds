# =======================================
# BANK OF DICTIONARIES TO DECODE GEANT4
# =======================================


Process_transport_subtype_dict = {91: 'TRANSPORTATION',
                                  92: 'COUPLED_TRANSPORTATION',
                                  401: 'STEP_LIMITER',
                                  402: 'USER_SPECIAL_CUTS',
                                  403: 'NEUTRON_KILLER',
                                  491: 'PARALLEL_WORLD_PROCESS'}


Process_em_subtype_dict = {1: 'fCoulombScattering',
                           2: 'fIonisation',
                           3: 'fBremsstrahlung',
                           4: 'fPairProdByCharged',
                           5: 'fAnnihilation',
                           6: 'fAnnihilationToMuMu',
                           7: 'fAnnihilationToHadrons',
                           8: 'fNuclearStopping',
                           9: 'fElectronGeneralProcess',
                           10: 'fMultipleScattering',
                           11: 'fRayleigh',
                           12: 'fPhotoElectricEffect',
                           13: 'fComptonScattering',
                           14: 'fGammaConversion',
                           15: 'fGammaConversionToMuMu',
                           16: 'fGammaGeneralProcess',
                           17: 'fPositronGeneralProcess',
                           18: 'fAnnihilationToTauTau',
                           21: 'fCerenkov',
                           22: 'fScintillation',
                           23: 'fSynchrotronRadiation',
                           24: 'fTransitionRadiation',
                           25: 'fSurfaceReflection',
                           51: 'fLowEnergyElastic',
                           52: 'fLowEnergyExcitation',
                           53: 'fLowEnergyIonisation',
                           54: 'fLowEnergyVibrationalExcitation',
                           55: 'fLowEnergyAttachment',
                           56: 'fLowEnergyChargeDecrease',
                           57: 'fLowEnergyChargeIncrease',
                           58: 'fLowEnergyElectronSolvation',
                           59: 'fLowEnergyMolecularDecay',
                           60: 'fLowEnergyTransportation',
                           61: 'fLowEnergyBrownianTransportation',
                           62: 'fLowEnergyDoubleIonisation',
                           63: 'fLowEnergyDoubleCap',
                           64: 'fLowEnergyIoniTransfer',
                           65: 'fLowEnergyStaticMol'}


Process_optical_subtype_dict = {31: 'fOpAbsorption',
                                32: 'fOpBoundary',
                                33: 'fOpRayleigh',
                                34: 'fOpWLS',
                                35: 'fOpMieHG',
                                36: 'fOpWLS2'}


Process_hadronic_subtype_dict = {111: 'fHadronElastic',
                                 121: 'fHadronInelastic',
                                 131: 'fCapture',
                                 132: 'fMuAtomicCapture',
                                 141: 'fFission',
                                 151: 'fHadronAtRest',
                                 152: 'fLeptonAtRest',
                                 161: 'fChargeExchange',
                                 210: 'fRadioactiveDecay',
                                 310: 'fEMDissociation',
                                 }


Process_decay_subtype_dict = {201: 'DECAY',
                              210: 'DECAY_Radioactive',
                              211: 'DECAY_Unknown',
                              221: 'DECAY_MuAtom',
                              231: 'DECAY_External'}


Process_ucn_subtype_dict = {41: 'fUCNLoss',
                            42: 'fUCNAbsorption',
                            43: 'fUCNBoundary',
                            44: 'fUCNMultiScattering'}


Process_type_subtype_dict = {0: {'Name': 'fNotDefined',         'Subtype': {0: 'fNotDefined'}},
                             1: {'Name': 'fTransportation',     'Subtype': Process_transport_subtype_dict},
                             2: {'Name': 'fElectromagnetic',    'Subtype': Process_em_subtype_dict},
                             3: {'Name': 'fOptical',            'Subtype': Process_optical_subtype_dict},
                             4: {'Name': 'fHadronic',           'Subtype': Process_hadronic_subtype_dict},
                             5: {'Name': 'fPhotolepton_hadron', 'Subtype': {0: 'fNotDefined'}},
                             6: {'Name': 'fDecay',              'Subtype': Process_decay_subtype_dict},
                             7: {'Name': 'fGeneral',            'Subtype': {0: 'fNotDefined'}},
                             8: {'Name': 'fParameterisation',   'Subtype': {0: 'fNotDefined'}},
                             9: {'Name': 'fUserDefined',        'Subtype': {0: 'fNotDefined'}},
                             10: {'Name': 'fParallel',          'Subtype': {491: 'PARALLEL_WORLD_PROCESS'}},
                             11: {'Name': 'fPhonon',            'Subtype': {0: 'fNotDefined'}},
                             12: {'Name': 'fUCN',               'Subtype': Process_ucn_subtype_dict}
                             }


Particle_type_dict = {0:    {'Name': 'All',                    'Symbol': 'All'},
                      11:   {'Name': 'electron',               'Symbol': '$e^-$'},
                      -11:  {'Name': 'positron',               'Symbol': '$e^+$'},
                      12:   {'Name': 'electron_neutrino',      'Symbol': r'${\nu}_e$'},
                      -12:  {'Name': 'electron_anti_neutrino', 'Symbol': r'$\overline{{\nu}_e}$'},
                      13:   {'Name': 'muon',                   'Symbol': r'${\mu}^-$'},
                      -13:  {'Name': 'anti_muon',              'Symbol': r'${\mu}^+$'},
                      14:   {'Name': 'muon_neutrino',          'Symbol': r'${\nu}_{\mu}$'},
                      -14:  {'Name': 'muon_anti_neutrino',     'Symbol': r'$\overline{{\nu}_{\mu}}$'},
                      22:   {'Name': 'photon',                 'Symbol': r'$\gamma$'},
                      111:  {'Name': 'pion_zero',              'Symbol': r'${\pi}^0$'},
                      211:  {'Name': 'pion_plus',              'Symbol': r'${\pi}^+$'},
                      -211: {'Name': 'pion_minus',             'Symbol': r'${\pi}^-$'},
                      221:  {'Name': 'eta_meson',              'Symbol': r'$\eta$'},
                      130:  {'Name': 'kaon_zero_L',            'Symbol': '$K^0_L$'},
                      310:  {'Name': 'kaon_zero_S',            'Symbol': '$K^0_S$'},
                      311:  {'Name': 'kaon_zero',              'Symbol': '$K^0$'},
                      321:  {'Name': 'kaon_plus',              'Symbol': '$K^+$'},
                      -321: {'Name': 'kaon_minus',             'Symbol': '$K^-$'},
                      2112: {'Name': 'neutron',                'Symbol': 'n'},
                      2212: {'Name': 'proton',                 'Symbol': 'p'},
                      3122: {'Name': 'lambda',                 'Symbol': r'$\Lambda$'},
                      1000010020: {'Name': 'deuterium',        'Symbol': 'D'},
                      }  # {113, 223, 331, 3112, 3212, 3222}
