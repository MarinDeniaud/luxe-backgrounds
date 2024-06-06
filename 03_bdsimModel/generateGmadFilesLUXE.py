import numpy as _np
import glob as _gl
import jinja2 as _jj
from datetime import datetime
import subprocess as _sub


main_path       = "~/PhD/PhD-git/luxe-backgrounds/"
gmad_path       = "gmad_files/"
gdml_path       = "../../gdml_files/"
template_path   = "gmad_files/"
beamDump_path   = gmad_path + "beamDump/"
beamGas_path    = gmad_path + "beamGas/"


template_default_tag = "TL-T20-FF-LUXE_template"


Day_dict = {1: {"Name": "Monday",    "Short": "Mon"},
            2: {"Name": "Tuesday",   "Short": "Tue"},
            3: {"Name": "Wednesday", "Short": "Wed"},
            4: {"Name": "Thursday",  "Short": "Thu"},
            5: {"Name": "Friday",    "Short": "Fri"},
            6: {"Name": "Saturday",  "Short": "Sat"},
            7: {"Name": "Sunday",    "Short": "Sun"}}


Month_dict = {1:  {"Name": "January",    "Short": "Jan"},
              2:  {"Name": "February",   "Short": "Feb"},
              3:  {"Name": "March",      "Short": "Mar"},
              4:  {"Name": "April",      "Short": "Apr"},
              5:  {"Name": "May",        "Short": "May"},
              6:  {"Name": "June",       "Short": "Jun"},
              7:  {"Name": "July",       "Short": "Jul"},
              8:  {"Name": "August",     "Short": "Aug"},
              9:  {"Name": "September",  "Short": "Sep"},
              10: {"Name": "October",    "Short": "Oct"},
              11: {"Name": "November",   "Short": "Nov"},
              12: {"Name": "December",   "Short": "Dec"}}


Twiss_dict = {"TL":   {'ALPHX': 0.000000,   'ALPHY': 0.000000,   'BETX': 0.200000,     'BETY': 3.000000,
                       'DX': 0.0,           'DY': 0.0,           'DPX': 0.0,           'DPY': 0.0},
              "T20":  {'ALPHX': 0.704704,   'ALPHY': -2.157083,  'BETX': 11.376521,    'BETY': 42.889192,
                       'DX': 8.975787e-17,  'DY': -0.025708,     'DPX': 5.688787e-18,  'DPY': -0.001629},
              "FF":   {'ALPHX': -1.224729,  'ALPHY': -2.153503,  'BETX': 11.655919,    'BETY': 50.099013,
                       'DX': -2.066821e-11, 'DY': -4.393263e-11, 'DPX': -2.244639e-12, 'DPY': -6.532824e-12},
              "LUXE": {'ALPHX': 3.256904,   'ALPHY': 3.256919,   'BETX': 26.767280,    'BETY': 26.764963,
                       'DX': -2.939631e-11, 'DY': -8.382907e-11, 'DPX': 3.983481e-12,  'DPY': 4.543499e-12}}


def getFirstLatticeSection(lattice):
    for element in lattice.split('-'):
        if element in Twiss_dict.keys():
            return element


def getDateTime():
    now = datetime.now()
    dayname = Day_dict[now.isoweekday()]["Short"]
    monthname = Month_dict[now.month]["Short"]
    date = [dayname, "{:0>2}".format(now.day), monthname, now.year]
    time = ["{:0>2}".format(now.hour), "{:0>2}".format(now.minute), "{:0>2}".format(now.second)]
    return date, time


def GenerateMainGmadFile(lattice, tag, date, time, pathtooutput=gmad_path, templatetag=template_default_tag, templatefolder=template_path,
                         sampleAll=True, samplerList=None):
    if sampleAll:
        samplers = "sample, all;"
    else:
        samplers = ""
        for sampler in samplerList:
            samplers += "sample, range = {};\n".format(sampler)
    paramdict = dict(tag=lattice+'_'+tag, dayname=date[0], day=date[1], month=date[2], year=date[3], hour=time[0], minute=time[1], second=time[2], samplers=samplers)
    GenerateOneGmadFile(pathtooutput + lattice + '_' + tag + ".gmad", templatetag + ".gmad", templatefolder, paramdict)


def GenerateBeamGmadFile(lattice, tag, date, time, pathtooutput=gmad_path, templatetag=template_default_tag, templatefolder=template_path,
                         distrType="gausstwiss", energy=14, particle="e-",
                         X0=0, Xp0=0, Y0=0, Yp0=0, emitx=3.58e-11, emity=3.58e-11,
                         sigmaX=10e-6, sigmaXp=10e-6, sigmaY=10e-6, sigmaYp=10e-6, sigmaT=100e-15, sigmaE=1e-6):
    firstSection = getFirstLatticeSection(lattice)
    paramdict = dict(dayname=date[0], day=date[1], month=date[2], year=date[3], hour=time[0], minute=time[1], second=time[2],
                     X0=X0, Xp0=Xp0, Y0=Y0, Yp0=Yp0, distrType=distrType, energy=energy, particle=particle,
                     alfx=Twiss_dict[firstSection]['ALPHX'], alfy=Twiss_dict[firstSection]['ALPHY'],
                     betx=Twiss_dict[firstSection]['BETX'], bety=Twiss_dict[firstSection]['BETY'],
                     dispx=Twiss_dict[firstSection]['DX'], dispxp=Twiss_dict[firstSection]['DPX'],
                     dispy=Twiss_dict[firstSection]['DY'], dispyp=Twiss_dict[firstSection]['DPY'],
                     emitx=emitx, emity=emity, sigmaX=sigmaX, sigmaXp=sigmaXp, sigmaY=sigmaY, sigmaYp=sigmaYp, sigmaT=sigmaT, sigmaE=sigmaE)
    GenerateOneGmadFile(pathtooutput + lattice + '_' + tag + "_beam.gmad", templatetag + "_beam.gmad", templatefolder, paramdict)


def GenerateComponentsGmadFile(lattice, tag, date, time, pathtooutput=gmad_path, templatetag=template_default_tag, templatefolder=template_path,
                               gdmlfolder=gdml_path, ICSC='ICSChamber.gdml', M001='FirstMagnet.gdml', DUM1='FirstDump.gdml',
                               IPCH='IPChamberWithAssembly.gdml', M002='SecondMagnet.gdml', EXT1='FirstExtraction.gdml',
                               DET1='FirstDetectors.gdml', GAMC='GammaChamber.gdml', DUM2='SecondDump.gdml', M003='ThirdMagnet.gdml',
                               EXT2='SecondExtraction.gdml', DUM3='ThirdDump.gdml', WALL='FinalWall.gdml',
                               FIELD1=0.0, FIELD2=0.0, FIELD3=0.0, biasMatDump1='biasDump', biasMatDump2='biasDump', biasMatDump3='biasDump'):
    paramdict = dict(dayname=date[0], day=date[1], month=date[2], year=date[3], hour=time[0], minute=time[1], second=time[2],
                     ICSC=gdmlfolder+ICSC, M001=gdmlfolder+M001, DUM1=gdmlfolder+DUM1, IPCH=gdmlfolder+IPCH, M002=gdmlfolder+M002,
                     EXT1=gdmlfolder+EXT1, DET1=gdmlfolder+DET1, GAMC=gdmlfolder+GAMC, DUM2=gdmlfolder+DUM2, M003=gdmlfolder+M003,
                     EXT2=gdmlfolder+EXT2, DUM3=gdmlfolder+DUM3, WALL=gdmlfolder+WALL,
                     FIELD1=FIELD1, FIELD2=FIELD2, FIELD3=FIELD3, biasMatDump1=biasMatDump1, biasMatDump2=biasMatDump2, biasMatDump3=biasMatDump3)
    GenerateOneGmadFile(pathtooutput + lattice  + '_' + tag + "_components.gmad", templatetag + "_components.gmad", templatefolder, paramdict)


def GenerateMaterialGmadFile(lattice, tag, date, time, pathtooutput=gmad_path, templatetag=template_default_tag, templatefolder=template_path,
                             T=300, density=1e-12, compList=["G4_H", "G4_C", "G4_O"], fracList=[0.482, 0.221, 0.297]):
    components = ''
    fractions = ''
    if len(compList) == len(fracList):
        for comp, frac in zip(compList, fracList):
            components += ('"' + comp + '", ')
            fractions += (str(frac) + ', ')
    else:
        raise IOError("Number of componemts should be consistant across all parameters")
    paramdict = dict(dayname=date[0], day=date[1], month=date[2], year=date[3], hour=time[0], minute=time[1], second=time[2],
                     T=T, density=density, components=components[:-2], fractions=fractions[:-2])
    GenerateOneGmadFile(pathtooutput + lattice + '_' + tag + "_material.gmad", templatetag + "_material.gmad", templatefolder, paramdict)


def GenerateObjectsGmadFile(lattice, tag, date, time, pathtooutput=gmad_path, templatetag=template_default_tag, templatefolder=template_path,
                            biasNameList=["biasElecBeamGas", "biasDump"], particleList=['e-', 'neutron'],
                            processList=[['eBrem', 'CoulombScat', 'electronNuclear'], ['neutronInelastic']],
                            xsecfactList=[[1.0, 1.0, 1.0], [1.0]], flagList=[[2, 2, 2], [1]]):
    xsecPattern = '{}: xsecBias, particle="{}", proc="{}", xsecfact={}, flag={};\n'
    biases = ''
    if len(biasNameList) == len(particleList) == len(processList) == len(xsecfactList) == len(flagList):
        for name, particle, processes, factors, flags in zip(biasNameList, particleList, processList, xsecfactList, flagList):
            if len(processes) == len(factors) == len(flags):
                procStr = ''
                factStr = ''
                flagStr = ''
                for proc, fact, flag in zip(processes, factors, flags):
                    procStr += proc + ' '
                    factStr += str(fact) + ', '
                    flagStr += str(flag) + ', '
                biases += xsecPattern.format(name, particle, procStr[:-1], '{' + factStr[:-2] + '}', '{' + flagStr[:-2] + '}')
            else:
                raise IOError("Number of processes should be consistant across all parameters")
    else:
        raise IOError("Number of biases should be consistant across all parameters")
    paramdict = dict(dayname=date[0], day=date[1], month=date[2], year=date[3], hour=time[0], minute=time[1], second=time[2], biases=biases)
    GenerateOneGmadFile(pathtooutput + lattice + '_' + tag + "_objects.gmad", templatetag + "_objects.gmad", templatefolder, paramdict)


def GenerateOptionsGmadFile(lattice, tag, date, time, pathtooutput=gmad_path, templatetag=template_default_tag, templatefolder=template_path,
                            physicsList=['em', 'em_extra', 'qgsp_bert', 'decay'],
                            useLENDGammaNuclear=0, useElectroNuclear=1, useMuonNuclear=1, useGammaToMuMu=1, usePositronToMuMu=1,
                            usePositronToHadrons=1, printPhysicsProcesses=0, worldMaterial='vacuum', checkOverlaps=0, storeEloss=0,
                            storeTrajectory=0, storeTrajectorySecondaryParticles=0, trajCutGTZ=18.5, trajectoryConnect=1, storeTrajectoryProcesses=1):
    physics = ''
    for physic in physicsList:
        physics += physic + ' '
    paramdict = dict(dayname=date[0], day=date[1], month=date[2], year=date[3], hour=time[0], minute=time[1], second=time[2], physicsList=physics[:-1],
                     useLENDGammaNuclear=useLENDGammaNuclear, useElectroNuclear=useElectroNuclear, useMuonNuclear=useMuonNuclear,
                     useGammaToMuMu=useGammaToMuMu, usePositronToMuMu=usePositronToMuMu, usePositronToHadrons=usePositronToHadrons,
                     printPhysicsProcesses=printPhysicsProcesses, worldMaterial=worldMaterial, checkOverlaps=checkOverlaps, storeEloss=storeEloss,
                     storeTrajectory=storeTrajectory, storeTrajectorySecondaryParticles=storeTrajectorySecondaryParticles, trajCutGTZ=trajCutGTZ,
                     trajectoryConnect=trajectoryConnect, storeTrajectoryProcesses=storeTrajectoryProcesses)
    GenerateOneGmadFile(pathtooutput + lattice + '_' + tag + "_options.gmad", templatetag + "_options.gmad", templatefolder, paramdict)


def GenerateSequenceGmadFile(lattice, tag, date, time, pathtooutput=gmad_path, templatetag=template_default_tag, templatefolder=template_path):
    line = ''
    for section in lattice.split('-'):
        if section in Twiss_dict.keys():
            line += section + ', '
    paramdict = dict(dayname=date[0], day=date[1], month=date[2], year=date[3], hour=time[0], minute=time[1], second=time[2], line=line[:-2])
    GenerateOneGmadFile(pathtooutput + lattice + '_' + tag + "_sequence.gmad", templatetag + "_sequence.gmad", templatefolder, paramdict)


def GenerateOneGmadFile(gmadfilename, templatefilename, templatefolder=template_path, paramdict=None):
    env = _jj.Environment(loader=_jj.FileSystemLoader(templatefolder))
    template = env.get_template(templatefilename)
    if paramdict is None:
        raise ValueError("No dictionary is provided to set the different parameters in file {}".format(templatefolder+templatefilename))
    f = open(gmadfilename, 'w')
    f.write(template.render(paramdict))
    f.close()


def GenerateSetGmadFiles(lattice="TL-T20-FF-LUXE", tag="default-tag", pathtooutput=gmad_path,
                         templatetag=template_default_tag, templatefolder=template_path,
                         sampleAll=True, samplerList=None, distrType="gausstwiss", energy=14, particle="e-",
                         FIELD1=0.0, FIELD2=0.0, FIELD3=0.0, gdmlfolder=gdml_path,
                         ICSC='ICSChamber.gdml', IPCH='IPChamberWithAssembly.gdml', GAMC='GammaChamber.gdml',
                         biasMatDump1='biasDump', biasMatDump2='biasDump', biasMatDump3='biasDump',
                         T=300, density=1e-12, compList=["G4_H", "G4_C", "G4_O"], fracList=[0.482, 0.221, 0.297],
                         biasNameList=["biasElecBeamGas", "biasDump"], particleList=['e-', 'neutron'],
                         processList=[['eBrem', 'CoulombScat', 'electronNuclear'], ['neutronInelastic']],
                         xsecfactList=[[1.0, 1.0, 1.0], [1.0]], flagList=[[2, 2, 2], [1]],
                         physicsList=['em', 'em_extra', 'qgsp_bert', 'decay'], printPhysicsProcesses=0, checkOverlaps=0,
                         useLENDGammaNuclear=0, useElectroNuclear=1, useMuonNuclear=1, useGammaToMuMu=1, usePositronToMuMu=1,
                         usePositronToHadrons=1, worldMaterial='vacuum', storeEloss=0,
                         storeTrajectory=0, storeTrajectorySecondaryParticles=0, trajCutGTZ=18.5, trajectoryConnect=1, storeTrajectoryProcesses=1):
    date, time = getDateTime()
    GenerateMainGmadFile(lattice, tag, date, time, pathtooutput, templatetag, templatefolder, sampleAll, samplerList)
    GenerateBeamGmadFile(lattice, tag, date, time, pathtooutput, templatetag, templatefolder, distrType, energy, particle)
    GenerateComponentsGmadFile(lattice, tag, date, time, pathtooutput, templatetag, templatefolder, gdmlfolder,
                               ICSC=ICSC, IPCH=IPCH, GAMC=GAMC, FIELD1=FIELD1, FIELD2=FIELD2, FIELD3=FIELD3,
                               biasMatDump1=biasMatDump1, biasMatDump2=biasMatDump2, biasMatDump3=biasMatDump3)
    GenerateMaterialGmadFile(lattice, tag, date, time, pathtooutput, templatetag, templatefolder, T, density, compList, fracList)
    GenerateObjectsGmadFile(lattice, tag, date, time, pathtooutput, templatetag, templatefolder,
                            biasNameList, particleList, processList, xsecfactList, flagList)
    GenerateOptionsGmadFile(lattice, tag, date, time, pathtooutput, templatetag, templatefolder, physicsList,
                            useLENDGammaNuclear, useElectroNuclear, useMuonNuclear, useGammaToMuMu, usePositronToMuMu,
                            usePositronToHadrons, printPhysicsProcesses, worldMaterial, checkOverlaps, storeEloss,
                            storeTrajectory, storeTrajectorySecondaryParticles, trajCutGTZ, trajectoryConnect, storeTrajectoryProcesses)
    GenerateSequenceGmadFile(lattice, tag, date, time, pathtooutput, templatetag, templatefolder)

    return lattice+tag


def GenerateSetGmadFirstDump(lattice="==-===-FF-LUXE", tag="first-dump", pathtooutput=beamDump_path,
                             templatetag=template_default_tag, templatefolder=template_path, **otherargs):
    GenerateSetGmadFiles(lattice=lattice, tag=tag, pathtooutput=pathtooutput, templatetag=templatetag, templatefolder=templatefolder,
                         FIELD1=1.2, FIELD2=0, FIELD3=0, **otherargs)  # 1.2 -1.5 1.4


def GenerateSetGmadSecondDump(lattice="==-===-FF-LUXE", tag="second-dump", pathtooutput=beamDump_path,
                              templatetag=template_default_tag, templatefolder=template_path, **otherargs):
    GenerateSetGmadFiles(lattice=lattice, tag=tag, pathtooutput=pathtooutput, templatetag=templatetag, templatefolder=templatefolder,
                         FIELD1=0, FIELD2=-1.5, FIELD3=0, **otherargs)  # 1.2 -1.5 1.4


def GenerateSetGmadThirdDump(lattice="==-===-FF-LUXE", tag="third-dump", pathtooutput=beamDump_path,
                             templatetag=template_default_tag, templatefolder=template_path, **otherargs):
    GenerateSetGmadFiles(lattice=lattice, tag=tag, pathtooutput=pathtooutput, templatetag=templatetag, templatefolder=templatefolder,
                         FIELD1=0, FIELD2=0, FIELD3=1.4, **otherargs)  # 1.2 -1.5 1.4


def GenerateSetGmadBeamGas(lattice="TL-T20-FF-====", tag="beam-gas-biased", pathtooutput=beamGas_path,
                           templatetag=template_default_tag, templatefolder=template_path,
                           xsecfact=[25000000.0, 1050000000000.0, 370000000000.0], **otherargs):
    GenerateSetGmadFiles(lattice=lattice, tag=tag, pathtooutput=pathtooutput, templatetag=templatetag, templatefolder=templatefolder,
                         biasNameList=["biasElecBeamGas", "biasDump"], particleList=['e-', 'neutron'],
                         processList=[['eBrem', 'CoulombScat', 'electronNuclear'], ['neutronInelastic']],
                         xsecfactList=[xsecfact, [1.0]], flagList=[[2, 2, 2], [1]], **otherargs)


def GenerateMultipleGmadSetsAndList(tag="T20-needle", tagfilename='tagfilelistneedle', valuetoscan='needleOffsetX',
                                    valuelist=['-0.50', '-0.40', '-0.30', '-0.20', '-0.10', '+0.00', '+0.10', '+0.20', '+0.30', '+0.40', '+0.50'],
                                    **otherargs):
    taglist = open(tagfilename, "w")

    for val in valuelist:
        paramdict = {valuetoscan: val}
        for arg in otherargs:
            paramdict[arg] = otherargs[arg]
        taglist.write(GenerateSetGmadFiles(tag=tag, **paramdict) + '\n')
    taglist.close()

    print("File names written in {}".format(tagfilename))