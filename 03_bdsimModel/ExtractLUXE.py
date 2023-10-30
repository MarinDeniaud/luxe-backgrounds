import pyg4ometry as _pyg4
import numpy as _np


def ViewAllLUXE(filename):
    reader = _pyg4.gdml.Reader(filename)

    reg = reader.getRegistry()
    wl = reg.getWorldVolume()

    v = _pyg4.visualisation.VtkViewerNew()
    v.addLogicalVolume(wl)
    v.buildPipelinesAppend()
    v.view()


def GetDump(filename):
    reader = _pyg4.gdml.Reader(filename)
    reg = reader.getRegistry()
    wl = reg.getWorldVolume()
    wl.dumpStructure()


ICSChamberList = ['GammaTargetChamber0x34893c0', 'GammaTargetContainer0x34897d0']
FirstMagnetList = ['Imprint_1_0x34a4f00']
FirstDumpList = ['BeamDumpAssembly0x3498760', 'Shielding0x3499570', 'BeamPipeNextToDump0x3498940', 'BeamSplitContainer0x349bc40',
                 'ShieldingPipe0x349a370', 'BeamPipeMB0x349bf20', 'BeamPipeMD0x349c4a0', 'ShieldAbsorberTop0x349a230']
IPChamberList = ['BeamPipeSIP0x34ade90', 'BeamPipeSIPVac0x34ae330', 'BeamPipeIPM0x34ae670', 'BeamPipeIPMVac0x34ae930',
                 'logicTAUICContainer_pv0x3495c90', 'logicTAUIChamberBPipe_pv0x3495da8', 'logicTAUIChamberBPipeOut_pv0x3495d38',
                 'logicTAUICTop_pv0x3495cc8', 'logicTAUICBottom_pv0x3495d00']
IPChamberList2 = ['BeamPipeSIP0x34ade90', 'BeamPipeIPM0x34ae670', 'Imprint_1_0x348f4a0']
SecondMagnetList = ['Imprint_1_0x34b3ba0']
ExtractionList = ['logicVCContainer_pv0x34a3af0', 'logicVCContainer_pv0x34a3b28',
                  'logicVCMagFieldJoin_pv0x34a3bd0', 'logicVCMagFieldJoin1_pv0x34a3c08']
ExtractionList2 = ['Imprint_1_0x34ad3c0']
ScintillatorList = ['scintArmPhysical0x3520d90']
ScintillatorList2 = ['HICSElectronScintillator0x3523710']
SecondDumpList = ['HICSDumpAssembly0x34b1010', 'HICSDump2T0x34f3790', 'HICSShieldingSide0x34f4c80', 'HICSShieldingMiddle0x34f53d0',
                  'HICSNeutronAbsorberSide0x34f6810', 'HICSNeutronAbsorberTop0x34f6e20', 'HICSNeutronAbsorberBottom0x34f6eb0',
                  'Imprint_1_0x34f7d30', 'BeamPipeGammaT1stC0x34c7740', 'BeamPipeGamma1stC2ndC0x34cd8b0', 'Collimator0x34a6d30']


def All(inputfilename):
    ICSChamber(inputfilename,   'ICSChamber.gmad',      view=False)
    FirstMagnet(inputfilename,  'FirstMagnet.gdml',     view=False)
    FirstDump(inputfilename,    'FirstDump.gdml',       view=False)
    IPChamber(inputfilename,    'IPChamber.gdml',       view=False)
    SecondMagnet(inputfilename, 'SecondMagnet.gdml',    view=False)
    Extraction(inputfilename,   'Extraction.gdml',      view=False)
    Scintillator(inputfilename, 'Scintillator.gdml',    view=False)
    SecondDump(inputfilename,   'SecondDump.gdml',      view=False)


def ICSChamber(inputfilename, outputfilename, view=True, write=True):
    keep(inputfilename, outputfilename, toKeep=ICSChamberList, view=view, write=write)


def FirstMagnet(inputfilename, outputfilename, view=True, write=True):
    keep(inputfilename, outputfilename, toKeep=FirstMagnetList, view=view, write=write)


def FirstDump(inputfilename, outputfilename, view=True, write=True):
    keep(inputfilename, outputfilename, toKeep=FirstDumpList, view=view, write=write)


def IPChamber(inputfilename, outputfilename, view=True, write=True):
    keep(inputfilename, outputfilename, toKeep=IPChamberList, view=view, write=write)


def SecondMagnet(inputfilename, outputfilename, view=True, write=True):
    keep(inputfilename, outputfilename, toKeep=SecondMagnetList, view=view, write=write)


def Extraction(inputfilename, outputfilename, view=True, write=True):
    keep(inputfilename, outputfilename, toKeep=ExtractionList, view=view, write=write)


def Scintillator(inputfilename, outputfilename, view=True, write=True):
    keep(inputfilename, outputfilename, toKeep=ScintillatorList, view=view, write=write)


def SecondDump(inputfilename, outputfilename, view=True, write=True):
    keep(inputfilename, outputfilename, toKeep=SecondDumpList, view=view, write=write)


def keep(inputfilename, outputfilename, toKeep=[], view=True, write=True):
    if not toKeep:
        raise ValueError('No volume given to be kept')
    reader = _pyg4.gdml.Reader(inputfilename)
    reg = reader.getRegistry()

    world_logical = reg.getWorldVolume()

    newDv = []
    for elem in toKeep:
        physical_volume_list = reg.findPhysicalVolumeByName(elem)
        try:
            physical = physical_volume_list[0]
            if world_logical == physical.motherVolume:
                newDv.append(physical)
            else:
                mother_logical = physical.motherVolume
                for mother_physical in world_logical.daughterVolumes:
                    if mother_physical.logicalVolume == mother_logical:
                        position_list = mother_physical.position.eval()
                        physical.position.x.expressionString += ('+' + str(position_list[0]))
                        physical.position.y.expressionString += ('+' + str(position_list[1]))
                        physical.position.z.expressionString += ('+' + str(position_list[2]))
                        newDv.append(physical)
        except IndexError:
            print("Physical volumne {} not found".format(elem))

    world_logical.daughterVolumes = newDv
    world_logical.clipSolid()

    if write:
        w = _pyg4.gdml.Writer()
        w.addDetector(reg)
        w.write(outputfilename)

    if view:
        v = _pyg4.visualisation.VtkViewerNew()
        v.addLogicalVolume(world_logical)
        v.buildPipelinesAppend()
        # v.addAxes()
        v.view()

    return world_logical
