import pyg4ometry as _pyg4
import numpy as _np


def ViewGDML(filename, axis=True):
    reader = _pyg4.gdml.Reader(filename)

    reg = reader.getRegistry()
    world_logical = reg.getWorldVolume()

    v = _pyg4.visualisation.VtkViewerNew()
    v.addLogicalVolume(world_logical)
    v.buildPipelinesAppend()
    if axis:
        v.addAxes(1000)
    v.view()

    return world_logical


def PrintDump(filename):
    reader = _pyg4.gdml.Reader(filename)
    reg = reader.getRegistry()
    wl = reg.getWorldVolume()
    wl.dumpStructure()


ICSChamberList = ['BremsTargetChamber0x349eee0', 'BremsTargetContainer0x349f330', 'BeamPipeTM0x349ff60', 'BeamPipeTMVac0x34a03d0']
FirstMagnetList = ['Imprint_1_0x34a4f00', 'DumpMagnetField0x34a7510']
FirstDumpList = ['BeamDumpAssembly0x3498760', 'Shielding0x3499570',
                 'BeamPipeNextToDump0x3498940', 'BeamPipeNextToDumpVac0x3498ee0',
                 'ShieldingPipe0x349a370', 'ShieldingPipeVac0x349a880',
                 'BeamPipeMB0x349bf20', 'BeamPipeMBVac0x349c1e0',
                 'BeamPipeMD0x349c4a0', 'BeamPipeMDVac0x349c800',
                 'BeamSplitContainer0x349bc40', 'ShieldAbsorberTop0x349a230']
IPChamberList = ['BeamPipeSIP0x34ade90', 'BeamPipeSIPVac0x34ae330',
                 'BeamPipeIPM0x34ae670', 'BeamPipeIPMVac0x34ae930',
                 'logicTAUICContainer_pv0x3495c90', 'logicTAUIChamberBPipeFlange_pv0x3495e18',
                 'logicTAUIChamberBPipe_pv0x3495da8', 'logicTAUIChamberBPipeVac_pv0x3495de0',
                 'logicTAUIChamberBPipeOut_pv0x3495d38', 'logicTAUIChamberBPipeOutVac_pv0x3495d70',
                 'logicTAUICBottom_pv0x3495d00']# , 'logicTAUICTop_pv0x3495cc8']
IPChamberWithLegsList = ['BeamPipeSIP0x34ade90', 'BeamPipeSIPVac0x34ae330',
                         'BeamPipeIPM0x34ae670', 'BeamPipeIPMVac0x34ae930',
                         'Imprint_1_0x348f4a0']
SecondMagnetList = ['Imprint_1_0x34b3ba0', 'IPMagnetField0x34b5950']
ExtractionList = ['logicVCContainer_pv0x34a3af0', 'logicVCContainer_pv0x34a3b28',
                  'logicVCMagFieldJoin_pv0x34a3bd0', 'logicVCMagFieldJoin1_pv0x34a3c08']
ExtractionList2 = ['Imprint_1_0x34ad3c0']
ScintillatorList = ['scintArmPhysical0x3520d90']
ScintillatorList2 = ['HICSElectronScintillator0x3523710']
DetectorsList = ['BeamPipeOPPPDGT0x348a380', 'BeamPipeOPPPDGTVac0x353a7e0',
                 'scintArmPhysical0x3520d90']
GammaChamberList = ['GammaTargetChamber0x34893c0', 'GammaTargetContainer0x34897d0']
SecondDumpList = ['HICSDumpAssembly0x34b1010', 'HICSDump2T0x34f3790', 'HICSShieldingSide0x34f4c80', 'HICSShieldingMiddle0x34f53d0',
                  'HICSNeutronAbsorberSide0x34f6810', 'HICSNeutronAbsorberTop0x34f6e20', 'HICSNeutronAbsorberBottom0x34f6eb0',
                  'Imprint_1_0x34f7d30',
                  'BeamPipeGammaT1stC0x34c7740', 'BeamPipeGammaT1stCVac0x34c7a60',
                  'BeamPipeGamma1stC2ndC0x34cd8b0', 'BeamPipeGamma1stC2ndCVac0x34c7dc0',
                  'Collimator0x34a6d30']


def All(inputfilename):
    ICSChamber(inputfilename,   view=False)
    FirstMagnet(inputfilename,  view=False)
    FirstDump(inputfilename,    view=False)
    IPChamber(inputfilename,    view=False)
    SecondMagnet(inputfilename, view=False)
    Extraction(inputfilename,   view=False)
    Detectors(inputfilename,    view=False)
    GammaChamber(inputfilename, view=False)
    SecondDump(inputfilename,   view=False)


def ICSChamber(inputfilename='lxgeomdump_fluka_3076aff1.gdml', outputfilename='ICSChamber.gdml', centerPhysical='BeamPipeTM0x349ff60', view=True, write=True):
    return keep(inputfilename, outputfilename, toKeep=ICSChamberList, centerPhysical=centerPhysical, view=view, write=write)


def FirstMagnet(inputfilename='lxgeomdump_fluka_3076aff1.gdml', outputfilename='FirstMagnet.gdml', centerPhysical='DMBPipe0x34a1570', view=True, write=True):
    return keep(inputfilename, outputfilename, toKeep=FirstMagnetList, centerPhysical=centerPhysical, view=view, write=write)


def FirstDump(inputfilename='lxgeomdump_fluka_3076aff1.gdml', outputfilename='FirstDump.gdml', centerPhysical='ShieldingPipe0x349a370', view=True, write=True):
    return keep(inputfilename, outputfilename, toKeep=FirstDumpList, centerPhysical=centerPhysical, view=view, write=write)


def IPChamber(inputfilename='lxgeomdump_fluka_3076aff1.gdml', outputfilename='IPChamber.gdml', centerPhysical='BeamPipeSIP0x34ade90', view=True, write=True):
    return keep(inputfilename, outputfilename, toKeep=IPChamberList, centerPhysical=centerPhysical, view=view, write=write)


def SecondMagnet(inputfilename='lxgeomdump_fluka_3076aff1.gdml', outputfilename='SecondMagnet.gdml', view=True, write=True):
    return keep(inputfilename, outputfilename, toKeep=SecondMagnetList, view=view, write=write)


def Extraction(inputfilename='lxgeomdump_fluka_3076aff1.gdml', outputfilename='Extraction.gdml', centerPhysical='logicBeamPipeVCG_pv0x34a3b60', view=True, write=True):
    return keep(inputfilename, outputfilename, toKeep=ExtractionList2, centerPhysical=centerPhysical, view=view, write=write)


def Scintillator(inputfilename='lxgeomdump_fluka_3076aff1.gdml', outputfilename='Scintillator.gdml', view=True, write=True):
    return keep(inputfilename, outputfilename, toKeep=ScintillatorList, view=view, write=write)


def Detectors(inputfilename='lxgeomdump_fluka_3076aff1.gdml', outputfilename='Detectors.gdml', centerPhysical='BeamPipeOPPPDGT0x348a380', view=True, write=True):
    return keep(inputfilename, outputfilename, toKeep=DetectorsList, centerPhysical=centerPhysical, view=view, write=write)


def GammaChamber(inputfilename='lxgeomdump_fluka_3076aff1.gdml', outputfilename='GammaChamber.gdml', centerPhysical='GAbsorber0x3489440', view=True, write=True):
    return keep(inputfilename, outputfilename, toKeep=GammaChamberList, centerPhysical=centerPhysical, view=view, write=write)


def SecondDump(inputfilename='lxgeomdump_fluka_3076aff1.gdml', outputfilename='SecondDump.gdml', centerPhysical='BeamPipeGammaT1stC0x34c7740', view=True, write=True):
    return keep(inputfilename, outputfilename, toKeep=SecondDumpList, centerPhysical=centerPhysical, view=view, write=write)


def keep(inputfilename, outputfilename, toKeep=[], centerPhysical=None, view=True, write=True):
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

    if centerPhysical is not None:
        center(world_logical, world_logical.registry.findPhysicalVolumeByName(centerPhysical)[0])

    if write:
        w = _pyg4.gdml.Writer()
        w.addDetector(reg)
        w.write(outputfilename)

    if view:
        v = _pyg4.visualisation.VtkViewerNew()
        v.addLogicalVolume(world_logical)
        v.buildPipelinesAppend()
        v.addAxes(1000)
        v.view()

    return world_logical


def center(worldLogical, centerPhysical, centerX=True, centerY=True, centerZ=False):
    centerPosition = findGlobalPosition(worldLogical, centerPhysical)
    centerBool = _np.array([centerX, centerY, centerZ])
    centerOffset = centerPosition*centerBool

    wl = worldLogical
    wl.solid.pX = _pyg4.gdml.Constant(wl.solid.name + "_centered_x", wl.solid.pX + 2 * _np.abs(centerOffset[0]), wl.registry, True)
    wl.solid.pY = _pyg4.gdml.Constant(wl.solid.name + "_centered_y", wl.solid.pY + 2 * _np.abs(centerOffset[1]), wl.registry, True)
    wl.solid.pZ = _pyg4.gdml.Constant(wl.solid.name + "_centered_z", wl.solid.pZ + 2 * _np.abs(centerOffset[2]), wl.registry, True)
    wl.mesh.remesh()

    for dv in wl.daughterVolumes:
        dv.position = dv.position - centerOffset


def addGeometryOnReference(mainfilename, addedfilename, outputfilename, referencename, view=True, write=True):
    reader1 = _pyg4.gdml.Reader(mainfilename)
    reader2 = _pyg4.gdml.Reader(addedfilename)

    reg1 = reader1.getRegistry()
    reg2 = reader2.getRegistry()

    world_logical_1 = reg1.getWorldVolume()
    world_logical_2 = reg2.getWorldVolume()

    reference_physical = world_logical_1.registry.findPhysicalVolumeByName(referencename)[0]
    position = findGlobalPosition(world_logical_1, reference_physical)

    physical_volume = _pyg4.geant4.PhysicalVolume([0, 0, 0], position, world_logical_2, addedfilename, reg1.getWorldVolume(), reg1)
    reg1.addVolumeRecursive(physical_volume)
    world_logical_1.mesh.remesh()

    if write:
        w = _pyg4.gdml.Writer()
        w.addDetector(reg1)
        w.write(outputfilename)

    if view:
        v = _pyg4.visualisation.VtkViewerNew()
        v.addLogicalVolume(world_logical_1)
        v.buildPipelinesAppend()
        v.addAxes(1000)
        v.view()

    return world_logical_1


def applyRotation(centerPosition, centerRotation):
    [x, y, z], [px, py, pz] = centerPosition, centerRotation

    if px == _np.pi/2: y, z = z, -y
    if px == -_np.pi/2: y, z = -z, y

    if py == _np.pi/2: x, z = -z, x
    if py == -_np.pi/2: x, z = z, -x

    if pz == _np.pi/2: x, y = y, -x
    if pz == -_np.pi/2: x, y = -y, x

    return _np.array([x, y, z])


def findGlobalPositionRotation(world_logical, physical_volume,
                               offset_position=_np.array([0.0, 0.0, 0.0]), offset_rotation=_np.array([0.0, 0.0, 0.0])):
    global_position = []
    global_rotation = []
    checkIfPhysicalVolumeInRegistry(world_logical, physical_volume.name)

    def recrsive(world_logical, physical_volume, offset_position, offset_rotation):
        for daughter in world_logical.daughterVolumes:
            position = offset_position + daughter.position.eval()
            rotation = offset_rotation + daughter.rotation.eval()
            if daughter.name == physical_volume.name:
                global_position.append(position)
                global_rotation.append(rotation)
                return 0
            recrsive(daughter.logicalVolume, physical_volume, position, rotation)

    recrsive(world_logical, physical_volume, offset_position, offset_rotation)
    return global_position[0], global_rotation[0]


def checkIfPhysicalVolumeInRegistry(world_logical, physical_volume_name):
    if len(world_logical.registry.findPhysicalVolumeByName(physical_volume_name)) == 0:
        raise Exception("Physical volume {} not in registery".format(physical_volume_name))


def checkIfPhysicalVolumeInWorldVolume(world_logical, physical_volume):
    ishere = []
    checkIfPhysicalVolumeInRegistry(world_logical, physical_volume.name)

    def recrsive(world_logical, physical_volume):
        for daughter in world_logical.daughterVolumes:
            if daughter.name == physical_volume.name:
                ishere.append(1)
                return 0
            recrsive(daughter.logicalVolume, physical_volume)

    recrsive(world_logical, physical_volume)
    if len(ishere) == 0:
        raise Exception("Physical volume {} not in world logical {}".format(physical_volume.name, world_logical.name))


def writeOptions(reg, outputfilename):
    w = _pyg4.gdml.Writer()
    w.addDetector(reg)
    w.write(outputfilename)


def viewOptions(world_logical, axis=True, axisLength=1000):
    v = _pyg4.visualisation.VtkViewerNew()
    v.addLogicalVolume(world_logical)
    v.buildPipelinesAppend()
    if axis:
        v.addAxes(axisLength)
    v.view()


# ====================OLD=======================


def OLDkeep(inputfilename, outputfilename, toKeep=[], centerPhysical=None, view=True, axis=True, write=True):
    if not toKeep:
        raise ValueError('No volume given to be kept')
    reader = _pyg4.gdml.Reader(inputfilename)
    reg = reader.getRegistry()

    world_logical = reg.getWorldVolume()

    newDv = []
    for elem in toKeep:
        physical_volume_list = reg.findPhysicalVolumeByName(elem)
        try:
            physical_volume = physical_volume_list[0]
            if world_logical == physical_volume.motherVolume:
                newDv.append(physical_volume)
            else:
                mother_logical = physical_volume.motherVolume
                for mother_physical in world_logical.daughterVolumes:
                    if mother_physical.logicalVolume == mother_logical:
                        position_list = mother_physical.position.eval()
                        rotation_list = mother_physical.rotation.eval()
                        addPositionStr(physical_volume, position_list)
                        addRotationStr(physical_volume, rotation_list)
                        newDv.append(physical_volume)
        except IndexError:
            print("Physical volumne {} not found".format(elem))

    world_logical.daughterVolumes = newDv
    world_logical.clipSolid()

    if centerPhysical is not None:
        center(world_logical, world_logical.registry.findPhysicalVolumeByName(centerPhysical)[0])

    if write: writeOptions(reg, outputfilename)
    if view: viewOptions(world_logical, axis, axisLength=1000)

    return world_logical


def findGlobalPosition(world_logical, physical_volume):
    mother_logical = physical_volume.motherVolume
    if world_logical == mother_logical:
        return physical_volume.position.eval()
    else:
        for mother_physical in world_logical.daughterVolumes:
            if mother_physical.logicalVolume == mother_logical:
                return _np.array(mother_physical.position.eval()) + _np.array(physical_volume.position.eval())
        raise Exception("Physical volume {} is too deep, can't find the global position".format(physical_volume.name))


def findGlobalRotation(world_logical, physical_volume):
    mother_logical = physical_volume.motherVolume
    if world_logical == mother_logical:
        return physical_volume.rotation.eval()
    else:
        for mother_physical in world_logical.daughterVolumes:
            if mother_physical.logicalVolume == mother_logical:
                return _np.array(mother_physical.rotation.eval()) + _np.array(physical_volume.rotation.eval())
        raise Exception("Physical volume {} is too deep, can't find the global rotation".format(physical_volume.name))
