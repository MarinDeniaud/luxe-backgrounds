# A toolkit to extract certain parts of a gdml file into smaller gdml using pyg4ometry.
# This script is intended to work with a specific gdml file of the LUXE experiment but
# beside the preconstructed lists and the LUXE related function this script should
# work with any given gdml file

import pyg4ometry as _pyg4
import numpy as _np

# LUXE file this script has been created for
luxe_file = "gdml_files/lxgeomdump_fluka_3076aff1.gdml"

# Lists of physical volumes in the LUXE file corresponding to the different parts of the experiment
ICSChamberList = ['BremsTargetChamber0x349eee0', 'BremsTargetContainer0x349f330', 'BeamPipeTM0x349ff60', 'BeamPipeTMVac0x34a03d0']
FirstMagnetList = ['Imprint_1_0x34a4f00', 'DumpMagnetField0x34a7510']
FirstDumpList = ['BeamDumpAssembly0x3498760', 'Shielding0x3499570', 'BeamPipeNextToDump0x3498940', 'BeamPipeNextToDumpVac0x3498ee0',
                 'ShieldingPipe0x349a370', 'ShieldingPipeVac0x349a880', 'BeamPipeMB0x349bf20', 'BeamPipeMBVac0x349c1e0',
                 'BeamPipeMD0x349c4a0', 'BeamPipeMDVac0x349c800', 'BeamSplitContainer0x349bc40', 'ShieldAbsorberTop0x349a230']
IPChamberList = ['BeamPipeSIP0x34ade90', 'BeamPipeSIPVac0x34ae330', 'BeamPipeIPM0x34ae670', 'BeamPipeIPMVac0x34ae930',
                 'logicTAUICContainer_pv0x3495c90', 'logicTAUIChamberBPipeFlange_pv0x3495e18', 'logicTAUIChamberBPipe_pv0x3495da8', 'logicTAUIChamberBPipeVac_pv0x3495de0',
                 'logicTAUIChamberBPipeOut_pv0x3495d38', 'logicTAUIChamberBPipeOutVac_pv0x3495d70', 'logicTAUICBottom_pv0x3495d00', 'logicTAUICTop_pv0x3495cc8']
IPChamberWithLegsList = ['BeamPipeSIP0x34ade90', 'BeamPipeSIPVac0x34ae330', 'BeamPipeIPM0x34ae670', 'BeamPipeIPMVac0x34ae930', 'Imprint_1_0x348f4a0']
SecondMagnetList = ['Imprint_1_0x34b3ba0', 'IPMagnetField0x34b5950']
FirstExtractList = ['Imprint_1_0x34ad3c0']
FirstDetectorsList = ['BeamPipeOPPPDGT0x348a380', 'BeamPipeOPPPDGTVac0x353a7e0', 'scintArmPhysical0x3520d90']
GammaChamberList = ['GammaTargetChamber0x34893c0', 'GammaTargetContainer0x34897d0']
SecondDumpList = ['HICSDumpAssembly0x34b1010', 'HICSDump2T0x34f3790', 'HICSShieldingSide0x34f4c80', 'HICSShieldingMiddle0x34f53d0',
                  'HICSNeutronAbsorberSide0x34f6810', 'HICSNeutronAbsorberTop0x34f6e20', 'HICSNeutronAbsorberBottom0x34f6eb0',
                  'Imprint_1_0x34f7d30', 'BeamPipeGammaT1stC0x34c7740', 'BeamPipeGammaT1stCVac0x34c7a60',
                  'BeamPipeGamma1stC2ndC0x34cd8b0', 'BeamPipeGamma1stC2ndCVac0x34c7dc0', 'Collimator0x34a6d30']
ThirdMagnetList = ['Imprint_1_0x3532360', 'GMagnetField0x3488670']
SecondExtractList = ['GMagnetWideChamber0x352f740']
SecondDetectorsList = ['BeamPipeVCGProf0x352f7d0', 'BeamProfilerContainer0x34c0e10']  # , 'BeamProfilerBox0x34bb440']  # Missing elements ? There is a cut in the beam pipe
ThirdDumpList = ['ComptShieldingConcrete0x353b3e0', 'ComptShieldingPlate0x353b6f0']
ThirdDetectorsList = ['GammaMonitorContainer0x34dade0', 'GammaBeamDumpAssembly0x34d8bf0']
FinalWallList = ['BSMCalo0x34b7cf0']


def All(inputfilename):
    """A shortcut to extract all the defined sections in LUXE"""
    ICSChamber(inputfilename,       view=False)
    FirstMagnet(inputfilename,      view=False)
    FirstDump(inputfilename,        view=False)
    IPChamber(inputfilename,        view=False)
    SecondMagnet(inputfilename,     view=False)
    FirstExtraction(inputfilename,  view=False)
    FirstDetectors(inputfilename,   view=False)
    GammaChamber(inputfilename,     view=False)
    SecondDump(inputfilename,       view=False)
    ThirdMagnet(inputfilename,      view=False)
    SecondExtraction(inputfilename, view=False)
    SecondDetectors(inputfilename,  view=False)
    ThirdDump(inputfilename,        view=False)
    ThirdDetectors(inputfilename,   view=False)
    FinalWall(inputfilename,        view=False)


# ================= A set of functions to extract each defined sections in LUXE =================

def ICSChamber(inputfilename=luxe_file, outputfilename='gdml_files/ICSChamber.gdml', centerPhysical='BremsTarget0x349f040',
               view=True, axis=True, write=True):
    return keepPhysicalVolume(inputfilename, outputfilename, toKeep=ICSChamberList, centerPhysicalName=centerPhysical,
                              view=view, axis=axis, write=write)


def FirstMagnet(inputfilename=luxe_file, outputfilename='gdml_files/FirstMagnet.gdml', centerPhysical='BeamPipeTM0x349ff60',
                view=True, axis=True, write=True):
    return keepPhysicalVolume(inputfilename, outputfilename, toKeep=FirstMagnetList, centerPhysicalName=centerPhysical,
                              view=view, axis=axis, write=write)


def FirstDump(inputfilename=luxe_file, outputfilename='gdml_files/FirstDump.gdml', centerPhysical='ShieldingPipe0x349a370',
              view=True, axis=True, write=True):
    return keepPhysicalVolume(inputfilename, outputfilename, toKeep=FirstDumpList, centerPhysicalName=centerPhysical,
                              view=view, axis=axis, write=write)


def IPChamber(inputfilename=luxe_file, outputfilename='gdml_files/IPChamber.gdml', centerPhysical='BeamPipeSIP0x34ade90',
              view=True, axis=True, write=True):
    return keepPhysicalVolume(inputfilename, outputfilename, toKeep=IPChamberList, centerPhysicalName=centerPhysical,
                              view=view, axis=axis, write=write)


def IPChamberWithAssembly(inputfilename='gdml_files/IPChamber.gdml', outputfilename='gdml_files/IPChamberWithAssembly_X_{}_e-3_Y_{}_e-3.gdml',
                          view=True, axis=True, write=True, offset_pos=[0, 0, 0], default_offest_pos=[0, 29.85, 0]):
    total_offset = _np.array(offset_pos)+_np.array(default_offest_pos)
    try:
        outputfilename = outputfilename.format(offset_pos[0], offset_pos[1])
    except:
        pass
    return addGeometryOnReference(inputfilename, 'gdml_files/Assembly.gdml', outputfilename, 'IPVolume0x347d9d0',
                                  view=view, axis=axis, write=write, offset_pos=total_offset)


def SecondMagnet(inputfilename=luxe_file, outputfilename='gdml_files/SecondMagnet.gdml', centerPhysical='IPMagnetField0x34b5950',
                 view=True, axis=True, write=True):
    return keepPhysicalVolume(inputfilename, outputfilename, toKeep=SecondMagnetList, centerPhysicalName=centerPhysical,
                              view=view, axis=axis, write=write)


def FirstExtraction(inputfilename=luxe_file, outputfilename='gdml_files/FirstExtraction.gdml', centerPhysical='logicBeamPipeVCG_pv0x34a3b60',
                    view=True, axis=True, write=True):
    return keepPhysicalVolume(inputfilename, outputfilename, toKeep=FirstExtractList, centerPhysicalName=centerPhysical,
                              view=view, axis=axis, write=write)


def FirstDetectors(inputfilename=luxe_file, outputfilename='gdml_files/FirstDetectors.gdml', centerPhysical='BeamPipeOPPPDGT0x348a380',
                   view=True, axis=True, write=True):
    return keepPhysicalVolume(inputfilename, outputfilename, toKeep=FirstDetectorsList, centerPhysicalName=centerPhysical,
                              view=view, axis=axis, write=write)


def GammaChamber(inputfilename=luxe_file, outputfilename='gdml_files/GammaChamber.gdml', centerPhysical='GAbsorber0x3489440',
                 view=True, axis=True, write=True):
    return keepPhysicalVolume(inputfilename, outputfilename, toKeep=GammaChamberList, centerPhysicalName=centerPhysical,
                              view=view, axis=axis, write=write)


def SecondDump(inputfilename=luxe_file, outputfilename='gdml_files/SecondDump.gdml', centerPhysical='BeamPipeGammaT1stC0x34c7740',
               view=True, axis=True, write=True):
    return keepPhysicalVolume(inputfilename, outputfilename, toKeep=SecondDumpList, centerPhysicalName=centerPhysical,
                              view=view, axis=axis, write=write)


def ThirdMagnet(inputfilename=luxe_file, outputfilename='gdml_files/ThirdMagnet.gdml', centerPhysical='GMagnetField0x3488670',
                view=True, axis=True, write=True):
    return keepPhysicalVolume(inputfilename, outputfilename, toKeep=ThirdMagnetList, centerPhysicalName=centerPhysical,
                              view=view, axis=axis, write=write)


def SecondExtraction(inputfilename=luxe_file, outputfilename='gdml_files/SecondExtraction.gdml', view=True, axis=True, write=True):
    return keepPhysicalVolume(inputfilename, outputfilename, toKeep=SecondExtractList, view=view, axis=axis, write=write)


def SecondDetectors(inputfilename=luxe_file, outputfilename='gdml_files/SecondDetectors.gdml', view=True, axis=True, write=True):
    return keepPhysicalVolume(inputfilename, outputfilename, toKeep=SecondDetectorsList, extand=True,
                              positionPlus=[0, 0, 147.5], view=view, axis=axis, write=write)


def ThirdDump(inputfilename=luxe_file, outputfilename='gdml_files/ThirdDump.gdml', centerPhysical='ComptShieldingPlate0x353b6f0',
              view=True, axis=True, write=True):
    return keepPhysicalVolume(inputfilename, outputfilename, toKeep=ThirdDumpList, centerPhysicalName=centerPhysical,
                              view=view, axis=axis, write=write)


def ThirdDetectors(inputfilename=luxe_file, outputfilename='gdml_files/ThirdDetectors.gdml', view=True, axis=True, write=True):
    return keepPhysicalVolume(inputfilename, outputfilename, toKeep=ThirdDetectorsList, extand=True,
                              positionPlus=[0, 0, 1370], positionMinus=[0, 0, 224.99], view=view, axis=axis, write=write)


def FinalWall(inputfilename=luxe_file, outputfilename='gdml_files/FinalWall.gdml', centerPhysical=None, view=True, axis=True, write=True):
    return keepPhysicalVolume(inputfilename, outputfilename, toKeep=FinalWallList, centerPhysicalName=centerPhysical,
                              view=view, axis=axis, write=write)

# ===============================================================================================


def ViewGDML(filename, axis=True, axisLength=1000):
    """Open a 3D view of a gdml file."""
    reader = _pyg4.gdml.Reader(filename)

    reg = reader.getRegistry()
    world_logical = reg.getWorldVolume()

    viewOptions(world_logical, axis=axis, axisLength=axisLength)

    return world_logical


def PrintDump(filename):
    """Print the structure of physical and logical volume in a gdml file."""
    reader = _pyg4.gdml.Reader(filename)
    reg = reader.getRegistry()
    wl = reg.getWorldVolume()
    wl.dumpStructure()


def movePhysicalVolume(inputfilename, outputfilename, rotation, position, physicalNames=[], view=True, axis=True, write=True):
    """Moves a physical volume in a gdml and save it in another gdml file."""
    reader = _pyg4.gdml.Reader(inputfilename)
    reg = reader.getRegistry()
    world_logical = reg.getWorldVolume()

    for elem in physicalNames:
        physical_volume_list = world_logical.registry.findPhysicalVolumeByName(elem)
        try:
            physical_volume = physical_volume_list[0]
            addRotationStr(physical_volume, rotation)
            addPositionStr(physical_volume, position)
        except IndexError:
            print("Physical volumne {} not found".format(elem))

    if write: writeOptions(reg, outputfilename)
    if view: viewOptions(world_logical, axis, axisLength=1000)

    return world_logical


def keepPhysicalVolume(inputfilename, outputfilename, toKeep=[], centerPhysicalName=None,
                       extand=False, positionPlus=[0, 0, 0], positionMinus=[0, 0, 0],
                       view=True, axis=True, write=True):
    """Select specific physical volumes in a gdml file and save them in another gdml file. Option to give a physical volume to center on
    in the transverse plane. Option to extant the world logical volume in all axes."""
    if not toKeep:
        raise ValueError('No volume given to be kept')
    reader = _pyg4.gdml.Reader(inputfilename)
    reg = reader.getRegistry()

    world_logical = reg.getWorldVolume()
    if centerPhysicalName is not None:
        checkIfPhysicalVolumeInRegistry(world_logical, centerPhysicalName)
        centerPhysical = world_logical.registry.findPhysicalVolumeByName(centerPhysicalName)[0]
        checkIfPhysicalVolumeInWorldVolume(world_logical, centerPhysical)
        centerPosition, centerRotation = findGlobalPositionRotation(world_logical, centerPhysical)

    world_logical.daughterVolumes = keepDV(world_logical, toKeep)
    beforePosition, beforeRotation = findGlobalPositionRotation(world_logical, world_logical.daughterVolumes[0])
    world_logical.clipSolid()
    afterPosition, afterRotation = findGlobalPositionRotation(world_logical, world_logical.daughterVolumes[0])

    if centerPhysicalName is not None:
        changedCenterPosition = applyRotation(centerRotation + afterRotation - beforeRotation, centerPosition + afterPosition - beforePosition)
        center(world_logical, changedCenterPosition)

    world_logical.material = world_logical.registry.materialDict['XFELVacuum0x26bd680']

    if extand:
        extandWorld(world_logical, positionPlus, positionMinus)

    if write: writeOptions(reg, outputfilename)
    if view: viewOptions(world_logical, axis, axisLength=1000)

    return world_logical


def keepDV(world_logical, physical_volume_names_list):
    """Select specific physical volumes to keep in the world logical volume."""
    newDv = []
    for elem in physical_volume_names_list:
        physical_volume_list = world_logical.registry.findPhysicalVolumeByName(elem)
        try:
            physical_volume = physical_volume_list[0]
            position_list, rotation_list = findGlobalPositionRotation(world_logical, physical_volume)
            changePositionStr(physical_volume, position_list)
            newDv.append(physical_volume)
        except IndexError:
            print("Physical volumne {} not found".format(elem))
    return newDv


def changePositionStr(physical_volume, position_list):
    """Change the position string of a physical volume."""
    physical_volume.position.x.expressionString = str(position_list[0])
    physical_volume.position.y.expressionString = str(position_list[1])
    physical_volume.position.z.expressionString = str(position_list[2])


def changeRotationStr(physical_volume, rotation_list):
    """Change the rotation string of a physical volume."""
    physical_volume.rotation.x.expressionString = str(rotation_list[0])
    physical_volume.rotation.y.expressionString = str(rotation_list[1])
    physical_volume.rotation.z.expressionString = str(rotation_list[2])


def addPositionStr(physical_volume, position_list):
    """Add to the position string of a physical volume."""
    physical_volume.position.x.expressionString += ('+' + str(position_list[0]))
    physical_volume.position.y.expressionString += ('+' + str(position_list[1]))
    physical_volume.position.z.expressionString += ('+' + str(position_list[2]))


def addRotationStr(physical_volume, rotation_list):
    """Add to the rotation string of a physical volume."""
    physical_volume.rotation.x.expressionString += ('+' + str(rotation_list[0]))
    physical_volume.rotation.y.expressionString += ('+' + str(rotation_list[1]))
    physical_volume.rotation.z.expressionString += ('+' + str(rotation_list[2]))


def center(world_logical, centerPosition, centerX=True, centerY=True, centerZ=False):
    """Center the world logical volume on a given position by changing the borders of the world logical.
    By default, only center on X and Y (transverse plane)."""
    wl = world_logical

    if centerX:
        wl.solid.pX = _pyg4.gdml.Constant(wl.solid.name + "_centered_x", wl.solid.pX + 2 * _np.abs(centerPosition[0]), wl.registry, True)
    if centerY:
        wl.solid.pY = _pyg4.gdml.Constant(wl.solid.name + "_centered_y", wl.solid.pY + 2 * _np.abs(centerPosition[1]), wl.registry, True)
    if centerZ:
        wl.solid.pZ = _pyg4.gdml.Constant(wl.solid.name + "_centered_z", wl.solid.pZ + 2 * _np.abs(centerPosition[2]), wl.registry, True)
    wl.mesh.remesh()

    for dv in wl.daughterVolumes:
        dv.position = dv.position - centerPosition*_np.array([centerX, centerY, centerZ])


def extandWorld(world_logical, positionPlus=[0, 0, 0], positionMinus=[0, 0, 0]):
    """Extand the borders of a world logical volume in all axes."""
    wl = world_logical

    if positionPlus[0] or positionMinus[0]:
        wl.solid.pX = _pyg4.gdml.Constant(wl.solid.name + "_extanted_x", wl.solid.pX + positionPlus[0] + positionMinus[0], wl.registry, True)
    if positionPlus[1] or positionMinus[1]:
        wl.solid.pY = _pyg4.gdml.Constant(wl.solid.name + "_extanted_y", wl.solid.pY + positionPlus[1] + positionMinus[1], wl.registry, True)
    if positionPlus[2] or positionMinus[2]:
        wl.solid.pZ = _pyg4.gdml.Constant(wl.solid.name + "_extanted_z", wl.solid.pZ + positionPlus[2] + positionMinus[2], wl.registry, True)
    wl.mesh.remesh()

    for dv in wl.daughterVolumes:
        dv.position = dv.position + 0.5 * (_np.array(positionMinus) - _np.array(positionPlus))


def addGeometryOnReference(mainfilename, addedfilename, outputfilename, referencename, offset_pos=[0, 0, 0], offset_rot=[0, 0, 0],
                           view=True, axis=True, write=True):
    """Combines a gdml on another gdml. Options to offset the position and rotation. The result is saved in another gdml file."""
    reader1 = _pyg4.gdml.Reader(mainfilename)
    reader2 = _pyg4.gdml.Reader(addedfilename)

    reg1 = reader1.getRegistry()
    reg2 = reader2.getRegistry()

    world_logical_1 = reg1.getWorldVolume()
    world_logical_2 = reg2.getWorldVolume()

    ref_physical = world_logical_1.registry.findPhysicalVolumeByName(referencename)[0]
    physical_volume = _pyg4.geant4.PhysicalVolume(offset_rot, offset_pos, world_logical_2, "added geometry", ref_physical.logicalVolume, reg1)

    reg1.addVolumeRecursive(physical_volume)
    reg1.setWorld(world_logical_1.name)

    if write: writeOptions(reg1, outputfilename)
    if view: viewOptions(world_logical_1, axis, axisLength=1000)

    return world_logical_1


def applyRotation(centerRotation, centerPosition):
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
    """Recusive method to find the global position and rotation of a physical volume in a world logical volume."""
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
    """Check if a physical volume is inside the registry of a given world logical volume."""
    if len(world_logical.registry.findPhysicalVolumeByName(physical_volume_name)) == 0:
        raise Exception("Physical volume {} not in registery".format(physical_volume_name))


def checkIfPhysicalVolumeInWorldVolume(world_logical, physical_volume):
    """Check recursivly if a physical volume is in the structure of a given world logical volume."""
    ishere = []
    checkIfPhysicalVolumeInRegistry(world_logical, physical_volume.name)

    def recursive(world_logical, physical_volume):
        for daughter in world_logical.daughterVolumes:
            if daughter.name == physical_volume.name:
                ishere.append(1)
                return 0
            recursive(daughter.logicalVolume, physical_volume)

    recursive(world_logical, physical_volume)
    if len(ishere) == 0:
        raise Exception("Physical volume {} not in world logical {}".format(physical_volume.name, world_logical.name))


def writeOptions(reg, outputfilename):
    """Writes a registry in a gdml file."""
    w = _pyg4.gdml.Writer()
    w.addDetector(reg)
    w.write(outputfilename)


def viewOptions(world_logical, axis=True, axisLength=1000):
    """3D visualiser for gdml. Options for axes"""
    v = _pyg4.visualisation.VtkViewerNew()
    v.addLogicalVolume(world_logical)
    v.buildPipelinesAppend()
    if axis:
        v.addAxes(axisLength)
    v.view()
