import pyg4ometry as _pyg4
import numpy as _np

import ExtractLUXE


def MakeAssembly(outputfilename='Assembly.gdml', view=True, axis=True, write=True):
    mount_physical = MakeMount(view=False, write=False)
    needle_physical, bar_physical = MakeNeedle(view=False, write=False)

    reg = mount_physical.registry
    world_logical = reg.getWorldVolume()

    new_needle_physical = _pyg4.geant4.PhysicalVolume([-_np.pi/2, 0, 0], [0, -30.85, 0], needle_physical.logicalVolume, "new_needle_physical", world_logical, reg)
    new_bar_physical = _pyg4.geant4.PhysicalVolume([-_np.pi / 2, 0, 0], [0, -28.6, 0], bar_physical.logicalVolume, "new_bar_physical", world_logical, reg)

    reg.addVolumeRecursive(new_needle_physical)
    reg.addVolumeRecursive(new_bar_physical)

    world_logical.clipSolid()
    world_logical.mesh.remesh()
    centerPosition, centerRotation = ExtractLUXE.findGlobalPositionRotation(world_logical, new_needle_physical)
    ExtractLUXE.center(world_logical, centerPosition, centerRotation)
    assembly = world_logical.assemblyVolume()
    assembly.registry.setWorld(assembly.name)

    if write:
        w = _pyg4.gdml.Writer()
        w.addDetector(reg)
        w.write("Assembly.gdml")

    if view:
        vis = _pyg4.visualisation.VtkViewerNew()
        vis.addLogicalVolume(world_logical)
        vis.buildPipelinesAppend()
        vis.addAxes(1000)
        vis.view()


def MakeMount(xsise=16e-3, ysize=55.2e-3, zsize=4e-3, radius=3.25e-3, view=True, write=True):
    reg = _pyg4.geant4.Registry()

    world_solid = _pyg4.geant4.solid.Box("world_solid", 100, 100, 100, reg, lunit='mm')
    box_solid = _pyg4.geant4.solid.Box("box_solid", xsise, ysize, zsize, reg, lunit="m")
    hole_solid = _pyg4.geant4.solid.Cons("hole_solid", 0, radius, 0, radius, zsize, 0, _np.pi * 2, reg, lunit="m", aunit="rad")
    mount_solid = _pyg4.geant4.solid.Subtraction("mount_solid", box_solid, hole_solid, [[0, 0, 0], [0, -19.1, 0]], reg)

    world_logical = _pyg4.geant4.LogicalVolume(world_solid, "G4_Galactic", "world_logical", reg)
    mount_logical = _pyg4.geant4.LogicalVolume(mount_solid, "G4_Al", "mount_logical", reg)

    reg.setWorld("world_logical")
    mount_physical = _pyg4.geant4.PhysicalVolume([0, 0, 0], [0, 0, 0], mount_logical, "mount_physical", world_logical, reg)

    # world_logical.clipSolid()

    if write:
        w = _pyg4.gdml.Writer()
        w.addDetector(reg)
        w.write("AssemblyBox.gdml")

    if view:
        vis = _pyg4.visualisation.VtkViewerNew()
        vis.addLogicalVolume(world_logical)
        vis.buildPipelinesAppend()
        vis.addAxes(1000)
        vis.view()

    return mount_physical


def MakeNeedle(radius1=125e-6, radius2=7e-6, length1=2.5e-3, length2=2e-3, view=True, write=True):
    reg = _pyg4.geant4.Registry()

    world_solid = _pyg4.geant4.solid.Box("world_solid", 100, 100, 100, reg, lunit='mm')
    needle_solid = _pyg4.geant4.solid.Cons("needle_solid", 0, radius1, 0, radius2, length1, 0, _np.pi*2, reg, lunit="m", aunit="rad")
    bar_solid = _pyg4.geant4.solid.Cons("bar_solid", 0, radius1, 0, radius1, length2, 0, _np.pi * 2, reg, lunit="m", aunit="rad")

    world_material = _pyg4.geant4.MaterialPredefined("G4_Galactic")
    needle_material = _pyg4.geant4.MaterialPredefined("G4_W")
    bar_material = _pyg4.geant4.MaterialPredefined("G4_W")

    world_logical = _pyg4.geant4.LogicalVolume(world_solid, world_material, "world_logical", reg)
    needle_logical = _pyg4.geant4.LogicalVolume(needle_solid, needle_material, "needle_logical", reg)
    bar_logical = _pyg4.geant4.LogicalVolume(bar_solid, bar_material, "bar_logical", reg)

    reg.setWorld("world_logical")
    needle_physical = _pyg4.geant4.PhysicalVolume([-_np.pi/2, 0, 0], [0, -2.25, 0], needle_logical, "needle_physical", world_logical, reg)
    bar_physical = _pyg4.geant4.PhysicalVolume([-_np.pi / 2, 0, 0], [0, 0, 0], bar_logical, "bar_physical", world_logical, reg)

    if write:
        w = _pyg4.gdml.Writer()
        w.addDetector(reg)
        w.write("AssemblyNeedle.gdml")

    if view:
        vis = _pyg4.visualisation.VtkViewerNew()
        vis.addLogicalVolume(world_logical)
        vis.buildPipelinesAppend()
        vis.addAxes(1000)
        vis.view()

    return needle_physical, bar_physical


def MakePinhole(Radius1=50e-6, Radius2=200e-6, length=1e-3, SideLength=0.005, view=True, write=True):
    reg = _pyg4.geant4.Registry()

    world_size = _pyg4.gdml.Constant("world_size", str(SideLength), reg)
    large_radius = _pyg4.gdml.Constant("large_radius", str(Radius1), reg)
    small_radius = _pyg4.gdml.Constant("small_radius", str(Radius2), reg)
    width = _pyg4.gdml.Constant("width", str(length), reg)

    world_solid = _pyg4.geant4.solid.Box("world_solid", world_size, world_size, world_size, reg, lunit='m')
    plate_solid = _pyg4.geant4.solid.Box("plate_solid", world_size, world_size, width, reg, lunit="m")
    needle_solid = _pyg4.geant4.solid.Cons("needle_solid", 0, large_radius, 0, small_radius, width, 0, _np.pi * 2, reg, lunit="m", aunit="rad")
    pinhole_solid = _pyg4.geant4.solid.Subtraction("pinhole_solid", plate_solid, needle_solid, [[0, 0, 0], [0, 0, 0]], reg)

    world_material = _pyg4.geant4.MaterialPredefined("G4_Galactic")
    pinhole_material = _pyg4.geant4.MaterialPredefined("G4_Al")

    world_logical = _pyg4.geant4.LogicalVolume(world_solid, world_material, "world_logical", reg)
    pinhole_logical = _pyg4.geant4.LogicalVolume(pinhole_solid, pinhole_material, "pinhole_logical", reg)

    pinhole_physical = _pyg4.geant4.PhysicalVolume([0, 0, 0], [0, 0, 0], pinhole_logical, "pinhole_physical", world_logical, reg)

    if write:
        w = _pyg4.gdml.Writer()
        w.addDetector(reg)
        w.write("Pinhole.gdml")

    if view:
        vis = _pyg4.visualisation.VtkViewerNew()
        vis.addLogicalVolume(world_logical)
        vis.buildPipelinesAppend()
        vis.addAxes(1000)
        vis.view()


def MakeCube(write=True, view=True):
    # registry to store gdml data5
    reg = _pyg4.geant4.Registry()

    # world solid and logical
    ws = _pyg4.geant4.solid.Box("ws", 50, 50, 50, reg)
    wl = _pyg4.geant4.LogicalVolume(ws, "G4_Galactic", "wl", reg)

    reg.setWorld(wl.name)

    # box placed at origin
    b1 = _pyg4.geant4.solid.Box("b1", 10, 10, 10, reg)
    b1_l = _pyg4.geant4.LogicalVolume(b1, "G4_Fe", "b1_l", reg)
    b1_p = _pyg4.geant4.PhysicalVolume([0, 0, 0], [0, 0, 0], b1_l, "b1_p", wl, reg)

    wl.clipSolid()

    if write:
        w = _pyg4.gdml.Writer()
        w.addDetector(reg)
        w.write("Cube.gdml")

    if view:
        vis = _pyg4.visualisation.VtkViewerNew()
        vis.addLogicalVolume(wl)
        vis.buildPipelinesAppend()
        vis.addAxes(1000)
        vis.view()
