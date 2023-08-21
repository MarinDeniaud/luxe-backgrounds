import pyg4ometry as _pyg4
import numpy as _np


def MakeNeedle(Radius1=125e-6, Radius2=7e-6, length=2.5e-3):
    reg = _pyg4.geant4.Registry()

    world_size = _pyg4.gdml.Constant("world_size", "0.015", reg)

    wx = _pyg4.gdml.Constant("wx", "world_size", reg)
    wy = _pyg4.gdml.Constant("wy", "world_size", reg)
    wz = _pyg4.gdml.Constant("wz", "world_size", reg)

    world_solid = _pyg4.geant4.solid.Box("world_solid", 2*wx, 2*wy, 2*wz, reg, lunit='m')
    needle_solid = _pyg4.geant4.solid.Cons("needle_solid", 0, Radius1, 0, Radius2, length, 0, _np.pi*2, reg, lunit="m", aunit="rad")

    world_material = _pyg4.geant4.MaterialPredefined("G4_Galactic")
    needle_material = _pyg4.geant4.MaterialPredefined("G4_W")

    world_logical = _pyg4.geant4.LogicalVolume(world_solid, world_material, "world_logical", reg)
    needle_logical = _pyg4.geant4.LogicalVolume(needle_solid, needle_material, "needle_logical", reg)

    needle_physical_1 = _pyg4.geant4.PhysicalVolume([-_np.pi/2, 0, 0], [0, 0, 0], needle_logical, "needle_physical_1", world_logical, reg)
    # needle_physical_2 = _pyg4.geant4.PhysicalVolume([_np.pi/2, 0, 0], [5, 0, 0], needle_logical, "needle_physical_2", world_logical, reg)

    reg.setWorld("world_logical")
    w = _pyg4.gdml.Writer()
    w.addDetector(reg)
    w.write("Needle.gdml")

    vis = _pyg4.visualisation.VtkViewer()
    vis.addLogicalVolume(world_logical)
    vis.addAxes(10)
    vis.view(interactive=True)


def MakePinhole(Radius1=50e-6, Radius2=200e-6, length=1e-3, SideLength=0.005):
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

    vis = _pyg4.visualisation.VtkViewer()
    vis.addLogicalVolume(world_logical)
    vis.addAxes(10)
    vis.view(interactive=True)
