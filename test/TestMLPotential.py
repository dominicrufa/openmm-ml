import openmm as mm
import openmm.app as app
import openmm.unit as unit
from openmmml import MLPotential
import unittest

class TestMLPotential(unittest.TestCase):

    def testCreateMixedSystem(self):
        pdb = app.PDBFile('alanine-dipeptide-explicit.pdb')
        ff = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
        mmSystem = ff.createSystem(pdb.topology, nonbondedMethod=app.PME)
        potential = MLPotential('ani2x')
        mlAtoms = [a.index for a in next(pdb.topology.chains()).atoms()]
        mixedSystem = potential.createMixedSystem(pdb.topology, mmSystem, mlAtoms, interpolate=False, implementation='torchani')
        interpSystem = potential.createMixedSystem(pdb.topology, mmSystem, mlAtoms, interpolate=True, implementation='torchani')
        platform = mm.Platform.getPlatformByName('CUDA')
        mmContext = mm.Context(mmSystem, mm.VerletIntegrator(0.001), platform)
        mixedContext = mm.Context(mixedSystem, mm.VerletIntegrator(0.001), platform)
        interpContext = mm.Context(interpSystem, mm.VerletIntegrator(0.001), platform)
        mmContext.setPositions(pdb.positions)
        mixedContext.setPositions(pdb.positions)
        interpContext.setPositions(pdb.positions)
        mmEnergy = mmContext.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        mixedEnergy = mixedContext.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        interpEnergy1 = interpContext.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        interpContext.setParameter('scale', 0)
        interpEnergy2 = interpContext.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        self.assertAlmostEqual(mixedEnergy, interpEnergy1, delta=1e-5*abs(mixedEnergy))
        self.assertAlmostEqual(mmEnergy, interpEnergy2, delta=1e-5*abs(mmEnergy))

        print(f"{mixedEnergy, mmEnergy}")


if __name__ == '__main__':
    unittest.main()

