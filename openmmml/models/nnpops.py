import torch
from torch import Tensor
from typing import Optional, Tuple

from NNPOps.BatchedNN import TorchANIBatchedNN
from NNPOps.EnergyShifter import TorchANIEnergyShifter, SpeciesEnergies
from NNPOps.SpeciesConverter import TorchANISpeciesConverter
from NNPOps.SymmetryFunctions import TorchANISymmetryFunctions

class ModOptimizedTorchANI(torch.nn.Module):

    from torchani.models import BuiltinModel # https://github.com/openmm/NNPOps/issues/44

    def __init__(self, model: BuiltinModel, atomicNumbers: Tensor) -> None:

        super().__init__()

        # Optimize the components of an ANI model
        self.species_converter = TorchANISpeciesConverter(model.species_converter, atomicNumbers)
        self.aev_computer = TorchANISymmetryFunctions(model.species_converter, model.aev_computer, atomicNumbers)
        self.neural_networks = TorchANIBatchedNN(model.species_converter, model.neural_networks, atomicNumbers)
        self.energy_shifter = TorchANIEnergyShifter(model.species_converter, model.energy_shifter, atomicNumbers)

    def forward(self, species_coordinates: Tuple[Tensor, Tensor],
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None) -> SpeciesEnergies:

        species_coordinates = self.species_converter(species_coordinates)
        species_aevs = self.aev_computer(species_coordinates, cell=cell, pbc=pbc)
        species_energies = self.neural_networks(species_aevs)
        #species_energies = self.energy_shifter(species_energies) # don't shift energies

        return species_energies
