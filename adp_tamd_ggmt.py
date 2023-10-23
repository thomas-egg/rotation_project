'''
Tom Egg
October 6, 2023
My attempt at writing code utilizing OpenMM and the ufedmm library
'''

# Import necessary libraries
import argparse
from numpy import pi
import random
import ufedmm
import parmed as pmd
from copy import deepcopy
from simtk import openmm, unit
from simtk.openmm import app
from sys import stdout, float_info

# Initialize parser and supply arguments
parser = argparse.ArgumentParser()
parser.add_argument('--case', dest='case', help='the simulation case', default='adp')
parser.add_argument('--platform', dest='platform', help='the computation platform', default='CPU')
parser.add_argument('--print', dest='print', help='print results?', choices=['yes', 'no'], default='yes')
args = parser.parse_args()

# Constants
temp = 300*unit.kelvin
gamma = 10/unit.picoseconds
dt = 1*unit.femtoseconds
nsteps = 1000000
mass = 30.0*unit.dalton*(unit.nanometer/unit.radians)**2
Ks = 1000*unit.kilojoules_per_mole/unit.radians**2
Ts =1500*unit.kelvin
limit = pi
sigma = pi/10
frequency=500
height = 2.*unit.kilojoules_per_mole
bias_factor=8
enforce_gridless=False
deposition_period = 200

# MAIN
if __name__ == '__main__':

    # Get data
    pdb = app.PDBFile('adp.pdb')
    pdb.topology.setUnitCellDimensions([2.5*unit.nanometers]*3)
    atoms = [f'{a.name}:{a.residue.name}' for a in pdb.topology.atoms()]

    # Dihedrals
    dihedral_atoms = {
        'phi': map(atoms.index, ['C:ACE', 'N:ALA', 'CA:ALA', 'C:ALA']),
        'psi': map(atoms.index, ['N:ALA', 'CA:ALA', 'C:ALA', 'N:NME']),
    }

    system = app.ForceField('amber03.xml').createSystem(
        pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds,
        removeCMMotion=False
    )
    
    # CVs
    phi = ufedmm.CollectiveVariable('phi', openmm.CustomTorsionForce('theta'))
    phi.force.addTorsion(*dihedral_atoms['phi'], [])    
    psi = ufedmm.CollectiveVariable('psi', openmm.CustomTorsionForce('theta'))
    psi.force.addTorsion(*dihedral_atoms['psi'], [])

    # Phi, psi, and omega angles
    s_phi = ufedmm.DynamicalVariable('s_phi', -limit, limit, mass, Ts, phi, Ks, sigma=sigma)
    s_psi = ufedmm.DynamicalVariable('s_psi', -limit, limit, mass, Ts, psi, Ks, sigma=sigma)


    # Setup simulation
    ufed = ufedmm.UnifiedFreeEnergyDynamics([s_phi, s_psi], temp, height, deposition_period)
    ufedmm.serialize(ufed, 'ufed_object.yml')
    integrator = ufedmm.GeodesicLangevinIntegrator(temp, gamma, 2*unit.femtoseconds)
    platform = openmm.Platform.getPlatformByName(args.platform)
    simulation = ufed.simulation(pdb.topology, system, integrator, platform)
    simulation.context.setPositions(pdb.positions)
    simulation.context.setVelocitiesToTemperature(temp)
    output1 = ufedmm.Tee(stdout, 'COLVAR_adp')
    reporter1 = ufedmm.StateDataReporter(output1, 100, step=True, multipleTemperatures=True, variables=True,speed=True, speed=True, separator='\t')
    simulation.reporters.append(reporter1)

    # Run simulation
    simulation.step(nsteps)
