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

# Specify solvent models
gb_models = ['HCT', 'OBC1', 'OBC2', 'GBn', 'GBn2']

# Initialize parser and supply arguments
parser = argparse.ArgumentParser()
parser.add_argument('--case', dest='case', help='the simulation case', default='adp')
parser.add_argument('--implicit-solvent', dest='gb_model', help='an implicit solvent model', choices=gb_models)
parser.add_argument('--salt-molarity', dest='salt_molarity', help='the salt molarity', type=float, default=float_info.min)
parser.add_argument('--seed', dest='seed', help='the RNG seed')
parser.add_argument('--platform', dest='platform', help='the computation platform', default='CPU')
parser.add_argument('--print', dest='print', help='print results?', choices=['yes', 'no'], default='yes')
args = parser.parse_args()

# Constants
temp = 300*unit.kelvin
gamma = 10/unit.picoseconds
dt = 1*unit.femtoseconds
nsteps = 300000000
mass = 30.0*unit.dalton*(unit.nanometer/unit.radians)**2
Ks = 1000*unit.kilojoules_per_mole/unit.radians**2
Ts =1500*unit.kelvin
limit = 180*unit.degrees
sigma = pi/10
frequency=500
height = 2.*unit.kilojoules_per_mole
bias_factor=8
enforce_gridless=False
deposition_period = 200

# MAIN
if __name__ == '__main__':
    
    # Seed process
    seed = random.SystemRandom().randint(0, 2**31) if args.seed is None else args.seed

    # Get data
    pdb = app.PDBFile('adp.pdb')
    pdb.topology.setUnitCellDimensions([2.5*unit.nanometers]*3)
    atoms = [f'{a.name}:{a.residue.name}' for a in pdb.topology.atoms()]

    # Dihedrals
    dihedral_atoms = {
        'phi': map(atoms.index, ['C:ACE', 'N:ALA', 'CA:ALA', 'C:ALA']),
        'psi': map(atoms.index, ['N:ALA', 'CA:ALA', 'C:ALA', 'N:NME']),
    }

    # Create system
    inpcrd = app.AmberInpcrdFile(f'{args.case}.crd')
    prmtop = app.AmberPrmtopFile(f'{args.case}.prmtop')

    system = prmtop.createSystem(
        nonbondedMethod=app.NoCutoff if prmtop.topology.getNumChains() == 1 else app.PME,
        implicitSolvent=app.GBn2,
        implicitSolventSaltConc=args.salt_molarity*unit.moles/unit.liter,
        constraints=None,
        rigidWater=True,
        removeCMMotion=False
    )

    # Phi, psi, and omega angles
    s_phi = ufedmm.DynamicalVariable('s_phi', -limit, limit, mass, Ts, phi, Ks, sigma=sigma)
    s_psi = ufedmm.DynamicalVariable('s_psi', -limit, limit, mass, Ts, psi, Ks, sigma=sigma)


    # Setup simulation
    ufed = ufedmm.UnifiedFreeEnergyDynamics([s_phi, s_psi], temp, height, period)
    ufedmm.serialize(ufed, 'ufed_object.yml')
    integrator = ufedmm.MiddleMassiveGGMTIntegrator(temp,40*unit.femtoseconds, dt, scheme='VV-Middle')
    integrator.setRandomNumberSeed(seed)
    platform = openmm.Platform.getPlatformByName(args.platform)
    simulation = ufed.simulation(prmtop.topology, system, integrator, platform)
    simulation.context.setPositions(inpcrd.positions)
    simulation.context.setVelocitiesToTemperature(temp, seed)
    output1 = ufedmm.Tee(stdout, 'COLVAR_adp')
    reporter1 = ufedmm.StateDataReporter(output1,10, step=True, multipleTemperatures=False,hillHeights=False ,variables=True,speed=True,separator='\t')
    simulation.reporters.append(reporter1)

    # Run simulation
    simulation.minimizeEnergy()
    simulation.step(nsteps)
