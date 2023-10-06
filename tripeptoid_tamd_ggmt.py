'''
Tom Egg
October 6, 2023
My attempt at writing code utilizing OpenMM and the ufedmm library
'''

# Import necessary libraries
import argparse
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
parser.add_argument('--case', dest='case', help='the simulation case', default='trisarcosine')
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
mass = 6.0*unit.dalton*(unit.nanometer/unit.radians)**2
Ks = 12000*unit.kilojoules_per_mole/unit.radians**2
Ts =3000*unit.kelvin
limit = 180*unit.degrees
sigma = 2.846*unit.degrees
frequency=500
height = 2.4*unit.kilojoules_per_mole
bias_factor=8
enforce_gridless=False
deposition_period = 500

# Functions
def dihedral_angle_cvs(prmtop_file, name, *atom_types):
    '''
    Function to define collective variables
    @param prmtop_file : common parameter file
    @param name : variable name
    @param *atom_types : arguments representing atom types
    '''

    # Set for selected angles
    selected_dihedrals = set()

    # Iterate over dihedral angles
    for dihedral in pmd.amber.AmberParm(prmtop_file).dihedrals:

        # Select relevant atoms
        atoms = [getattr(dihedral, f'atom{i+1}') for i in range(4)]     # 4 atoms / dihedral
        types = [atom.type for atom in atoms]

        # Check for matches
        if all(a == b for a, b in zip(types, atom_types)):
            selected_dihedrals.add(tuple(a.idx for a in atoms))
        
        elif all(a == b for a, b in zip(reversed(types), atom_types)):
            selected_dihedrals.add(tuple(a.idx for a in reversed(atoms)))

        # Prepare to add CVs
        n = len(selected_dihedrals)
        collective_variables = []

        # Use OpenMM to add forces
        for i, dihedral in enumerate(selected_dihedrals):

            # Add force
            force = openmm.CustomTorsionForce('theta')
            force.addTorsion(*dihedral, [])
            cv_name = name if n==1 else f{name}{i+1}
            cv = ufedmm.CollectiveVariable(cv_name, force)
            collective_variables.append(cv)
        
        # Return CVs
        return collective_variables

# MAIN
if __name__ == '__main__':
    
    # Seed process
    seed = random.SystemRandom().randint(0, 2**31) if args.seed is None else args.seed

    # Create system
    inpcrd = app.AmberInpcrdFile(f'{args.case}.inpcrd')
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
    phi_angles = dihedral_angle_cvs(f'{args.case}.prmtop', 'phi', 'c', 'n', 'c3', 'c')
    s_phi_1 = ufedmm.DynamicalVariable('s_phi_1', -limit, limit, mass, Ts, phi_angles[0], Ks, sigma=None)
    s_phi_2 = ufedmm.DynamicalVariable('s_phi_2', -limit, limit, mass, Ts, phi_angles[1], Ks, sigma=None)

    psi_angles = dihedral_angle_cvs(f'{args.case}.prmtop', 'psi', 'n', 'c3', 'c', 'n')
    s_psi_1 = ufedmm.DynamicalVariable('s_psi_1', -limit, limit, mass, Ts, psi_angles[0], Ks, sigma=None)
    s_psi_2 = ufedmm.DynamicalVariable('s_psi_2', -limit, limit, mass, Ts, psi_angles[1], Ks, sigma=None)

    omega_angles = dihedral_angle_cvs(f'{args.case}.prmtop', 'omega', 'c3', 'c', 'n', 'c3')
    s_omega_1 = ufedmm.DynamicalVariable('s_omega_1', -limit, limit, mass, Ts, omega_angles[2], Ks, sigma=None)

    # Setup simulation
    ufed = ufedmm.UnifiedFreeEnergyDynamics([s_phi_1,s_phi_2,s_psi_1,s_psi_2,s_omega_1], temp)
    ufedmm.serialize(ufed, 'ufed_object.yml')
    integrator = ufedmm.MiddleMassiveGGMTIntegrator(temp,40*unit.femtoseconds, dt, scheme='VV-Middle')
    integrator.setRandomNumberSeed(seed)
    platform = openmm.Platform.getPlatformByName(args.platform)
    simulation = ufed.simulation(prmtop.topology, system, integrator, platform)
    simulation.context.setPositions(inpcrd.positions)
    simulation.context.setVelocitiesToTemperature(temp, seed)
    output1 = ufedmm.Tee(stdout, 'COLVAR')
    reporter1 = ufedmm.StateDataReporter(output1,10, step=True, multipleTemperatures=False,hillHeights=False ,variables=True,speed=True,separator='\t')
    simulation.reporters.append(reporter1)

    # Run simulation
    simulation.minimizeEnergy()
    simulation.step(nsteps)