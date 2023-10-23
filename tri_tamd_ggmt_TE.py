# Import necessary libraries
import argparse
import random
import ufedmm

from simtk import openmm, unit
from openmm import app
from sys import stdout
from sys import stdout, float_info

# Arguments
gb_models = ['HCT', 'OBC1', 'OBC2', 'GBn', 'GBn2']
parser = argparse.ArgumentParser()
parser.add_argument('--implicit-solvent', dest='gb_model', help='an implicit solvent model', choices=gb_models)
parser.add_argument('--salt-molarity', dest='salt_molarity', help='the salt molarity', type=float, default=float_info.min)
parser.add_argument('--seed', dest='seed', help='the RNG seed')
parser.add_argument('--platform', dest='platform', help='the computation platform', default='CPU')
parser.add_argument('--print', dest='print', help='print results?', choices=['yes', 'no'], default='yes')
args = parser.parse_args()

# Simulation Variables
seed = random.SystemRandom().randint(0, 2**31) if args.seed is None else args.seed
temp = 300*unit.kelvin
gamma = 10/unit.picoseconds
dt = 1*unit.femtoseconds
nsteps = 100000000
mass = 6.0*unit.dalton*(unit.nanometer/unit.radians)**2
Ks = 12000*unit.kilojoules_per_mole/unit.radians**2
Ts = 3000*unit.kelvin
limit = 180*unit.degrees
sigma = 2.846*unit.degrees
height = 2.4*unit.kilojoules_per_mole
deposition_period = 500
bias_factor=8
enforce_gridless=False

# Read in files
pdb = app.PDBFile('trisarcosine.pdb')
prmtop = app.AmberPrmtopFile('trisarcosine.prmtop')

# Initialize dihedrals
pdb.topology.setUnitCellDimensions([2.5*unit.nanometers]*3)
atoms = [f'{a.name}:{a.residue.name}:{a.residue.index+1}' for a in pdb.topology.atoms()]
dihedral_atoms = {
        'omega_1' : map(atoms.index, ['C1:ace:1', 'C2:ace:1', 'N:SAR:2', 'CA:SAR:2']),
        'phi_1' : map(atoms.index, ['C2:ace:1', 'N:SAR:2', 'CA:SAR:2', 'C:SAR:2']),
        'psi_1' : map(atoms.index, ['N:SAR:2', 'CA:SAR:2', 'C:SAR:2', 'N:SAR:3']),
        'omega_2' : map(atoms.index, ['CA:SAR:2', 'C:SAR:2', 'N:SAR:3', 'CA:SAR:3']),
        'phi_2' : map(atoms.index, ['C:SAR:2', 'N:SAR:3', 'CA:SAR:3', 'C:SAR:3']),
        'psi_2' : map(atoms.index, ['N:SAR:3', 'CA:SAR:3', 'C:SAR:3', 'N2:ndm:4']),
        'omega_3' : map(atoms.index, ['CA:SAR:3', 'C:SAR:3', 'N2:ndm:4', 'C6:ndm:4'])
    }

print(atoms)

# Initialize system
system = prmtop.createSystem(
        nonbondedMethod=app.NoCutoff if prmtop.topology.getNumChains() == 1 else app.PME,
        implicitSolvent=app.GBn2,
        implicitSolventSaltConc=args.salt_molarity*unit.moles/unit.liter,
        constraints=None,
        rigidWater=True,
        removeCMMotion=False,
)



# Add forces
omega_1 = ufedmm.CollectiveVariable('omega_1', openmm.CustomTorsionForce('theta')) 
omega_1.force.addTorsion(*dihedral_atoms['omega_1'], [])
phi_1 = ufedmm.CollectiveVariable('phi_1', openmm.CustomTorsionForce('theta'))
phi_1.force.addTorsion(*dihedral_atoms['phi_1'], [])
psi_1 = ufedmm.CollectiveVariable('psi_1', openmm.CustomTorsionForce('theta'))
psi_1.force.addTorsion(*dihedral_atoms['psi_1'], [])
omega_2 = ufedmm.CollectiveVariable('omega_2', openmm.CustomTorsionForce('theta')) 
omega_2.force.addTorsion(*dihedral_atoms['omega_2'], [])
phi_2 = ufedmm.CollectiveVariable('phi_2', openmm.CustomTorsionForce('theta'))
phi_2.force.addTorsion(*dihedral_atoms['phi_2'], [])
psi_2 = ufedmm.CollectiveVariable('psi_2', openmm.CustomTorsionForce('theta'))
psi_2.force.addTorsion(*dihedral_atoms['psi_2'], [])
omega_3 = ufedmm.CollectiveVariable('omega_3', openmm.CustomTorsionForce('theta')) 
omega_3.force.addTorsion(*dihedral_atoms['omega_3'], [])

# Initialize extended variables
s_omega_1 = ufedmm.DynamicalVariable('s_omega_1', -limit, limit, mass, Ts, phi, Ks, sigma=None)
s_phi_1 = ufedmm.DynamicalVariable('s_phi_1', -limit, limit, mass, Ts, phi, Ks, sigma=None)
s_psi_1 = ufedmm.DynamicalVariable('s_psi_1', -limit, limit, mass, Ts, psi, Ks, sigma=None)
s_omega_2 = ufedmm.DynamicalVariable('s_omega_2', -limit, limit, mass, Ts, phi, Ks, sigma=None)
s_phi_2 = ufedmm.DynamicalVariable('s_phi_2', -limit, limit, mass, Ts, phi, Ks, sigma=None)
s_psi_2 = ufedmm.DynamicalVariable('s_psi_2', -limit, limit, mass, Ts, psi, Ks, sigma=None)
s_omega_3 = ufedmm.DynamicalVariable('s_omega_3', -limit, limit, mass, Ts, phi, Ks, sigma=None)

# Set up simulation
ufed = ufedmm.UnifiedFreeEnergyDynamics([s_omega_1, s_phi_1, s_psi_1, s_omega_2, s_phi_2, s_psi_2, s_omega_3], temp, height, deposition_period)
ufedmm.serialize(ufed, 'ufed_object.yml')
integrator = ufedmm.MiddleMassiveGGMTIntegrator(temp,40*unit.femtoseconds, dt, scheme='VV-Middle')
integrator.setRandomNumberSeed(seed)
platform = openmm.Platform.getPlatformByName(args.platform)
simulation = ufed.simulation(prmtop.topology, system, integrator, platform)
simulation.context.setPositions(pdb.positions)
simulation.context.setVelocitiesToTemperature(temp, seed)
output = ufedmm.Tee(stdout, 'tri_output_2.csv')
reporter = ufedmm.StateDataReporter(output, 10, step=True, multipleTemperatures=False, hillHeights=False, variables=True, speed=True)
simulation.reporters.append(reporter)

# Minimize energy and run
simulation.minimizeEnergy()
simulation.step(nsteps)
