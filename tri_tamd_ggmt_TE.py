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
        'phi' : map(atoms.index, ['C2:ace:1', 'N:SAR:2', 'CA:SAR:2', 'C:SAR:2']),
        'psi' : map(atoms.index, ['N:SAR:2', 'CA:SAR:2', 'C:SAR:2', 'N:SAR:3'])
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
phi = ufedmm.CollectiveVariable('phi', openmm.CustomTorsionForce('theta'))
phi.force.addTorsion(*dihedral_atoms['phi'], [])
psi = ufedmm.CollectiveVariable('psi', openmm.CustomTorsionForce('theta'))
psi.force.addTorsion(*dihedral_atoms['psi'], [])

# Initialize extended variables
s_phi = ufedmm.DynamicalVariable('s_phi', -limit, limit, mass, Ts, phi, Ks, sigma=None)
s_psi = ufedmm.DynamicalVariable('s_psi', -limit, limit, mass, Ts, psi, Ks, sigma=None)

# Set up simulation
ufed = ufedmm.UnifiedFreeEnergyDynamics([s_phi, s_psi], temp, height, deposition_period)
ufedmm.serialize(ufed, 'ufed_object.yml')
integrator = ufedmm.MiddleMassiveGGMTIntegrator(temp,40*unit.femtoseconds, dt, scheme='VV-Middle')
integrator.setRandomNumberSeed(seed)
platform = openmm.Platform.getPlatformByName(args.platform)
simulation = ufed.simulation(prmtop.topology, system, integrator, platform)
simulation.context.setPositions(pdb.positions)
simulation.context.setVelocitiesToTemperature(temp, seed)
output = ufedmm.Tee(stdout, 'tri_output.csv')
reporter = ufedmm.StateDataReporter(output, 10, step=True, multipleTemperatures=False, hillHeights=False, variables=True, speed=True)
simulation.reporters.append(reporter)

# Minimize energy and run
simulation.minimizeEnergy()
simulation.step(nsteps)
