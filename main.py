import numpy as np
from pyscf import gto, dft
from pyscf.dft import numint
from pyscf.tools import cubegen
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes


def molecular_structure_from_smiles(smiles: str) -> gto.Mole:
    mol_rdkit = Chem.MolFromSmiles(smiles)
    mol_rdkit = Chem.AddHs(mol_rdkit)
    AllChem.EmbedMolecule(mol_rdkit, AllChem.ETKDG())
    AllChem.MMFFOptimizeMolecule(mol_rdkit)
    atoms = [[a.GetSymbol(), mol_rdkit.GetConformer().GetAtomPosition(a.GetIdx())] for a in mol_rdkit.GetAtoms()]
    mol_pyscf = gto.Mole(atom=atoms, basis="6-31g*").build()
    return mol_pyscf

def run_rks_on_molecular_structure(mol_pyscf: gto.Mole) -> dft.rks.RKS:
    mf = dft.RKS(mol_pyscf)
    mf.xc = "b3lyp"
    mf.kernel()
    return mf

def process_homo(mf: dft.rks.RKS) -> tuple[np.ndarray, np.ndarray]:
    n_electrons = mol_pyscf.nelectron
    homo_index = n_electrons // 2 - 1

    nx, ny, nz = 128, 128, 128
    cube = cubegen.Cube(mol_pyscf, nx, ny, nz, margin=10.0)
    grid_coords = cube.get_coords()

    grid_origin = grid_coords[0]

    ao_values = numint.eval_ao(mol_pyscf, grid_coords, deriv=0)
    mo_coeff_homo = mf.mo_coeff[:, homo_index]
    orbital_values = np.dot(ao_values, mo_coeff_homo)
    orbital_cube = orbital_values.reshape(nx, ny, nz)

    smoothing_sigma = 5
    surface_isovalue = 0.00006

    smoothed_orbital_cube = gaussian_filter(orbital_cube, sigma=smoothing_sigma)

    verts, faces, normals, values = marching_cubes(
        smoothed_orbital_cube,
        level=surface_isovalue,
        spacing=(cube.xs[1] - cube.xs[0], cube.ys[1] - cube.ys[0], cube.zs[1] - cube.zs[0])
    )

    verts += grid_origin

    return verts, faces


def save_mesh_as_obj(vertices: np.ndarray, faces: np.ndarray, filename: str) -> None:
    with open(filename, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

if __name__ == "__main__":
    output_obj_file = f"artistic_surface.obj"
    smiles = "CCCCCCC=CCCCCCCCc1cccc(O)c1O"
    mol_pyscf = molecular_structure_from_smiles(smiles)
    mf = run_rks_on_molecular_structure(mol_pyscf)
    verts, faces = process_homo(mf)
    save_mesh_as_obj(verts, faces, output_obj_file)
