from cubdl.example_picmus_torch import load_datasets; 
data = load_datasets("simulation", "resolution_distorsion", "iq")
print(data.idata.shape)