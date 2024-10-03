from models_rqc import *
from train_transformer import *
nr = 5

# QuAN_50
model = QuANn(set_size=10000, channel=16, dim_output=1, kersize=2, stride=1, \
              dim_hidden=16, num_heads=4, Nr=nr, Nc=nr, p_outputs=1, miniset=5, minisettype='miniset_A2', ln=True)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(model, total_params)

# QuAN_2
model = QuAN2(set_size=10000, channel=16, dim_output=1, kersize=2, stride=1, \
              dim_hidden=16, num_heads=4, Nr=nr, Nc=nr, p_outputs=1,  ln=True)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(model, total_params)

# PAB
model = PAB(set_size=10000, channel=16, dim_output=1, kersize=2, stride=1, \
              dim_hidden=16, num_heads=4, Nr=nr, Nc=nr, p_outputs=1,  ln=True)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(model, total_params)

# SMLP
model = Set_MLP(set_size=10000, channel=16, dim_output=1, kersize=2, stride=1, \
              dim_hidden=16, num_heads=4, Nr=nr, Nc=nr, p_outputs=1,  ln=True)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(model, total_params)

# Transf.
model = make_model(Lv=5, Lh=5, Vsrc=2, Vtgt=1, Nl=6, d_model=16, d_ff = 32, h=4)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(model, total_params)

# CNN
model = Conv3d(set_size=10000, channel=16, dim_output=1, kersize=2, stride=1, \
              dim_hidden=16, num_heads=4, Nr=nr, Nc=nr, p_outputs=1,  ln=True)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(model, total_params)

# MLP
model = MLP3d(set_size=10000, channel=16, dim_output=1, kersize=2, stride=1, \
              dim_hidden=16, num_heads=4, Nr=nr, Nc=nr, p_outputs=1,  ln=True)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(model, total_params)

