# QuAN

This is the code repository for "Attention to Quantum complexity" (https://arxiv.org/pdf/2405.11632)

Due to the large volume, the `Data_src` folder is not present in this repository. It is provided upon reasonable request.
- `HCBH`: data preprocessing, training, and testing results for driven hard-core bose-Hubbard model.
- `RQC`: data preprocessing, training, and testing for the random quantum circuit.
- `TC`: data preprocessing, training, and testing for the mixed-state toric code.
- `Figure`: `Data_out` folder contains training outcomes and testing results. `*.ipynb` files generates figure for the main/SM text in https://arxiv.org/abs/2405.11632.
- python version 3.10.6, pytorch version 1.13.0

## HCBH (driven hard-core bose-Hubbard model)
### data preprocessing
```python3 data_generation_hcbh.py```

### training
Run the bash file using the command below. Pick `model_name` from `c21` (QuAN4), `c11` (QuAN2), `c01` (PAB), `c00` (SMLP).
```
bash HCBH/train_QuAN_hcbh.sh "model_name"
```

### testing
```
python3 HCBH/test_phasediagram_hcbh.py
```

## RQC (random quantum circuit)
### training
No need to preprocess the data. Run the bash file using the command below. 
- Pick system size `Nq` from 20, 25, 30, 36.
- `Depth` refers to the circuit depth, ranges from 4 to 20 with only even numbers.
```
bash RQC/train_QuAN_rqc.sh "Nq" "Depth"
```

### testing
- Pick system size `Nq`.
- `Noise` refers to the noise level in the data. Pick between `0` (simulated data) and `google` (experimental data).
```
bash RQC/test_QuAN_rqc.sh "Nq" "Noise"
```

## TC (Mixed-state Toric code)
### data preprocessing
Run `data_generation_tc.ipynb`.
### training
Run the bash file using the command below. Pick model_name from `c11` (QuAN2), `c01` (PAB), `c00` (SMLP).
```
bash TC/train_QuAN_tc.sh "model_name"
```

### testing
```
python3 TC/test_phasediagram_tc.py
python3 TC/test_confidence_tc.py
```

