import numpy as np
import pickle


filename = '~/QuAN/Data_src/HCBH_dataset_n8/coherent-like_state_measurements'
g = open(filename, 'rb')
data = pickle.load(g)
g.close()

print(data['state_vectors'].shape)
print(data['pauli_strings'])

for basis in range(7):
    for n_d in range(2,11):
        print(n_d-6, data['drive_detunings'][n_d]/data['J'])
        for n_t in range(14, 31):
            pdf = (data['state_vectors'][n_d, basis, n_t]*10000)
            strings = []
            for string, count in enumerate(pdf):
                strings += [string]*round(count)
            np.random.shuffle(strings)
            bitstring = np.array([bin(string).replace('0b', '').zfill(16) for i, string in enumerate (strings)])
            bitstrings = np.array([list(bits) for i, bits in enumerate(bitstring)], dtype = int)
            if bitstrings.shape[0] != 10000: print('Error'); break
                
            np.savez_compressed(f'~/QuAN/Data_src/HCBH_dataset_n8/state_{basis:d}_d{n_d-6:d}_t{n_t}', s=bitstrings, c=None)
