from __future__ import division
from __future__ import print_function
from typing import final
from PIL.ImagePalette import raw
import numpy as np
import time
import cantera as ct
import matplotlib.pyplot as plt
print('Running Cantera version: ' + ct.__version__)

MixtureTemperature = [950, 1000, 1050, 1100, 1150, 1200]  # change
Phi = [0.5, 0.75, 0.9, 1.0, 1.25, 1.5]

# all -6 except cn1 which is -9 for the mix. temp. shown above.
timestep = [1e-6]
nt = int(4.5e-4/timestep[0])
dt = timestep[0]
fuel = "H2"
oxidizer = "O2:1.0,N2:3.76"
#oxidizer = "O2"

gas = ct.Solution('conaire2004.cti')

final_input_nodes = None
final_output_nodes = None

for gastemp in MixtureTemperature:

    for phi in Phi:

        # Initial temperature, Pressure and stoichiometry
        gas.TP = gastemp, ct.one_atm  # change
        gas.set_equivalence_ratio(phi, fuel, oxidizer)

        pres = np.zeros(nt+1, 'd')
        temp = np.zeros(nt+1, 'd')
        #H_Y = np.zeros(nt+1, 'd')
        H2_Y = np.zeros(nt+1, 'd')
        O_Y = np.zeros(nt+1, 'd')
        O2_Y = np.zeros(nt+1, 'd')
        OH_Y = np.zeros(nt+1, 'd')
        H2O_Y = np.zeros(nt+1, 'd')
        N2_Y = np.zeros(nt+1, 'd')
        #HO2_Y = np.zeros(nt+1, 'd')
        #H2O2_Y = np.zeros(nt+1, 'd')

        # Create the batch reactor and fill it with the gas
        r = ct.IdealGasConstPressureReactor(contents=gas, name='Batch Reactor')
        #r = ct.IdealGasReactor(contents=gas, name='Batch Reactor')

        # Now create a reactor network consisting of the single batch reactor
        sim = ct.ReactorNet([r])

        pres[0] = r.thermo.P
        temp[0] = r.thermo.T
        #H_Y[0] = r.thermo.Y[gas.species_index('H')]
        H2_Y[0] = r.thermo.Y[gas.species_index('H2')]
        O_Y[0] = r.thermo.Y[gas.species_index('O')]
        O2_Y[0] = r.thermo.Y[gas.species_index('O2')]
        OH_Y[0] = r.thermo.Y[gas.species_index('OH')]
        H2O_Y[0] = r.thermo.Y[gas.species_index('H2O')]
        N2_Y[0] = r.thermo.Y[gas.species_index('N2')]
        #HO2_Y[0] = r.thermo.Y[gas.species_index('HO2')]
        #H2O2_Y[0] = r.thermo.Y[gas.species_index('H2O2')]

    # Run the simulation

    # Initial simulation time
        time = 0.0

    # Loop for nt time steps of dt seconds.
        for n in range(1, nt+1):
            time += dt
            sim.advance(time)
            pres[n] = r.thermo.P
            temp[n] = r.thermo.T
            #H_Y[n] = r.thermo.Y[gas.species_index('H')]
            H2_Y[n] = r.thermo.Y[gas.species_index('H2')]
            O_Y[n] = r.thermo.Y[gas.species_index('O')]
            O2_Y[n] = r.thermo.Y[gas.species_index('O2')]
            OH_Y[n] = r.thermo.Y[gas.species_index('OH')]
            H2O_Y[n] = r.thermo.Y[gas.species_index('H2O')]
            N2_Y[n] = r.thermo.Y[gas.species_index('N2')]
            #HO2_Y[n] = r.thermo.Y[gas.species_index('HO2')]
            #H2O2_Y[n] = r.thermo.Y[gas.species_index('H2O2')]

        pres = pres.reshape(nt+1, 1)
        temp = temp.reshape(nt+1, 1)
        #H_Y = H_Y.reshape(nt+1, 1)
        H2_Y = H2_Y.reshape(nt+1, 1)
        O_Y = O_Y.reshape(nt+1, 1)
        O2_Y = O2_Y.reshape(nt+1, 1)
        OH_Y = OH_Y.reshape(nt+1, 1)
        H2O_Y = H2O_Y.reshape(nt+1, 1)
        N2_Y = N2_Y.reshape(nt+1, 1)
        #HO2_Y = HO2_Y.reshape(nt+1, 1)
        #H2O2_Y = H2O2_Y.reshape(nt+1, 1)

        raw_nodes = np.hstack((temp, pres, H2_Y, O_Y, O2_Y, OH_Y, H2O_Y, N2_Y))

        #raw_nodes = np.hstack(
        #    (temp, pres, H_Y, H2_Y, O_Y, O2_Y, OH_Y, H2O_Y, N2_Y, HO2_Y, H2O2_Y))

        if final_input_nodes is None:
            final_input_nodes = raw_nodes[:-1, :]
            final_output_nodes = raw_nodes[1:, 2:]
        else:
            final_input_nodes = np.vstack(
                (final_input_nodes, raw_nodes[:-1, :]))
            final_output_nodes = np.vstack(
                (final_output_nodes, raw_nodes[1:, 2:]))

print(final_input_nodes.shape)
print(final_output_nodes.shape)

np.save('input_1.npy', final_input_nodes)
#np.save('output_H.npy', final_output_nodes[:, 0])
np.save('output_H2.npy', final_output_nodes[:,0])
np.savetxt('output_H2.csv', final_output_nodes[:,0], delimiter=';')
np.save('output_O.npy', final_output_nodes[:, 1])
np.save('output_O2.npy', final_output_nodes[:, 2])
np.save('output_OH.npy', final_output_nodes[:, 3])
np.save('output_H2O.npy', final_output_nodes[:, 4])
np.save('output_N2.npy', final_output_nodes[:, 5])
#np.save('output_HO2.npy', final_output_nodes[:, 7])
#np.save('output_H2O2.npy', final_output_nodes[:, 8])


#np.savetxt('input_1.csv', final_input_nodes, delimiter=';')
#np.savetxt('output_1.csv', final_output_nodes, delimiter=';')
