import time
from matplotlib import pyplot as plt
import numpy as np
import cantera as ct
import os
import torch

# try:
#     os.chdir("F:/Documents/Masters/HiWi/Cantera")
# except:
#     os.chdir("./Cantera/")

MixtureTemperature = [1200]
gastemp = MixtureTemperature[0]
timestep = [1e-6]

dt = timestep[0]

nt = int(1.7e-4/dt)

fuel = "H2"
oxidizer = "O2:1.0, N2:3.76"

phi = 0.5

n_runs = 500

ODE_times = np.zeros(n_runs)
ANN_times = np.zeros(n_runs)

temp_mean = 1884.074369553378
temp_std = 433.62573988997286
temp_min = -1.577568693177102
temp_ptp = 2.4984734392012085
H2_mean = 0.004196851599658142
H2_std = 0.005872701842874536
H2_min = -0.6578026261588739
H2_ptp = 2.4066852553305984
O_mean = 0.00548001627256922
O_std = 0.0044416373696016965
O_min = -1.2337829085449719
O_ptp = 3.921584082993612
O2_mean = 0.1408999623686791
O2_std = 0.051084186856168334
O2_min = -0.623821397036226
O2_ptp = 2.3607351211584477
OH_mean = 0.00922919219697454
OH_std = 0.00543497046557566
OH_min = -1.6981126678481413
OH_ptp = 2.5963241667141967
H2O_mean = 0.08393592703309831
H2O_std = 0.04880731005035305
H2O_min = -1.7197408942739134
H2O_ptp = 2.454892711296279
N2_mean = 0.7559036945562518
N2_std = 1.1102230246251565e-16
N2_min = 1.0
N2_ptp = np.finfo(float).eps

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def extract_parameters(net):
    parameters = []
    for name, param in net.named_parameters():
        parameters.append(param)
    W1 = parameters[0].detach().numpy()
    B1 = parameters[1].detach().numpy()
    W2 = parameters[2].detach().numpy()
    B2 = parameters[3].detach().numpy()
    return W1, B1, W2, B2

net_temp = torch.jit.load("../ANN_results/final_temp.pt")
net_temp.eval()
W1_temp, B1_temp, W2_temp, B2_temp = extract_parameters(net_temp)
net_H2 = torch.jit.load("../ANN_results/final_H2.pt")
net_H2.eval()
W1_H2, B1_H2, W2_H2, B2_H2 = extract_parameters(net_H2)
net_O = torch.jit.load("../ANN_results/final_O.pt")
net_O.eval()
W1_O, B1_O, W2_O, B2_O = extract_parameters(net_O)
net_O2 = torch.jit.load("../ANN_results/final_O2.pt")
net_O2.eval()
W1_O2, B1_O2, W2_O2, B2_O2 = extract_parameters(net_O2)
net_OH = torch.jit.load("../ANN_results/final_OH.pt")
net_OH.eval()
W1_OH, B1_OH, W2_OH, B2_OH = extract_parameters(net_OH)
net_H2O = torch.jit.load("../ANN_results/final_H2O.pt")
net_H2O.eval()
W1_H2O, B1_H2O, W2_H2O, B2_H2O = extract_parameters(net_H2O)
net_N2 = torch.jit.load("../ANN_results/final_N2.pt")
net_N2.eval()
W1_N2, B1_N2, W2_N2, B2_N2 = extract_parameters(net_N2)

for j in range(n_runs):
    gas = ct.Solution("conaire2004.cti")
    gas.TP = gastemp, ct.one_atm
    gas.set_equivalence_ratio(phi, fuel, oxidizer)
    Time = np.zeros(nt+1, 'd')
    temp = np.zeros(nt+1, 'd')
    H_Y = np.zeros(nt+1, 'd')
    H2_Y = np.zeros(nt+1, 'd')
    O_Y = np.zeros(nt+1, 'd')
    O2_Y = np.zeros(nt+1, 'd')
    OH_Y = np.zeros(nt+1, 'd')
    H2O_Y = np.zeros(nt+1, 'd')
    N2_Y = np.zeros(nt+1, 'd')

    r = ct.IdealGasConstPressureReactor(contents=gas, name='Batch Reactor')

    sim = ct.ReactorNet([r])

    Time[0] = 0
    temp[0] = r.thermo.T
    H2_Y[0] = r.thermo.Y[gas.species_index('H2')]
    O_Y[0] = r.thermo.Y[gas.species_index('O')]
    O2_Y[0] = r.thermo.Y[gas.species_index('O2')]
    OH_Y[0] = r.thermo.Y[gas.species_index('OH')]
    H2O_Y[0] = r.thermo.Y[gas.species_index('H2O')]
    N2_Y[0] = r.thermo.Y[gas.species_index('N2')]

    time_ = 0.0
    
    for i in range(1, nt+1):
        time_ += dt
        sim.advance(time_)
        Time[i] = time_
        temp[i] = r.thermo.T
        H2_Y[i] = r.thermo.Y[gas.species_index('H2')]
        O_Y[i] = r.thermo.Y[gas.species_index('O')]
        O2_Y[i] = r.thermo.Y[gas.species_index('O2')]
        OH_Y[i] = r.thermo.Y[gas.species_index('OH')]
        H2O_Y[i] = r.thermo.Y[gas.species_index('H2O')]
        N2_Y[i] = r.thermo.Y[gas.species_index('N2')]
        if((OH_Y[i] - OH_Y[i-1]) > 1e-6):
            break
    n = i

    T = temp[n]
    H2 = H2_Y[n]
    O = O_Y[n]
    O2 = O2_Y[n]
    OH = OH_Y[n]
    H2O = H2O_Y[n]
    N2 = N2_Y[n]


    start = time.time()

    for i in range(n+1, nt+1):
        #print(i)
        #T = temp[i-1]
        #T = (T - temp_mean)/temp_std
        #T = -1 + 2*(T - temp_min)/temp_ptp
        #T = 0.0018460361650363035*T - 3.215243356964089
        #H2 = H2_Y[i-1]
        #H2 = (H2 - H2_mean)/H2_std
        #H2 = -1 + 2*(H2 - H2_min)/H2_ptp
        #H2 = 141.5053129450966*H2 - 1.0472306393416786
        #O = O_Y[i-1]
        #O = (O - O_mean)/O_std
        #O = -1 + 2*(O - O_min)/O_ptp
        #O = 114.82206843471701*O - 1.0
        #O2 = O2_Y[i-1]
        #O2 = (O2 - O2_mean)/O2_std
        #O2 = -1 + 2*(O2 - O2_min)/O2_ptp
        #O2 = 16.58426580068494*O2 - 2.8082248491267303
        #OH = OH_Y[i-1]
        #OH = (OH - OH_mean)/OH_std
        #OH = -1 + 2*(OH - OH_min)/OH_ptp
        #OH = 141.7339502487151*OH -1.0000000000000002
        #H2O = H2O_Y[i-1]
        #H2O = (H2O - H2O_mean)/H2O_std
        #H2O = -1 + 2*(H2O - H2O_min)/H2O_ptp
        #H2O = 16.692162595666428*H2O - 1.0
        N2 = N2_Y[i-1]
        N2 = (N2 - N2_mean)/N2_std
        N2 = -1 + 2*(N2 - N2_min)/N2_ptp

        # inp = np.array([0.0018460361650363035*temp[i-1] - 3.215243356964089, 141.5053129450966*H2_Y[i-1] - 1.0472306393416786, 
        #                 114.82206843471701*O_Y[i-1] - 1.0, 16.58426580068494*O2_Y[i-1] - 2.8082248491267303, 141.7339502487151*OH_Y[i-1] -1.0000000000000002, 
        #                 16.692162595666428*H2O_Y[i-1] - 1.0, N2], dtype = float)

        #T = (net_temp(torch.reshape(torch.Tensor(inp),(1,7))).detach().numpy()[0][0]+1.0)*temp_ptp*0.5 + temp_min
        #T = np.matmul(W1_temp, inp) + B1_temp
        T = tanh(np.array([-7.82901316e-04*temp[i-1]+1.36358014e+00 + 71.02654567*H2_Y[i-1]-0.52564228 + 54.15356658*O_Y[i-1] -0.4716303 + 8.57580843*O2_Y[i-1] -1.45214739 \
            + 28.99373183*OH_Y[i-1] -0.20456448 + 4.1682707*H2O_Y[i-1]-0.24971424 + 0.12049278*N2 +  0.16342281, #0
            1.07503362e-04*temp[i-1]-1.87238732e-01 + 3.18053851*H2_Y[i-1]-0.02353804 + 20.65316601*O_Y[i-1] -0.17987105 - 7.87263854*O2_Y[i-1] + 1.33307916 \
            + 6.46384909*OH_Y[i-1] -0.04560551 - 8.05380053*H2O_Y[i-1] + 0.48248994 + 0.28593954*N2 + 0.09447618, #1
            -4.49098251e-04*temp[i-1] + 7.82194951e-01 - 35.25344927*H2_Y[i-1] + 0.26089828 + 31.44448847*O_Y[i-1] - 0.27385405 - 4.90536296*O2_Y[i-1] + 0.8306284 \
            + 60.50705959*OH_Y[i-1] -0.4269059 - 4.26715774*H2O_Y[i-1] + 0.2556384 - 0.4535593*N2 - 0.00609491, #2
            -7.39706636e-04*temp[i-1] + 1.28834792e+00 + 77.58464619*H2_Y[i-1] - 0.57417645 - 53.54516233*O_Y[i-1] + 0.46633163 - 7.28351716*O2_Y[i-1] + 1.23332284 \
            - 35.82929379*OH_Y[i-1] + 0.2527926 - 5.69828283*H2O_Y[i-1] + 0.34137475 - 0.45130032*N2 - 0.00698619, #3
            7.46762186e-04*temp[i-1] - 1.30063658e+00 - 48.91676645*H2_Y[i-1] + 0.36201564 -13.07386806*O_Y[i-1] + 0.11386198 -7.62422266*O2_Y[i-1] + 1.29101474 \
            + 33.15201743*OH_Y[i-1] -0.23390315 -7.07403535*H2O_Y[i-1] + 0.42379382 - 0.36562628*N2 - 0.03995477, #4
            5.35616539e-04*temp[i-1] -9.32883954e-01 -71.72758563*H2_Y[i-1] + 0.53083042 + 25.19433174*O_Y[i-1] - 0.21942064 -9.77260533*O2_Y[i-1] + 1.65480181 \
            + 11.9014735*OH_Y[i-1] -0.08397052 + 5.53812504*H2O_Y[i-1] -0.33177996 + 0.24671593*N2 + 0.05717678, #5
            9.08596479e-04*temp[i-1] -1.58250355e+00 + 31.35743584*H2_Y[i-1] -0.23206526 -13.60439654*O_Y[i-1] + 0.11848242 + 8.17124388*O2_Y[i-1] - 1.38364221 \
            -74.04672669*OH_Y[i-1] + 0.52243465 + 1.69779725*H2O_Y[i-1] -0.10171224 + 0.25353682*N2 + 0.00968669, #6
            -9.92548167e-04*temp[i-1] + 1.72872231e+00 + 66.16404105*H2_Y[i-1] -0.4896566 -10.39946459*O_Y[i-1] + 0.09057026 + 9.05466749*O2_Y[i-1] - 1.53323292 \
            -19.85750329*OH_Y[i-1] + 0.14010407 + 3.49724493*H2O_Y[i-1] -0.20951419 -0.24792303*N2 + 0.11718795, #7
            -3.06957269e-04*temp[i-1] + 5.34627836e-01 -25.89710666*H2_Y[i-1] + 0.1916553 -44.79215434*O_Y[i-1] + 0.39010057 + 0.25192528*O2_Y[i-1] -0.04265868 \
            + 77.12865916*OH_Y[i-1] -0.54417914 -6.85349917*H2O_Y[i-1] + 0.41058186 -0.2558582*N2 + 0.05204082, #8
            -3.07614163e-04*temp[i-1] + 5.35771948e-01 -48.3069111*H2_Y[i-1] + 0.35750232 + 11.59876043*O_Y[i-1] -0.10101508 + 2.20741404*O2_Y[i-1] -0.3737829 \
            + 5.54974164*OH_Y[i-1] -0.03915605 -2.23543511*H2O_Y[i-1] + 0.13392124 -0.35659584*N2 - 0.00284972, #9
            ], dtype=float))
        #T = tanh(T)
        T = (np.matmul(W2_temp, T) + B2_temp)[0]
        #T = (T + 1.0)*temp_ptp*0.5 + temp_min
        #T = T*temp_std + temp_mean
        temp[i] = 541.7011968345346*T + 1741.701174581734
        #H2 = (net_H2(torch.reshape(torch.Tensor(inp),(1,7))).detach().numpy()[0][0]+1.0)*H2_ptp*0.5 + H2_min
        #H2 = np.matmul(W1_H2, inp) + B1_H2
        H2 = tanh(np.array([5.91633657e-04*temp[i-1]-1.03044904e+00 + 49.53619181*H2_Y[i-1]-0.36659979 -15.16945348*O_Y[i-1] + 0.1321127 +11.72368334*O2_Y[i-1] -1.98517916 \
            -44.88387366*OH_Y[i-1] + 0.31667694 + 4.387683*H2O_Y[i-1]-0.26285887 + 0.537303*N2 -0.17578924, #0
            7.84964299e-04*temp[i-1]-1.36717324e+00 -27.24700932*H2_Y[i-1] + 0.20164545 + 64.05962237*O_Y[i-1] -0.5579034 -6.83825367*O2_Y[i-1] + 1.15792608 \
            -54.66178991*OH_Y[i-1] + 0.38566476 + 5.60033155*H2O_Y[i-1]-0.33550665 + 0.39647752*N2 + 0.04157723, #1
            7.53580945e-05*temp[i-1]-1.31251282e-01 + 24.23824836*H2_Y[i-1] -0.17937868 + 36.67571646*O_Y[i-1] -0.31941348 -8.69457664*O2_Y[i-1] + 1.47225849 \
            +15.75178682*OH_Y[i-1] -0.1111363 + 4.33837318*H2O_Y[i-1]-0.2599048 - 0.672414*N2 + 0.1025602, #2
            8.41333825e-04*temp[i-1]-1.46535211e+00 -0.68028679*H2_Y[i-1] + 0.00503456 -44.09070288*O_Y[i-1] + 0.38399154 -2.92801913*O2_Y[i-1] + 0.49580344 \
            -56.81352104*OH_Y[i-1] + 0.40084624 + 9.48579378*H2O_Y[i-1]-0.5682783 + 0.03225607*N2 + 0.05296063, #3
            -4.45014653e-04*temp[i-1] + 7.75082543e-01 + 17.48505871*H2_Y[i-1] -0.12940072 -41.85990759*O_Y[i-1] + 0.36456326 -7.04885361*O2_Y[i-1] + 1.19358711 \
            -40.34434216*OH_Y[i-1] + 0.2846484 -1.69105729*H2O_Y[i-1] + 0.10130846 -0.21707626*N2 + 0.00020093, #4
            8.45462070e-04*temp[i-1] -1.47254228e+00 -36.25738715*H2_Y[i-1] + 0.26832806 + 27.27709339*O_Y[i-1] -0.23755968 -0.31398518*O2_Y[i-1] + 0.05316732 \
            -86.78422695*OH_Y[i-1] + 0.61230373 + 7.75241527*H2O_Y[i-1] -0.46443444 + 0.0035437415*N2 - 0.034982275, #5
            -1.60117368e-04*temp[i-1] + 2.78876609e-01 -24.74902765*H2_Y[i-1] + 0.18315878 -16.14975691*O_Y[i-1] + 0.14065029 -8.72981468*O2_Y[i-1] + 1.47822537 \
            + 76.06384789*OH_Y[i-1] -0.53666639 -1.80370675*H2O_Y[i-1] + 0.1080571 + 0.37241516*N2 + 0.015036589, #6
            -3.08349567e-04*temp[i-1] + 5.37052803e-01 -74.34131388*H2_Y[i-1] + 0.5501737 -42.10585745*O_Y[i-1] + 0.36670527 + 2.49526575*O2_Y[i-1] -0.42252502 \
            + 0.361902*OH_Y[i-1] -0.00255339 + 5.77730825*H2O_Y[i-1] -0.34610903 + 0.2976406*N2 -0.032516055, #7
            4.21897182e-04*temp[i-1] -7.34818818e-01 + 78.31079171*H2_Y[i-1] -0.5795504 -21.10396048*O_Y[i-1] + 0.18379708 + 6.25287521*O2_Y[i-1] -1.05880355 \
            -28.60917696*OH_Y[i-1] + 0.20185126 + 1.69738351*H2O_Y[i-1] -0.10168745 -0.088204354*N2 -0.008780442, #8
            1.06089167e-03*temp[i-1] -1.84775627e+00 -74.91138478*H2_Y[i-1] + 0.55439259 + 11.62027985*O_Y[i-1] - 0.1012025 + 8.59414612*O2_Y[i-1] -1.45525253 \
            + 20.44909155*OH_Y[i-1] - 0.144278 + 0.23328668*H2O_Y[i-1] -0.01397582 + 0.60529536*N2 -0.025092002, #9
            ], dtype=float))
        #H2 = tanh(H2)
        H2 = (np.matmul(W2_H2, H2) + B2_H2)[0]
        #H2 = (H2 + 1.0)*H2_ptp*0.5 + H2_min
        #H2 = H2*H2_std + H2_mean
        H2_Y[i] = 0.007066872467099489*H2 + 0.007400645371866703
        #O = (net_O(torch.reshape(torch.Tensor(inp),(1,7))).detach().numpy()[0][0]+1.0)*O_ptp*0.5 + O_min
        #O = np.matmul(W1_O, inp) + B1_O
        #O = tanh(O)
        O = tanh(np.array([-8.29788405e-04*temp[i-1] + 1.44524344e+00 + 48.24765093*H2_Y[i-1]-0.35706375 + 9.58030407*O_Y[i-1] -0.08343609 -8.55812224*O2_Y[i-1] + 1.44915258 \
            -44.26913433*OH_Y[i-1] + 0.31233966 -6.39942879*H2O_Y[i-1] + 0.38337925 - 0.14302148*N2 + 0.027550502, #0
            4.10945500e-04*temp[i-1]-7.15744259e-01 + 21.59094292*H2_Y[i-1] -0.15978691 + 42.48701294*O_Y[i-1] -0.3700248 -3.08896039*O2_Y[i-1] + 0.52305573 \
            + 74.46829069*OH_Y[i-1] -0.52540898 -2.11140926*H2O_Y[i-1] + 0.12649105 -0.13834399*N2 + 0.022833882, #1
            5.96330995e-04*temp[i-1]-1.03863039e+00 -5.03708401*H2_Y[i-1] + 0.03727767 -45.99195646*O_Y[i-1] + 0.4005498 + 6.50737621*O2_Y[i-1] -1.10189838 \
            -62.67121803*OH_Y[i-1] + 0.44217506 -6.04189756*H2O_Y[i-1] + 0.36196014 -0.12718129*N2 -0.028727852, #2
            -6.97527729e-04*temp[i-1] + 1.21488486e+00 -50.2809381*H2_Y[i-1] + 0.37211139 + 15.59137446*O_Y[i-1] -0.13578726 -3.56410283*O2_Y[i-1] + 0.60351192 \
            + 65.90369384*OH_Y[i-1] -0.4649817 -6.82308459*H2O_Y[i-1] + 0.40875977 + 0.31312883*N2 - 0.008106859, #3
            4.90528253e-04*temp[i-1] -8.54353634e-01 + 75.66219588*H2_Y[i-1] -0.55994908 + 42.96069493*O_Y[i-1] -0.37415016 -6.59640985*O2_Y[i-1] + 1.1169745 \
            -37.38152074*OH_Y[i-1] + 0.26374429 + 6.75478859*H2O_Y[i-1] -0.40466827 + 0.47680983*N2 + 0.006241843, #4
            2.45437169e-05*temp[i-1] -4.27478206e-02 -59.88469579*H2_Y[i-1] + 0.4431854 + 2.41089105*O_Y[i-1] -0.02099676 + 8.74366553*O2_Y[i-1] -1.48057075 \
            + 47.08003373*OH_Y[i-1] -0.33217189 + 7.33406892*H2O_Y[i-1] -0.439372 + 0.39459267*N2 + 0.017852105, #5
            4.41637973e-04*temp[i-1] -7.69201376e-01 + 61.35769593*H2_Y[i-1] -0.45408655 -50.16602402*O_Y[i-1] + 0.43690228 + 0.2776009*O2_Y[i-1] -0.04700635 \
            -32.70346511*OH_Y[i-1] + 0.2307384 + 4.67057401*H2O_Y[i-1] -0.27980641 + 0.41033995*N2 + 0.08484277, #6
            -5.71938973e-04*temp[i-1] + 9.96146781e-01 -15.62217477*H2_Y[i-1] + 0.11561418 + 45.79950861*O_Y[i-1] - 0.39887375 -7.63709694*O2_Y[i-1] + 1.29319475 \
            -40.10416809*OH_Y[i-1] + 0.28295386 + 6.13500762*H2O_Y[i-1] -0.36753821 + 0.24014696*N2 -0.0441051, #7
            -4.72354041e-05*temp[i-1] + 8.22699588e-02 -38.39606182*H2_Y[i-1] + 0.28415564 + 10.83012092*O_Y[i-1] -0.0943209 + 6.28445628*O2_Y[i-1] -1.0641512  \
            -75.4163001*OH_Y[i-1] + 0.53209764 + 8.51844603*H2O_Y[i-1] -0.51032609 + 0.014659929*N2 -0.0010702207, #8
            -4.38747616e-04*temp[i-1] + 7.64167238e-01 -28.19686476*H2_Y[i-1] + 0.208675 -0.2021984*O_Y[i-1] + 0.00176097 + 8.09856843*O2_Y[i-1] -1.37133602 \
            -61.60948161*OH_Y[i-1] + 0.43468401 -1.58023617*H2O_Y[i-1] + 0.09466935 -0.16465998*N2 + 0.030183421, #9
            ], dtype=float))    
        O = (np.matmul(W2_O, O) + B2_O)[0]
        #O = (O + 1.0)*O_ptp*0.5 + O_min
        #O = O*O_std + O_mean
        O_Y[i] = 0.008709127205529814*O + 0.008709127205529814
        #O2 = (net_O2(torch.reshape(torch.Tensor(inp),(1,7))).detach().numpy()[0][0]+1.0)*O2_ptp*0.5 + O2_min
        #O2 = np.matmul(W1_O2, inp) + B1_O2
        #O2 = tanh(O2)
        O2 = tanh(np.array([-4.66104431e-04*temp[i-1] + 8.11814636e-01 + 43.3297615*H2_Y[i-1]-0.3206682 + 52.32499263*O_Y[i-1] -0.45570502 + 4.56626513*O2_Y[i-1] -0.77320874 \
            + 65.09683364*OH_Y[i-1] -0.45928892 + 2.97500314*H2O_Y[i-1] -0.17822754 + 0.5273992*N2 -0.07689126, #0
            -2.19816798e-04*temp[i-1] + 3.82855176e-01 -62.69386101*H2_Y[i-1] + 0.46397503 -22.38090752*O_Y[i-1] + 0.19491817 -7.91675376*O2_Y[i-1] + 1.34054922 \
            + 21.15911442*OH_Y[i-1] -0.14928755 + 5.15839487*H2O_Y[i-1] -0.30903095 -0.67193305*N2 + 0.11066341, #1
            -9.14456632e-04*temp[i-1] + 1.59271019e+00 -39.26436375*H2_Y[i-1] + 0.29058163 -56.52380785*O_Y[i-1] + 0.49227303 -6.09303488*O2_Y[i-1] + 1.03173768 \
            -55.03729529*OH_Y[i-1] + 0.38831413 -2.45574846*H2O_Y[i-1] + 0.14711985 -0.45480683*N2 + 0.05198546, #2
            -6.05930380e-04*temp[i-1] + 1.05534966e+00 + 44.86099678*H2_Y[i-1] -0.33200033 -55.36357032*O_Y[i-1] + 0.48216838 -3.45824617*O2_Y[i-1] + 0.58558714 \
            + 20.62674247*OH_Y[i-1] -0.14553142 -6.84157537*H2O_Y[i-1] + 0.40986753 + 0.58841217*N2 -0.060958847, #3
        -7.90585206e-04*temp[i-1] + 1.37696318e+00 -6.30695438*H2_Y[i-1] + 0.04667553 + 43.21868366*O_Y[i-1] -0.37639701 -6.04974949*O2_Y[i-1] + 1.02440814 \
            -17.47618271*OH_Y[i-1] + 0.12330273 + 7.48359701*H2O_Y[i-1] -0.44832999 -0.5437266*N2 -0.001950814, #4
            -9.10584484e-04*temp[i-1] + 1.58596607e+00 -41.92593587*H2_Y[i-1] + 0.31027898 -68.66672345*O_Y[i-1] + 0.59802723 -2.30092531*O2_Y[i-1] + 0.38961723 \
            -9.90153854*OH_Y[i-1] + 0.06986003 -3.76959338*H2O_Y[i-1] + 0.22583014 + 0.07650118*N2 + 0.0337341, #5
            6.75724547e-04*temp[i-1] -1.17691024e+00 + 51.02108394*H2_Y[i-1] -0.37758895 + 2.72635681*O_Y[i-1] -0.02374419 -8.62889977*O2_Y[i-1]  + 1.46113739 \
            + 68.6684756*OH_Y[i-1] -0.48448855 -2.69159696*H2O_Y[i-1] + 0.16124915 + 0.26749286*N2 -0.012082251, #6
            -1.03667092e-03*temp[i-1] + 1.80557095e+00 -59.27618945*H2_Y[i-1] + 0.43868206 -36.24159616*O_Y[i-1] + 0.31563267 + 2.08163836*O2_Y[i-1] -0.35248522 \
            -24.32910036*OH_Y[i-1] + 0.1716533 -8.13270603*H2O_Y[i-1] + 0.48721704 + 0.5070971*N2 + 0.0036787065, #7
            -6.02778944e-04*temp[i-1] + 1.04986080e+00 + 47.03727189*H2_Y[i-1] -0.34810617 + 62.70982959*O_Y[i-1] -0.54614788 -0.67834879*O2_Y[i-1] + 0.11486526  \
            -2.32426071*OH_Y[i-1] + 0.01639876 -4.91205905*H2O_Y[i-1] + 0.29427338 + 0.40083733*N2 -0.033395156, #8
            -4.42610631e-04*temp[i-1] + 7.70895456e-01 + 20.86664949*H2_Y[i-1] -0.15442667 + 59.06049629*O_Y[i-1] -0.51436538 + 10.38121046*O2_Y[i-1] -1.75785733 \
            -7.30049542*OH_Y[i-1] + 0.05150845 + 2.89355067*H2O_Y[i-1] - 0.17334786 -0.2716199*N2 + 0.029553583, #9
            ], dtype=float))        
        O2 = (np.matmul(W2_O2, O2) + B2_O2)[0]
        #O2 = (O2 + 1.0)*O2_ptp*0.5 + O2_min
        #O2 = O2*O2_std + O2_mean
        O2_Y[i] = 0.06029811702358867*O2 + 0.16933067058119322
        #OH = (net_OH(torch.reshape(torch.Tensor(inp),(1,7))).detach().numpy()[0][0]+1.0)*OH_ptp*0.5 + OH_min
        #OH = np.matmul(W1_OH, inp) + B1_OH
        #OH = tanh(OH)
        OH = tanh(np.array([9.50982249e-04*temp[i-1] -1.65632690e+00 -79.70559256*H2_Y[i-1]+0.58987282 -11.40687418*O_Y[i-1] +0.09934392 -0.90182901*O2_Y[i-1] +0.15270731 \
            -27.01969388*OH_Y[i-1] + 0.19063671 -0.8003104*H2O_Y[i-1] + 0.04794528 -0.5201399*N2 -0.06454871, #0
            -5.67795045e-04*temp[i-1] + 9.88929297e-01 -45.71978473*H2_Y[i-1] + 0.33835591 + 2.95791685*O_Y[i-1] -0.02576087 -1.06567215*O2_Y[i-1] + 0.18045098 \
            -24.43714819*OH_Y[i-1] + 0.17241563 + 7.66260588*H2O_Y[i-1]-0.45905411 + 0.23092018*N2 -0.0039156117, #1
            -8.52161168e-04*temp[i-1] + 1.48421011e+00 -68.00448442*H2_Y[i-1] + 0.50327707 -58.01668084*O_Y[i-1] + 0.50527465 -6.42830714*O2_Y[i-1] + 1.08850956 \
            -64.96798049*OH_Y[i-1] + 0.45837981 + 6.56909975*H2O_Y[i-1] -0.39354396-0.0019529513*N2 -0.004284829, #2
            -1.03020311e-03*temp[i-1] + 1.79430596e+00 + 41.35530408*H2_Y[i-1]-0.30605594 + 45.68188885*O_Y[i-1] -0.39784938 + 8.53020801*O2_Y[i-1] -1.44442584 \
            -50.50803854*OH_Y[i-1] + 0.35635808 -8.24269211*H2O_Y[i-1] + 0.49380612 -0.39253667*N2 + 0.027126111, #3
        -7.91848872e-04*temp[i-1] + 1.37916411e+00 + 23.68727703*H2_Y[i-1] -0.17530114 -45.17818597*O_Y[i-1] + 0.39346257 + 8.6413262*O2_Y[i-1] -1.46324156 \
            + 10.92863815*OH_Y[i-1] -0.07710671 -2.36346667*H2O_Y[i-1] + 0.1415914 -0.025811803*N2 -0.035061654, #4
            -1.23941379e-04*temp[i-1] + 2.15868846e-01 + 18.21049224*H2_Y[i-1] -0.1347694 -62.7201297*O_Y[i-1] + 0.54623759 + 8.54061592*O2_Y[i-1] -1.44618822 \
            + 65.64645218*OH_Y[i-1] -0.46316674 + 5.35362409*H2O_Y[i-1] -0.32072681 -0.4789706*N2 + 0.013016694, #5
            3.49124440e-04*temp[i-1] -6.08070448e-01 + 6.13675188*H2_Y[i-1] -0.04541592 -60.85245581*O_Y[i-1] + 0.52997178 + 6.95398513*O2_Y[i-1]  -1.17752296 \
            -86.4252798*OH_Y[i-1] + 0.60977119 + 10.78630536*H2O_Y[i-1] -0.64618981 + 0.25980282*N2 -0.021370932, #6
            7.11088883e-04*temp[i-1] -1.23850434e+00 -47.04700937*H2_Y[i-1] + 0.34817823 -65.88083387*O_Y[i-1] + 0.57376456 + 4.38776836*O2_Y[i-1] -0.74298376 \
            + 13.24887972*OH_Y[i-1] -0.09347711 -3.11807986*H2O_Y[i-1] + 0.18679903 + 0.5280894*N2 -0.0030887057, #7
            1.1167465e-03*temp[i-1] -1.9450387e+00 -22.46229598*H2_Y[i-1] + 0.16623549 -53.87514588*O_Y[i-1] + 0.4692055 -4.50878044*O2_Y[i-1] + 0.76347482  \
            + 76.71350767*OH_Y[i-1] -0.54125005 -2.82488356*H2O_Y[i-1] + 0.16923413 + 0.28312764*N2 -0.004935419, #8
            -2.15884222e-05*temp[i-1] + 3.76005803e-02 + 50.72932207*H2_Y[i-1] -0.37542972 -42.36459559*O_Y[i-1] + 0.36895865 + 5.11200075*O2_Y[i-1] -0.86561852 \
            -6.74535188*OH_Y[i-1] + 0.04759165 + 5.97605555*H2O_Y[i-1] -0.35801566 -0.35819423*N2 + 0.008593773, #9
            ], dtype=float))
        OH = (np.matmul(W2_OH, OH) + B2_OH)[0]
        #OH = (OH + 1.0)*OH_ptp*0.5 + OH_min
        #OH = OH*OH_std + OH_mean
        OH_Y[i] = 0.007055472582575998*OH + 0.007055472582575998
        #H2O = (net_H2O(torch.reshape(torch.Tensor(inp),(1,7))).detach().numpy()[0][0]+1.0)*H2O_ptp*0.5 + H2O_min

        #H2O = np.matmul(W1_H2O, inp) + B1_H2O
        #H2O = tanh(H2O)
        H2O = tanh(np.array([5.04853582e-04*temp[i-1] -8.79304076e-01 -71.74341707*H2_Y[i-1]+0.53094759 -61.24513305*O_Y[i-1] +0.53339165 -3.27484347*O2_Y[i-1] +0.55453144 \
            -2.24509211*OH_Y[i-1] + 0.01584019 -7.03412077*H2O_Y[i-1] + 0.4214026 -0.55354816*N2 + 0.014537469, #0
            -3.42795106e-04*temp[i-1] + 5.97046638e-01 -27.99568807*H2_Y[i-1] + 0.20718616 -17.99311946*O_Y[i-1] + 0.15670437 + 7.2836951*O2_Y[i-1] -1.23335298 \
            + 73.72386967*OH_Y[i-1] -0.52015674 -0.0325356*H2O_Y[i-1] + 0.00194915 -0.07465306*N2 + 0.04667145, #1
            -6.92935859e-04*temp[i-1] + 1.20688720e+00 -84.21129415*H2_Y[i-1] + 0.62321792 -20.64127155*O_Y[i-1] + 0.17976746 -4.30335102*O2_Y[i-1] + 0.72868931 \
            + 9.90708148*OH_Y[i-1] -0.06989914 -5.4039427*H2O_Y[i-1] + 0.32374132 + 0.41016045*N2 + 0.059154492, #2
            -6.31294868e-04*temp[i-1] + 1.09952701e+00 -6.01786358*H2_Y[i-1] + 0.04453607 -21.67764424*O_Y[i-1] + 0.18879336 + 3.8249406*O2_Y[i-1] -0.64767976 \
            -6.16413834*OH_Y[i-1] + 0.04349091 -2.38900579*H2O_Y[i-1] + 0.14312141 -0.19104345*N2 -0.15884192, #3
            -8.94431216e-05*temp[i-1] + 1.55783190e-01 -58.44609102*H2_Y[i-1] + 0.43253879 + 23.72311142*O_Y[i-1] -0.2066076 -2.36752841*O2_Y[i-1] + 0.40089517 \
            -27.8113878*OH_Y[i-1] + 0.19622248 + 2.5261654*H2O_Y[i-1] -0.15133841 + 0.005798343*N2 + 0.22503251, #4
            6.01489090e-04*temp[i-1] -1.04761426e+00 + 30.34698909*H2_Y[i-1] -0.2245873 + 37.7051874*O_Y[i-1] -0.32837927 + 0.06647803*O2_Y[i-1] -0.01125677 \
            -84.40533724*OH_Y[i-1] + 0.59551954 -10.11922043*H2O_Y[i-1] + 0.60622585 + 0.41132563*N2 -0.077791885, #5
            -4.40898886e-04*temp[i-1] + 7.67914107e-01 + 31.83350743*H2_Y[i-1] -0.2355885 + 26.4591926*O_Y[i-1] -0.23043647 + 1.26443378*O2_Y[i-1] -0.21410742 \
            -5.36308297*OH_Y[i-1] + 0.03783908 -3.15517957*H2O_Y[i-1] + 0.18902162 -0.30188745*N2 -0.0031284257, #6
            8.83106386e-04*temp[i-1] -1.53810743e+00 -19.60818608*H2_Y[i-1] + 0.14511323 + 39.20316254*O_Y[i-1] -0.34142533 + 0.54721133*O2_Y[i-1] -0.09265966 \
            -65.20581708*OH_Y[i-1] + 0.46005785 + 2.05313668*H2O_Y[i-1] -0.12300004 -0.14567617*N2 -0.006538765, #7
            -7.73394140e-04*temp[i-1] + 1.34702148e+00 + 49.78438121*H2_Y[i-1] -0.36843655 + 3.07573701*O_Y[i-1] -0.02678698 + 2.96826191*O2_Y[i-1] -0.50261778  \
            -13.35558537*OH_Y[i-1] + 0.09422997 + 10.10642164*H2O_Y[i-1] -0.60545909 + 0.19844683*N2 + 0.0165289, #8
            6.54310881e-04*temp[i-1] -1.13961403e+00 -59.98962362*H2_Y[i-1] + 0.44396193 -5.42392805*O_Y[i-1] + 0.04723768 -2.33595995*O2_Y[i-1] + 0.39554966 \
            + 63.40308954*OH_Y[i-1] -0.44733876 + 9.07028817*H2O_Y[i-1] -0.54338604 + 0.0689505*N2 + 0.009344069, #9
            ], dtype=float))    
        H2O = (np.matmul(W2_H2O, H2O) + B2_H2O)[0]
        #H2O = (H2O + 1.0)*H2O_ptp*0.5 + H2O_min
        #H2O = H2O*H2O_std + H2O_mean
        H2O_Y[i] = 0.05990835485029466*H2O + 0.05990835485029466
        #N2 = (net_N2(torch.reshape(torch.Tensor(inp),(1,7))).detach().numpy()[0][0]+1.0)*N2_ptp*0.5 + N2_min
        #N2 = np.matmul(W1_N2, inp) + B1_N2
        #N2 = tanh(N2)
        N2 = tanh(np.array([-3.24225444e-04*temp[i-1] + 5.64703837e-01 + 43.88534636*H2_Y[i-1]-0.32477989 -29.34535599*O_Y[i-1] +0.25557244 + 3.05600481*O2_Y[i-1] -0.51747534 \
            + 69.13915179*OH_Y[i-1] -0.48780939 -4.95346954*H2O_Y[i-1] + 0.29675421 -0.4639011*N2 + 0.100900024, #0
            9.31375588e-04*temp[i-1] -1.62217796e+00 + 76.11177331*H2_Y[i-1] -0.56327624 + 35.81710149*O_Y[i-1] -0.31193569 -8.51133261*O2_Y[i-1] + 1.44122966 \
            + 0.95357915*OH_Y[i-1] -0.00672795 -9.04000547*H2O_Y[i-1] + 0.54157186 + 0.1720649*N2 -0.084695265, #1
            -5.10771176e-04*temp[i-1] + 8.89610756e-01 -40.11690605*H2_Y[i-1] + 0.296891 -52.28068846*O_Y[i-1] + 0.45531917 -4.40181246*O2_Y[i-1] + 0.74536186 \
            + 84.6978324*OH_Y[i-1] -0.59758323 -5.66838028*H2O_Y[i-1] + 0.33958334 -0.22917096*N2 + 0.08727136, #2
            4.04988982e-04*temp[i-1] -7.05369786e-01 -0.39324231*H2_Y[i-1] + 0.00291025 + 63.86232663*O_Y[i-1] -0.55618513 + 3.21980383*O2_Y[i-1] -0.54521154 \
            -43.928494*OH_Y[i-1] + 0.30993629 -0.56278407*H2O_Y[i-1] + 0.03371547 + 0.2502995*N2 + 0.10138382, #3
            -8.09868688e-05*temp[i-1] + 1.41054925e-01 -2.18886381*H2_Y[i-1] + 0.016199 + 14.9078217*O_Y[i-1] -0.12983412 + 0.13003515*O2_Y[i-1] -0.02201894 \
            + 63.66115065*OH_Y[i-1] -0.4491595 + 3.07486505*H2O_Y[i-1] -0.18421011 + 0.05510376*N2 -0.06984349, #4
            1.11893714e-03*temp[i-1] -1.94885413e+00 -71.76300169*H2_Y[i-1] + 0.53109253 + 52.91557234*O_Y[i-1] -0.46084845 + 4.64162931*O2_Y[i-1] -0.7859702 \
            + 3.30904005*OH_Y[i-1] -0.02334684 -5.37522005*H2O_Y[i-1] + 0.32202059 + 0.07682459*N2 + 0.07108728, #5
            1.83511205e-04*temp[i-1] -3.19621680e-01 + 17.6427176*H2_Y[i-1] -0.1305675 + 37.23681288*O_Y[i-1] -0.32430014 + 1.80366334*O2_Y[i-1] -0.30541552 \
            + 16.55403474*OH_Y[i-1] -0.11679654 -0.59904325*H2O_Y[i-1] + 0.0358877 -0.49833515*N2 + 0.066599086, #6
            -2.55835592e-04*temp[i-1] + 4.45589152e-01 + 81.2464852*H2_Y[i-1] -0.60127642 -40.80187347*O_Y[i-1] + 0.35534871 -5.77905094*O2_Y[i-1] + 0.97857057 \
            -23.92218334*OH_Y[i-1] + 0.16878231 + 4.13970053*H2O_Y[i-1] -0.24800265 + 0.18649413*N2 -0.015138224, #7
            -1.65026750e-04*temp[i-1] + 2.87427284e-01 + 15.20362318*H2_Y[i-1] -0.11251662 -49.19116457*O_Y[i-1] + 0.42841211 -3.93689383*O2_Y[i-1] + 0.66663687\
            -8.51225673*OH_Y[i-1] + 0.06005799 + 1.71193648*H2O_Y[i-1] -0.1025593 -0.15399148*N2 + 0.14287375, #8
            3.90203972e-05*temp[i-1] -6.79618717e-02 -2.14309257*H2_Y[i-1] + 0.01586027 -1.62944872*O_Y[i-1] + 0.01419108 + 7.71130407*O2_Y[i-1] -1.30576029 \
            -73.5588464*OH_Y[i-1] + 0.51899242 + 6.59450927*H2O_Y[i-1] -0.3950662 + 0.4328615*N2 -0.031717267, #9
            ], dtype=float))
        N2 = (np.matmul(W2_N2, N2) + B2_N2)[0]
        #N2 = (N2 + 1.0)*N2_ptp*0.5 + N2_min
        #N2 = N2*N2_std + N2_mean
        N2_Y[i] = 1.232595164407831e-32*N2 + 0.755903694556252
    
    end = time.time()
    ANN_times[j] = end - start

print("Done")

for j in range(n_runs):
    gas = ct.Solution("conaire2004.cti")

    gas.TP = gastemp, ct.one_atm
    gas.set_equivalence_ratio(phi, fuel, oxidizer)
    Time = np.zeros(nt+1, 'd')
    temp = np.zeros(nt+1, 'd')
    H_Y = np.zeros(nt+1, 'd')
    H2_Y = np.zeros(nt+1, 'd')
    O_Y = np.zeros(nt+1, 'd')
    O2_Y = np.zeros(nt+1, 'd')
    OH_Y = np.zeros(nt+1, 'd')
    H2O_Y = np.zeros(nt+1, 'd')
    N2_Y = np.zeros(nt+1, 'd')

    r = ct.IdealGasConstPressureReactor(contents=gas, name='Batch Reactor')
    sim = ct.ReactorNet([r])
    Time[0] = 0
    temp[0] = r.thermo.T
    H2_Y[0] = r.thermo.Y[gas.species_index('H2')]
    O_Y[0] = r.thermo.Y[gas.species_index('O')]
    O2_Y[0] = r.thermo.Y[gas.species_index('O2')]
    OH_Y[0] = r.thermo.Y[gas.species_index('OH')]
    H2O_Y[0] = r.thermo.Y[gas.species_index('H2O')]
    N2_Y[0] = r.thermo.Y[gas.species_index('N2')]

    time_ = 0.0
    start = time.time()
    for i in range(1, nt+1):
        time_ += dt
        sim.advance(time_)
        Time[i] = time_
        temp[i] = r.thermo.T
        H2_Y[i] = r.thermo.Y[gas.species_index('H2')]
        O_Y[i] = r.thermo.Y[gas.species_index('O')]
        O2_Y[i] = r.thermo.Y[gas.species_index('O2')]
        OH_Y[i] = r.thermo.Y[gas.species_index('OH')]
        H2O_Y[i] = r.thermo.Y[gas.species_index('H2O')]
        N2_Y[i] = r.thermo.Y[gas.species_index('N2')]
    end = time.time()
    ODE_times[j] = end - start

print("Done")


plt.plot(ODE_times, label = 'ODE Times')
plt.plot(ANN_times, label = 'ANN Times')
plt.legend()
plt.xlabel('Run #')
plt.ylabel('Time taken')
plt.show()

print(ANN_times.mean()/ODE_times.mean())

print(ANN_times.mean())
