import brian2 as b2
b2.prefs.codegen.target = 'numpy'
import numpy as np
import matplotlib.pyplot as plt
import random
from brian2 import *
#%matplotlib inline

#Global Units
ms = b2.ms
mV = b2.mV
Hz = b2.Hz
amp = b2.amp
siemens = b2.siemens
nS = b2.nS
ohm = b2.ohm
second = b2.second
volt = b2.volt

inp_neu = 25
hidden_neu = 10
output_neu = 2
duration = 15*second

#Regular spiking
c = -50*mV
vr = -60*mV
vt = -40*mV
k = 0.7
C = 100
a = 0.01/ms
b = -2./ms
d = 100*mV/ms

#Input images

'''circle_img & cross_img '''
circle_img = np.array([[0, 1, 1, 1, 0],
                      [1, 0, 0, 0, 1],
                      [1, 0, 0, 0, 1],
                      [1, 0, 0, 0, 1],
                      [0, 1, 1, 1, 0]])
cross_img = np.array([[1, 0, 0, 0, 1],
                      [0, 1, 0, 1, 0],
                      [0, 0, 1, 0, 0],
                      [0, 1, 0, 1, 0],
                      [1, 0, 0, 0, 1]])

zeros_img = np.zeros(inp_neu)
#Format input image
data = [circle_img.reshape(inp_neu)*255, cross_img.reshape(inp_neu)*255, zeros_img]
inp_data_o = data[0]
inp_data_o = circle_img.reshape(inp_neu)*600 
inp_data_x = cross_img.reshape(inp_neu)*600
inp_data_0 =np.zeros(inp_neu)



'''######### Izhikevich Equation #######'''
### This equation implements the equation from the given the reference paper
izh_mod_eqs = '''
              dv/dt = ((((k/ms/mV)*(v-vr)*(v-vt)))/C) -u/C+I/C : volt
              du/dt = (a*(b*v - u)): volt/second
              I = digit_ta_input(t,i): volt/second 
              '''

izh_mod_eqs_hidden = '''
                     dv/dt = ((((k/ms/mV)*(v-vr)*(v-vt)))/C) -u/C: volt
                     du/dt = (a*(b*v-u)) :volt/second
                     I : volt/second
                     '''
izh_mod_eqs_output = '''
                     simulate: boolean (shared)
                     dv/dt = ((((k/ms/mV)*(v-vr)*(v-vt)))/C) - u/C  + (I*simulate)/C: volt
                     du/dt = (a*(b*v - u)): volt/second
                     I = digit_ta_input((t-10*ms), i) : volt/second
                     '''

reset_iz = '''
v = c
u = u + d
'''
#STDP PARAMETERS
taupre = taupost = 20*ms
wmax = 1.0 *mV
Apre = 0.1 
Apost = -Apre * taupre / taupost * 1.05


stdp_eqs = '''
           w:volt
           dapre/dt = -apre/taupre : 1 (clock-driven)
           dapost/dt = -apost/taupost : 1(clock-driven)
           #learning : boolean (shared)
           '''

stdp_pre = '''
           v_post += w
           apre += Apre 
           w = clip(w + apost*mV, 0*mV, wmax)
           '''
stdp_post = '''
            apost += Apost 
            w = clip(w + apre*mV, 0*mV, wmax)
            '''
			
#INPUT LAYER 
izh_input = b2.NeuronGroup(inp_neu,
                           izh_mod_eqs,
                           threshold = 'v > 30*mV',
                           reset = reset_iz,
                           method = 'euler',
                           name = 'Izh')

izh_input.v = c  #
izh_input.u = d

#HIDDEN LAYER 
izh_hidden = b2.NeuronGroup(hidden_neu,
                            izh_mod_eqs_hidden,
                            threshold = 'v > 30*mV',
                            reset = reset_iz,
                            method = 'euler',
                            name = 'HiddenLayer')

izh_hidden.v = c 
izh_hidden.u = d
#OUTPUT LAYER
izh_output = b2.NeuronGroup(output_neu,
                            izh_mod_eqs_output,
                            threshold = 'v> 30*mV',
                            reset = reset_iz,
                            method = 'euler',
                            name = 'OutputLayer')
izh_circle = izh_output[0]
izh_cross = izh_output[1]
izh_output.u = d


#####   '''  INP -> Hidden '''
Syn_inp_hid = b2.Synapses(izh_input,
                          izh_hidden,
                          model = stdp_eqs,
                          on_pre = stdp_pre,
                          on_post = stdp_post,
                          method = 'euler',
                          name = 'Syn_inp_hid')
#####   '''  Hidden -> Output '''
Syn_hid_out = b2.Synapses(izh_hidden,
                          izh_output,
                          model = stdp_eqs,
                          on_pre = stdp_pre,
                          on_post = stdp_post,
                          method = 'euler', 
                          name = 'Syn_hid_out')
#MAKE CONNECTIONS
Syn_inp_hid.connect() #all-to-all
Syn_hid_out.connect() #all-to-all

Syn_inp_hid.w = 'rand()*6*mV'  #2 ass
Syn_hid_out.w = 'rand()*6*mV'

spike_mon_izh_inp = b2.SpikeMonitor(izh_input)
spike_mon_out = b2.SpikeMonitor(izh_circle)
state_mon_hid_out_w =b2.StateMonitor(Syn_hid_out, ['w'], record = True)

state_circle = b2.StateMonitor(izh_circle, ['v'], record = True)
state_cross = b2.StateMonitor(izh_cross, ['v'], record = True)


#Running
inputs = +[0.]*9 
inputs = inputs*150

 

idx = [2, 0,0, 1,1,0,1,1,1,2,1,2,0,2, 2,2,0, 2,1, 0]
for idxx in idx:
    inputs = inputs + [[0.,0.]]*50
digit_ta_input = b2.TimedArray(inputs*(volt/second),dt = 10*ms)

#Training
Syn_inp_hid.learning = True
Syn_hid_out.learning = True
b2.run(duration) #15*second

#Testing 
b2.run(10*second)  
 

weights_hid_out = np.array(state_mon_hid_out_w.w)/wmax

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(311)
plt.plot(spike_mon_izh_inp.t/second, spike_mon_izh_inp.i,  '.k', markersize = 1.0)    
ax.set_ylabel("Input Layer")
plt.tight_layout()

ax = fig.add_subplot(312)
plt.plot(state_circle.t/second, state_circle.v.T/mV, color = 'blue', lw = 0.5)
plt.plot(state_cross.t/second, state_cross.v.T/mV, color = 'green', lw = 0.5) 
ax.set_ylabel('Output Layer')
plt.tight_layout()

ax = fig.add_subplot(313)
plt.plot(state_mon_hid_out_w.t/second, weights_hid_out.T)
ax.set_ylim(0, wmax*1000)
ax.set_xlabel('Time')
ax.set_ylabel('Weight/wmax hidOut')
plt.tight_layout()
plt.show() 			
