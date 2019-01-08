from hamest import *
import numpy as np
from keras.models import Sequential
from keras.optimizers import Adam, Nadam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TerminateOnNaN

join_hists = lambda h: {k:(sum((_.get(k,[None]) for _ in h),[])) for k in h[-1].keys()}

import sys
n = 3
SPAM = 0.0
average_over = 1024
batches_worth_of_data = 1
batch_size = 512
lfact = float(sys.argv[1])
TYPE = sys.argv[2]

Delta_drive = 1.

steps_per_epoch = 32
max_epochs = 200 # 1000

namestr = '_'.join(map(str,[n, SPAM, average_over, batches_worth_of_data, batch_size, lfact, TYPE]))
model_checkpoint_filepath = 'l_mod_sp_checkpoint_'+namestr
model1_checkpoint_filepath = 'll_mod_sp_checkpoint_'+namestr

# PHYSICAL PARAMETERS

from ringham import *

nb_drives = qubitring_ndrives(n)
hs = np.stack([_.full() for _ in qubitring_all_ops(n)])

_lfactors = np.array([0.9,1,1.1])*lfact

_epsilons = [0.1]*n
_etas = [0.01]*n
_deltas = [2.]*qubitring_ndrives(n)
jitter = 0.05
s = np.random.RandomState(seed=36)
_epsilons *= s.normal(loc=1., scale=jitter, size=n)
_etas     *= s.normal(loc=1., scale=jitter, size=n)
_deltas   *= s.normal(loc=1., scale=jitter, size=qubitring_ndrives(n))

H0 = qubitring_H0(n, _epsilons, _etas).full()
Hdrives = np.stack([_.full() for _ in qubitring_Hdrives(n, _deltas)])

i = qutip.identity(2).full()
s = qutip.sigmap().full() # XXX CAREFUL HERE!!! We have flipped excited and ground state in other parts of the code. While it did not matter there, it does matter here.
k = lambda a,b,c: np.kron(a,np.kron(b,c))
Ls = [k(s,i,i),k(i,s,i),k(i,i,s)]
Lsfact = [l**0.5*r for l,r in zip(_lfactors,Ls)]

val_seed = 1+1
train_seed = 42+1+1
validation_data = next(qubitring_datagen_lindblad(H0, Hdrives, Lsfact, n=n, batch_size=512, Delta=1., seed=val_seed, average_over=float('inf')))
gen_train = qubitring_datagen_lindblad(H0, Hdrives, Lsfact, n=n, batch_size=batch_size, Delta=Delta_drive,
                              seed=train_seed, average_over=average_over, step_reset=batches_worth_of_data,
                              spam=SPAM)
data_est_std = next(qubitring_datagen_lindblad(H0, Hdrives, Lsfact, n=n, batch_size=batch_size, Delta=Delta_drive,
                                      seed=train_seed, average_over=average_over,
                                      spam=SPAM))

noise = 0.1
_lfactors_false = np.random.normal(loc=1., scale=noise, size=n)*_lfactors
_epsilons_false = np.random.normal(loc=1., scale=noise, size=n)*_epsilons
_etas_false     = np.random.normal(loc=1., scale=noise, size=n)*_etas
_deltas_false   = np.random.normal(loc=1., scale=noise, size=qubitring_ndrives(n))*_deltas
guess_w_b = qubitring_perfect_pauli_weights(n, _epsilons_false, _etas_false, _deltas_false)

if TYPE=='hamiltonian':
    model = Sequential()
    model.add(StateProbabilitiesPaulied(nqubits=n, ndrives=nb_drives, paulies=hs))
    model.compile(loss='mse', optimizer=Nadam(lr=1e-3), metrics=['mse', 'mae', 'binary_crossentropy'])
    model.set_weights(guess_w_b)

    hists = []

    callbacks = ([
        #MacKayRegularization(data_est_std, forced_factor=0.0001**(1./max_epochs)),
        CalcLogMSE(),
        ReduceLROnPlateau(monitor='log_mse', epsilon=0.02, min_lr=1e-6, patience=20),
        ModelCheckpoint(model_checkpoint_filepath, monitor='val_loss', save_best_only=True, save_weights_only=True),
        TerminateOnNaN(),
    ])
    hists.append(model.fit_generator(gen_train, verbose=2, steps_per_epoch=steps_per_epoch, epochs=max_epochs,
                                     callbacks=callbacks,
                                     max_queue_size=500, workers=1, validation_data=validation_data).history)
    jhists = join_hists(hists)

    with open('l_'+namestr,'w') as f:
        f.write(str(jhists))
    
if TYPE=='lindbladian':
    timesteps = 50
    def repeat(x, timesteps):
        x_rep = np.array([x]*timesteps)
        x_rep = np.transpose(x_rep, [1,0,2])
        return x_rep
    def repeat_gen(gen, timesteps, steps_worth_of_data):
        data = []
        for i in range(steps_worth_of_data):
            x,y = next(gen)
            data.append((repeat(x,timesteps),y))
        import itertools
        yield from itertools.cycle(data)
    validation_data = repeat(validation_data[0],timesteps), validation_data[1]
    model1 = Sequential()
    unitary = StateProbabilitiesPaulied(nqubits=n, ndrives=nb_drives, paulies=hs)
    nonunitary = StateProbabilitiesTimeDepLindbladRK4(timesteps=timesteps, baseham=unitary,lindblads=np.array(Ls),normalize=False, hermify=False)
    model1.add(nonunitary)
    model1.compile(loss='mse', optimizer=Nadam(lr=1e-3), metrics=['mse', 'mae', 'binary_crossentropy'])
    model1.set_weights([*guess_w_b,_lfactors_false])

    hists1 = []

    callbacks = ([
        CalcLogMSE(),
        ReduceLROnPlateau(monitor='log_mse', epsilon=0.02, min_lr=1e-6, patience=20),
        ModelCheckpoint(model1_checkpoint_filepath, monitor='val_loss', save_best_only=True, save_weights_only=True),
        TerminateOnNaN(),
    ])
    hists1.append(model1.fit_generator(repeat_gen(gen_train,timesteps,batches_worth_of_data),
                                       verbose=1, steps_per_epoch=steps_per_epoch, epochs=max_epochs,
                                       callbacks=callbacks,
                                       max_queue_size=500, workers=1, validation_data=validation_data).history)
    jhists1 = join_hists(hists1)

    with open('l_sp_'+namestr,'w') as f:
        f.write(str(jhists1))
