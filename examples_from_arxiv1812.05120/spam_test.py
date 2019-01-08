from hamest import *
import numpy as np
from keras.models import Sequential
from keras.optimizers import Adam, Nadam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TerminateOnNaN

join_hists = lambda h: {k:(sum((_.get(k,[None]) for _ in h),[])) for k in h[-1].keys()}

import sys

n = int(sys.argv[1])#3
SPAM = float(sys.argv[2])#0.0
average_over = float(sys.argv[3])#1000
batches_worth_of_data = int(sys.argv[4])#10
batch_size = int(sys.argv[5])#512

Delta_drive = 1.

max_epochs = 2000
steps_per_epoch = 512

namestr = '_'.join(map(str,[n, SPAM, average_over, batches_worth_of_data, batch_size]))
model_checkpoint_filepath = 'mod_checkpoint_'+namestr
model1_checkpoint_filepath = 'mod1_checkpoint_'+namestr

# PHYSICAL PARAMETERS

from ringham import *

nb_drives = qubitring_ndrives(n)
hs = np.stack([_.full() for _ in qubitring_all_ops(n)])

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

val_seed = 1
train_seed = 42+1
validation_data = next(qubitring_datagen(H0, Hdrives, n=n, batch_size=512, Delta=1., seed=val_seed, average_over=float('inf')))

gen_train = qubitring_datagen(H0, Hdrives, n=n, batch_size=batch_size, Delta=Delta_drive,
                              seed=train_seed, average_over=average_over, step_reset=batches_worth_of_data,
                              spam=SPAM)
data_est_std = next(qubitring_datagen(H0, Hdrives, n=n, batch_size=batch_size, Delta=Delta_drive,
                                      seed=train_seed, average_over=average_over,
                                      spam=SPAM))

# MODEL 0

model = Sequential()
model.add(StateProbabilitiesPaulied(nqubits=n, ndrives=nb_drives, paulies=hs,
                                    l1_lambda=2e-3))
model.compile(loss='mse', optimizer=Nadam(lr=5e-4), metrics=['mse', 'mae', 'binary_crossentropy'])

hists = []

# MODEL 0 guesses

guess_size = 10
guess_distance = 0.1
gen_guess = qubitring_datagen(H0, Hdrives, n=n, batch_size=batch_size, Delta=Delta_drive,
                              seed=train_seed, average_over=average_over, step_reset=1)
noise = 0.30
_epsilons_false = np.random.normal(loc=1., scale=noise, size=n)*_epsilons
_etas_false     = np.random.normal(loc=1., scale=noise, size=n)*_etas
_deltas_false   = np.random.normal(loc=1., scale=noise, size=qubitring_ndrives(n))*_deltas
guess_w_b = lambda : qubitring_perfect_pauli_weights(n, _epsilons_false, _etas_false, _deltas_false, noise=guess_distance)

starts = []
best = float('inf')
for i in range(guess_size):
    print('\rbest:',best, end='', flush=True)
    w, b = guess_w_b()
    model.set_weights([w, b])
    start = model.fit_generator(gen_guess, verbose=0, steps_per_epoch=1, epochs=1, callbacks=[]).history
    starts.append(start)
    if start['mean_squared_error'][-1] < best:
        best = start['mean_squared_error'][-1]
        best_start = start
        best_ws = [w, b]
print()
model.set_weights(best_ws)
hists.append(best_start)

# MODEL 0 main

callbacks = ([
    MacKayRegularization(data_est_std, forced_factor=0.0001**(1./max_epochs)),
    CalcLogMSE(),
    ReduceLROnPlateau(monitor='log_mse', epsilon=0.02, min_lr=1e-6, patience=100),
    ModelCheckpoint(model_checkpoint_filepath, monitor='val_loss', save_best_only=True, save_weights_only=True),
    TerminateOnNaN(),
])
hists.append(model.fit_generator(gen_train, verbose=2, steps_per_epoch=steps_per_epoch, epochs=max_epochs,
                                 callbacks=callbacks,
                                 max_queue_size=500, workers=1, validation_data=validation_data).history)
jhists = join_hists(hists)

# MODEL 0 spam supression

supression_hists = []
for step in range(5):
    gen_train = qubitring_datagen(H0, Hdrives, n=n, batch_size=batch_size, Delta=Delta_drive*5**(step+1),
                                  seed=train_seed, average_over=average_over, step_reset=batches_worth_of_data,
                                  spam=SPAM)
    data_est_std = next(qubitring_datagen(H0, Hdrives, n=n, batch_size=batch_size, Delta=Delta_drive*5**(step+1),
                                          seed=train_seed, average_over=average_over,
                                          spam=SPAM))
    callbacks = ([
        MacKayRegularization(data_est_std, forced_factor=0.1**(1./max_epochs)),
        CalcLogMSE(),
        ReduceLROnPlateau(monitor='log_mse', epsilon=0.02, min_lr=1e-7*0.5**step, patience=100),
        ModelCheckpoint(model_checkpoint_filepath+'_step%d'%step, monitor='val_loss', save_best_only=True, save_weights_only=True),
        TerminateOnNaN(),
    ])
    supression_hists.append(model.fit_generator(
        gen_train, verbose=2, steps_per_epoch=steps_per_epoch, epochs=max_epochs//10,
        callbacks=callbacks,
        max_queue_size=500, workers=1, validation_data=validation_data).history)

    
    
    
with open('h_'+namestr,'w') as f:
    f.write(str((jhists,supression_hists)))
