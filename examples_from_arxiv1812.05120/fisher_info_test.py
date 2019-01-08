import numpy as np
import pandas as pd
from tqdm import tqdm
from hamest import *
import keras
from keras.models import Sequential
from keras.optimizers import Adam, Nadam
from ringham import *
import pickle
import sys

n=3
batch_size = int(sys.argv[1])
LOSS_INDEX = int(sys.argv[2])

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
train_seed = 42
validation_data = next(qubitring_datagen(H0, Hdrives, n=n, batch_size=batch_size, Delta=1., seed=val_seed, average_over=float('inf')))

model = Sequential()
model.add(StateProbabilitiesPaulied(nqubits=n, ndrives=nb_drives, paulies=hs,
                                    l1_lambda=0))
model.compile(loss='mse', optimizer=Nadam(lr=5e-4), metrics=['mse', 'mae', 'binary_crossentropy'])
noise = 0.10
_epsilons_false = np.random.normal(loc=1., scale=noise, size=n)*_epsilons
_etas_false     = np.random.normal(loc=1., scale=noise, size=n)*_etas
_deltas_false   = np.random.normal(loc=1., scale=noise, size=qubitring_ndrives(n))*_deltas
w_b = qubitring_perfect_pauli_weights(n, _epsilons_false, _etas_false, _deltas_false, noise=0)
model.set_weights(w_b)


class FIStateProbabilitiesPaulied(Layer):
    '''Calculate the Fisher Information matrix for a `StateProbabilitiesPaulied` estimator.
    
    XXX There is a lot of code duplication here.
    XXX You have to say the batch size in advance, including when creating the input to this layer.
    '''
    def __init__(self, ndrives, nqubits, paulies, batch_size=512, l1_lambda=None, **kwargs):
        self.ndrives = ndrives
        self.nqubits = nqubits
        self.lenpaulies = paulies.shape[0]
        self.paulies = tf.constant(value=paulies, dtype='complex128')
        self.regularizer_k = L1variable(l1=l1_lambda) if l1_lambda else None
        self.regularizer_b = L1variable(l1=l1_lambda) if l1_lambda else None
        self.batch_size = batch_size
        super().__init__(input_shape=(self.ndrives,), **kwargs)

    def build(self, input_shape):
        w_init = 'normal'
        b_init = 'normal'
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(self.ndrives,self.lenpaulies,),
                                      initializer=w_init,
                                      trainable=False,
                                      regularizer=self.regularizer_k)
        self.bias = self.add_weight(name='bias', 
                                    shape=(self.lenpaulies,),
                                    initializer=b_init,
                                    trainable=False,
                                    regularizer=self.regularizer_b)
        super().build(input_shape)
        
    def tensorflowH(self, x):
        pauli_weights = K.bias_add(K.dot(x, self.kernel), self.bias)
        self._tensorflowH = tf.tensordot(tf.cast(pauli_weights, 'complex128'), self.paulies, axes=[[1],[0]])
        return self._tensorflowH

    def call(self, x):
        H = self.tensorflowH(x)
        e, v = tf.self_adjoint_eig(H)
        expe = tf.exp(-1j*e)
        amp = tf.einsum('is,ijs->ij', tf.conj(v)[:,0,:]*expe, v)
        P = tf.abs(amp)**2
        oneoverp = 1./P
        def jac(singlep, wrt): # this works on p of shape (2**nqubits,)
            #print(singlep.shape)
            p_list = tf.unstack(singlep,axis=0,)
            jacobian_list = [tf.gradients(p_, wrt)[0] for p_ in p_list]  # list [grad(y0, x), grad(y1, x), ...]
            stack = tf.stack(jacobian_list)
            #print(jacobian_list[0].shape, stack.shape)
            return stack
        def bodywrt(allp,wrt):
            def body(old_g, t):
                g = jac(allp[t,:],wrt)
                new_g = tuple(tf.cond(tf.equal(ti, t),
                                      lambda :g,
                                      lambda :old_g[ti])
                              for ti in range(len(old_g)))
                return new_g, t + 1
            return body
        def cond(_, t):
            return tf.less(t, tf.shape(x)[0])

        p_k = tf.while_loop(cond, bodywrt(P,self.kernel),
                            [(tf.zeros((2**self.nqubits,*self.kernel.shape), dtype='float64'),)*self.batch_size,
                             tf.constant(0)])
        p_b = tf.while_loop(cond, bodywrt(P,self.bias),
                            [(tf.zeros((2**self.nqubits,*self.bias.shape  ), dtype='float64'),)*self.batch_size,
                             tf.constant(0)])
        p_k = tf.stack(p_k[0],name='dpdk')
        p_b = tf.stack(p_b[0],name='dpdb')
        #p_b = tf.Print(p_b,[p_b[0,:,0]], summarize=15)
        #print(p_k.shape,p_b.shape,oneoverp.shape)
        I_k = tf.einsum('ipjk,iplm,ip->ijklm', p_k, p_k, oneoverp)
        I_b = tf.einsum('ipj,ipk,ip->ijk', p_b, p_b, oneoverp)
        #print(I_k.shape,I_b.shape)
        return [I_k, I_b]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], *[self.ndrives,self.lenpaulies]*2),
                (input_shape[0], *[self.lenpaulies]*2)]


i = keras.Input(batch_shape=(batch_size,nb_drives),)
hk,hb = FIStateProbabilitiesPaulied(nqubits=n, ndrives=nb_drives, paulies=hs, batch_size=batch_size)(i)
fi = keras.Model(inputs=[i],outputs=[hk,hb])
fi.compile(loss='mse', optimizer=Nadam(lr=5e-4),)

fi.set_weights(model.get_weights())


class ReLURegularizer(Regularizer):
    '''A naive soft cutoff regularizer.
    
    XXX For production use this need to be annealable.'''
    def __call__(self, x):
        regularization = K.relu(K.mean(K.abs(x)**2)-1)*20
        return regularization

    def get_config(self):
        raise NotImplementedError

class AvgFIStateProbabilitiesPaulied(Layer):
    '''Optimize the drives for maximum Fisher Information (for a `StateProbabilitiesPaulied` estimator).
    
    Looks only at the diagonals of the Fisher Information matrix.
    
    XXX There is a lot of code duplication here (and the gradient is evaluated manually, because the while_loop op is slow and painful)
    XXX You have to say the drive batch size in advance (it is not the input batch size)!
    XXX This is takes input of batch size 1 only (and does not use the input for calculations - a side effect of using Keras for something it is not meant for).
    '''
    def __init__(self, ndrives, nqubits, paulies, batch_size=512, power_reg=None, use_tanh=False, **kwargs):
        self.ndrives = ndrives
        self.nqubits = nqubits
        self.lenpaulies = paulies.shape[0]
        self.paulies = tf.constant(value=paulies, dtype='complex128')
        self.regularizer_d = ReLURegularizer() if power_reg else None
        self.batch_size = batch_size
        self.use_tanh = use_tanh
        super().__init__(input_shape=(self.ndrives,), **kwargs)

    def build(self, input_shape):
        w_init = 'normal'
        b_init = 'normal'
        d_init = 'normal'
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(self.ndrives,self.lenpaulies,),
                                      initializer=w_init,
                                      trainable=False)
        self.bias = self.add_weight(name='bias', 
                                    shape=(self.lenpaulies,),
                                    initializer=b_init,
                                    trainable=False)
        self.drives = self.add_weight(name='bias', 
                                      shape=(self.batch_size, self.ndrives),
                                      initializer=d_init,
                                      trainable=True,
                                      regularizer=self.regularizer_d)
        super().build(input_shape)
        
    def tensorflowH(self, x):
        pauli_weights = K.bias_add(K.dot(x, self.kernel), self.bias)
        self._tensorflowH = tf.tensordot(tf.cast(pauli_weights, 'complex128'), self.paulies, axes=[[1],[0]])
        return self._tensorflowH # indices [batch, hilbert, hilbert]
    
    def call(self, x):
        if self.use_tanh:
            d = K.tanh(self.drives)
        else:
            d = self.drives
        H = self.tensorflowH(d)
        def HtoP(H):
            e, v = tf.self_adjoint_eig(H)
            cv = tf.conj(v)
            expe = tf.exp(-1j*e)
            amp = tf.einsum('is,ijs->ij', cv[:,0,:]*expe, v)
            return tf.abs(amp)**2
        P = HtoP(H)
        oneoverP = 1/P
        delta = tf.constant(1e-6,dtype=tf.complex128) # XXX No, I do not want to talk about this. It haunts my dreams already.
        deltar = tf.cast(delta,tf.float64)
        P_dbias = HtoP(tf.reshape(tf.expand_dims(H,1)+tf.expand_dims(self.paulies,0)*delta,[-1,2**self.nqubits,2**self.nqubits]))
        P_dbias = tf.reshape(P_dbias, [self.batch_size,self.lenpaulies,2**self.nqubits])
        P_dbias = (P_dbias - tf.expand_dims(P,1))/deltar
        P_dkernel = HtoP(tf.reshape(
            tf.expand_dims(tf.expand_dims(H,1),1)+tf.einsum('bp,qij->bpqij',tf.cast(d,tf.complex128),self.paulies)*delta,
            [-1,2**self.nqubits,2**self.nqubits]))
        P_dkernel = tf.reshape(P_dkernel, [self.batch_size,self.ndrives,self.lenpaulies,2**self.nqubits])
        P_dkernel = (P_dkernel - tf.expand_dims(tf.expand_dims(P,1),1))/deltar
        
        I_k = tf.reshape(tf.einsum('bdei,bdei,bi->de', P_dkernel, P_dkernel, oneoverP), [-1])
        I_b = tf.einsum('bdi,bdi,bi->d', P_dbias, P_dbias, oneoverP)
        I = tf.reshape(tf.concat([I_k,I_b],0), [1,-1])/self.batch_size
        return I

    def compute_output_shape(self, input_shape):
        return (1, (self.ndrives+1)*self.lenpaulies)
    
def negmean(y_true, y_pred):
    return -tf.reduce_mean(y_pred)
def negmeangeom(y_true, y_pred):
    return -tf.exp(tf.reduce_mean(tf.log(y_pred)))
def negmeanharm(y_true, y_pred):
    return -1/tf.reduce_mean(1./y_pred)
def negmeaninf(y_true, y_pred):
    return -tf.reduce_min(y_pred)

means = [negmean, negmeangeom, negmeanharm, negmeaninf]
loss = means[LOSS_INDEX]

i = keras.Input(batch_shape=(1,1))
o = AvgFIStateProbabilitiesPaulied(nqubits=n, ndrives=nb_drives, paulies=hs, batch_size=batch_size, power_reg=True, use_tanh=False)(i)
fio = keras.Model(inputs=[i],outputs=[o])
fio.compile(loss=loss, optimizer=Nadam(lr=0.05), metrics=means)
fio.set_weights([validation_data[0]]+model.get_weights())


fake_data = np.ones((1,))
for lr in [0.01, 0.001, 0.0001, 0.00001]:
    K.set_value(fio.optimizer.lr, lr)
    fio.fit(x=fake_data,y=fake_data,batch_size=1,epochs=600,verbose=2)

fio.save_weights('optimized_drives_%s_%s.weights'%(LOSS_INDEX,batch_size))

opt_drives = fio.get_weights()[0]

ks, bs = fi.predict(validation_data[0], batch_size=batch_size)
k = np.mean(ks, axis=0)
b = np.mean(bs, axis=0)
s = k.shape
k_flat = np.reshape(k, (s[0]*s[1],s[2]*s[3]))
indices = [i*s[0]+i for i in range(s[0])]
k_onlydiag = k_flat[indices,:][:,indices]

ks, bs = fi.predict(opt_drives, batch_size=batch_size)
kn = np.mean(ks, axis=0)
bn = np.mean(bs, axis=0)
s = kn.shape
kn_flat = np.reshape(kn, (s[0]*s[1],s[2]*s[3]))
indices = [i*s[0]+i for i in range(s[0])]
kn_onlydiag = kn_flat[indices,:][:,indices]

np.savez('fi_and_drives_%s_%s.weights'%(LOSS_INDEX,batch_size),
         validation_data[0], k, b, k_flat, k_onlydiag,
         opt_drives, kn, bn, kn_flat, kn_onlydiag)

starting_w = model.get_weights()
validation_data = next(qubitring_datagen(H0, Hdrives, n=n, batch_size=512, Delta=1., seed=2, average_over=float('inf')))
model = Sequential()
model.add(StateProbabilitiesPaulied(nqubits=n, ndrives=nb_drives, paulies=hs,
                                    l1_lambda=2e-4))
model.compile(loss='mse', optimizer=Nadam(lr=5e-4), metrics=['mse', 'mae', 'binary_crossentropy'])

hists = []
hists_opt = []
avgs = [1,2,4,8,16,32,64,128]
for a in avgs:
    print(a, flush=True)
    gen_train = qubitring_datagen(H0, Hdrives, n=n, batch_size=batch_size, Delta=1.,
                                  seed=1, average_over=a, step_reset=1,
                                  spam=0)
    gen_train_opt = qubitring_datagen(H0, Hdrives, n=n, batch_size=batch_size, Delta=1.,
                                  seed=1, average_over=a, step_reset=1,
                                  spam=0, specify_drives=opt_drives)
    model.set_weights(starting_w)
    _hists = []
    for i in range(16):
        print(a,i,'opt',flush=True)
        K.set_value(model.optimizer.lr, 5e-4/3**i)
        K.set_value(model.layers[0].regularizer_k.l1,2e-4/3**i)
        K.set_value(model.layers[0].regularizer_b.l1,2e-4/3**i)
        _hists.append(model.fit_generator(gen_train, verbose=0, steps_per_epoch=100, epochs=25,
                                    validation_data=validation_data,
                                    max_queue_size=500, workers=1).history)
    model.set_weights(starting_w)
    _hists_opt = []
    for i in range(16):
        print(a,i,'rand',flush=True)
        K.set_value(model.optimizer.lr, 5e-4/2**i)
        K.set_value(model.layers[0].regularizer_k.l1,2e-4/2**i)
        K.set_value(model.layers[0].regularizer_b.l1,2e-4/2**i)
        _hists_opt.append(model.fit_generator(gen_train_opt, verbose=0, steps_per_epoch=100, epochs=25,
                                    validation_data=validation_data,
                                    max_queue_size=500, workers=1).history)
    hists.append(_hists)
    hists_opt.append(_hists_opt)

with open('hists_%s_%s.weights'%(LOSS_INDEX,batch_size),'wb') as f:
    pickle.dump([hists,hists_opt],f)
