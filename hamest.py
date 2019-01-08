from keras import backend as K
from keras.engine.topology import Layer
from keras.models import Sequential
from keras.optimizers import Nadam, Adam
from keras.callbacks import Callback
from keras.regularizers import Regularizer
K.set_floatx('float64')

import tensorflow as tf
import numpy as np
import scipy

# XXX There is code repetition between
# the StateProbabilities* classes!

class L1variable(Regularizer):
    '''A regulizer that permits annealing (See `MacKayRegularization`).'''
    def __init__(self, l1=0.):
        self.l1 = K.variable(l1)
        self.l1_val = l1

    def __call__(self, x):
        regularization = 0.
        if self.l1_val:
            regularization += self.l1*K.sum(K.abs(x))
        return regularization

    def get_config(self):
        raise NotImplementedError

class CalcLogMSE(Callback):
    '''Add `log(mse)` to the history of the optimization.'''
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['log_mse'] = np.log(logs.get('mean_squared_error'))

class MacKayRegularization(Callback):
    '''A clumsy naive version of MacKay regularization. Works only on `StateProbabilitiesPaulied` and `StateProbabilities`.'''
    def __init__(self, data_est_std, forced_factor=0.001**(1./1000), nested=False):
        super(MacKayRegularization, self).__init__()
        self.data_est_std = data_est_std
        self.forced_factor = forced_factor
    def on_epoch_begin(self, epoch, logs=None):
        w_k, w_b = self.model.get_weights()
        s_k = np.std(w_k)
        s_b = np.std(w_b)
        s_e = np.std(self.model.predict(self.data_est_std[0])-self.data_est_std[1])
        l1_k = (s_e/s_k)**2/np.size(self.data_est_std[1]) 
        l1_b = (s_e/s_b)**2/np.size(self.data_est_std[1])
        K.set_value(self.model.layers[0].regularizer_k.l1,
                    K.cast_to_floatx(
                        min(
                            l1_k,
                            K.get_value(self.model.layers[0].regularizer_k.l1)*self.forced_factor)))
        K.set_value(self.model.layers[0].regularizer_b.l1,
                    K.cast_to_floatx(
                        min(
                            l1_b,
                            K.get_value(self.model.layers[0].regularizer_b.l1)*self.forced_factor)))
        logs['std_k'] = s_k
        logs['std_b'] = s_b
        logs['std_e'] = s_e
        logs['w_l1_k_mackay'] = l1_k
        logs['w_l1_b_mackay'] = l1_b
        logs['w_l1_k'] = float(K.get_value(self.model.layers[0].regularizer_k.l1))
        logs['w_l1_b'] = float(K.get_value(self.model.layers[0].regularizer_b.l1))
        self.logs = logs.copy()
    def on_epoch_end(self, epoch, logs=None):
        logs.update(self.logs)

class StateProbabilitiesPaulied(Layer):
    '''Calculate state probabilities given input drives with predefined Hamiltonian components.
    
    `Hij = Skij (Akl dl + Bl)` where `Sk` is a list of
    given hamiltonian components (usually Pauli matrices);
    `Akl` and `Bl` are trainable; `dl` is the input vector
    of drives. Output is the component-wise squared absolute
    value of the state vector. Initial state is the vacuum.
    
    Considered only constant pulses of one time unit duration.
    '''
    def __init__(self, ndrives, nqubits, paulies, l1_lambda=None, **kwargs):
        self.ndrives = ndrives
        self.nqubits = nqubits
        self.lenpaulies = paulies.shape[0]
        self.paulies = tf.constant(value=paulies, dtype='complex128')
        self.regularizer_k = L1variable(l1=l1_lambda) if l1_lambda else None
        self.regularizer_b = L1variable(l1=l1_lambda) if l1_lambda else None
        super().__init__(input_shape=(self.ndrives,), **kwargs)

    def build(self, input_shape):
        w_init = 'normal'
        b_init = 'normal'
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.lenpaulies),
                                      initializer=w_init,
                                      trainable=True,
                                      regularizer=self.regularizer_k)
        self.bias = self.add_weight(name='bias', 
                                    shape=(self.lenpaulies,),
                                    initializer=b_init,
                                    trainable=True,
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
        return tf.abs(amp)**2

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 2**self.nqubits)
    
class StateProbabilities(Layer):
    '''Calculate state probabilities given input drives with arbitrary Hamiltonian (linear in drives).
    
    `Hij = (Mij+Mji)+i(Mij-Mji) where Mij=(Aijl dl + Bij)`;
    `Aijl` and `Bij` are trainable; `dl` is the input vector
    of drives. Output is the component-wise squared absolute
    value of the state vector. Initial state is the vacuum.
    
    Considered only constant pulses of one time unit duration.
    '''
    def __init__(self, ndrives, nqubits, l1_lambda=None, **kwargs):
        self.ndrives = ndrives
        self.nqubits = nqubits
        self.regularizer_k = L1variable(l1=l1_lambda) if l1_lambda else None
        self.regularizer_b = L1variable(l1=l1_lambda) if l1_lambda else None
        super().__init__(input_shape=(self.ndrives,), **kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], (2**self.nqubits)**2),
                                      initializer='normal',
                                      trainable=True,
                                      regularizer=self.regularizer_k)
        self.bias = self.add_weight(name='bias', 
                                    shape=((2**self.nqubits)**2,),
                                    initializer='normal',
                                    trainable=True,
                                    regularizer=self.regularizer_b)
        super().build(input_shape)
        
    def tensorflowH(self, x):
        preH = tf.reshape(K.bias_add(K.dot(x, self.kernel), self.bias),
                          [-1, 2**self.nqubits, 2**self.nqubits])
        preHt = tf.transpose(preH, perm=(0,2,1))
        sym = preH+preHt
        ant = preH-preHt
        self._tensorflowH = tf.complex(sym, ant)
        return self._tensorflowH

    def call(self, x):
        H = self.tensorflowH(x)
        e, v = tf.self_adjoint_eig(H)
        expe = tf.exp(-1j*e)
        amp = tf.einsum('is,ijs->ij', tf.conj(v)[:,0,:]*expe, v)
        return tf.abs(amp)**2

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 2**self.nqubits)

def pauli_to_arb_weights(w,b,hs):
    '''Transform the weights of `StateProbabilitiesPaulied` to `StateProbabilities` weights.'''
    b_ = np.tensordot(b,hs,[0,0])
    b_ = (np.real(b_)+np.imag(b_))/2
    b_ = b_.flatten()
    w_ = np.tensordot(w,hs,[1,0])
    w_ = (np.real(w_)+np.imag(w_))/2
    w_.shape = w_.shape[0], w_.shape[1]*w_.shape[2]
    return w_,b_

class StateProbabilitiesTimeDep(Layer):
    '''Calculate state probabilities given input time-dependent drives.
    
    The evaluation is done with repeated application of
    the time-evolution operator `dU=exp(-iHdt)`. `exp` is
    calculated either with Taylor expansion (Horner's form)
    or with eigen decomposition as specified by `taylorord`.
    `timesteps` specifies the timesteps of duration
    `1/timesteps` time units. `normalize` specifies whether
    the state vector is normalized after each time step.
    `baseham` is either `StateProbabilitiesPaulied` or
    `StateProbabilities`.
    '''
    def __init__(self, timesteps, baseham, normalize=False, taylorord='eig', **kwargs):
        self.timesteps = timesteps
        self.baseham = baseham
        self.ndrives = baseham.ndrives
        self.nqubits = baseham.nqubits
        self.normalize = normalize
        self.taylorord = taylorord
        super().__init__(input_shape=(self.timesteps,self.ndrives), **kwargs)

    def build(self, input_shape):
        self.baseham.build((*input_shape[:-2],input_shape[-1]))
        super().build(input_shape)

    def call(self, x):
        x = tf.transpose(x, [1,0,2]) # [time, batch, drives]
        def prop(prev_out, curr_in):
            H = self.baseham.tensorflowH(curr_in) # [batch, i, j]
            if self.taylorord == 'eig':
                e, v = tf.self_adjoint_eig(H)
                expe = tf.exp(-1j/self.timesteps*e)
                next_out = tf.einsum('bij,bj->bi',
                                     v,
                                     expe*tf.einsum('bji,bj->bi', tf.conj(v), prev_out))
                return next_out
            next_out = prev_out
            for i in range(self.taylorord,0,-1):
                next_out = prev_out+(-1j/self.timesteps/i)*tf.einsum('bij,bj->bi',H,next_out)
            if self.normalize:
                next_out = next_out/tf.norm(next_out, 2, axis=-1, keep_dims=True)
            return next_out
        init = tf.zeros((tf.shape(x)[1],2**self.nqubits), dtype=tf.complex128)
        init += tf.one_hot(0, 2**self.nqubits, dtype=tf.complex128)
        amp = tf.scan(prop, x, initializer=init)[-1]
        return tf.abs(amp)**2
    
    @property
    def trainable_weights(self):
        if not self.trainable:
            return []
        if isinstance(self.baseham, Layer):
            return self.baseham.trainable_weights
        return []

    @property
    def non_trainable_weights(self):
        if isinstance(self.baseham, Layer):
            if not self.trainable:
                return self.baseham.weights
            return self.baseham.non_trainable_weights
        return []
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], 2**self.nqubits)
    
class StateProbabilitiesTimeDepLindblad(Layer):
    '''Calculate state probabilities given input time-dependent drives under Lindblad equation.
    
    The evaluation is done with Euler's method.
    `timesteps` specifies the timesteps of duration
    `1/timesteps` time units. `normalize` specifies whether
    the state matrix is normalized after each time step.
    `hermify` specifies whether the state matrix is projected
    onto the space of Hermitian matrices after each step.
    `hermify` is executed before `normalize`.
    `baseham` is either `StateProbabilitiesPaulied` or
    `StateProbabilities`. `lindblads` is the list of Lindblad operators.
    `reg` is the regulizer for the Lindblad weights.
    '''
    def __init__(self, timesteps, baseham, lindblads=None, l1_lambda=None, normalize=False, hermify=False, **kwargs):
        self.timesteps = timesteps
        self.baseham = baseham
        if lindblads is not None: # TODO sparsity
            self.lenlindblads = lindblads.shape[0]
            lindbladsdag = np.transpose(np.conj(lindblads),[0,2,1])
            lindbladssq = np.einsum('bij,bjk->bik', lindbladsdag, lindblads)
            self.lindblads = tf.constant(value=lindblads, dtype='complex128')
            self.lindbladsdag = tf.constant(value=lindbladsdag, dtype='complex128')
            self.lindbladssq = tf.constant(value=lindbladssq, dtype='complex128')
        else:
            self.lindblads = None
        self.regularizer = L1variable(l1=l1_lambda) if l1_lambda else None
        self.ndrives = baseham.ndrives
        self.nqubits = baseham.nqubits
        self.normalize = normalize
        self.hermify = hermify
        super().__init__(input_shape=(self.timesteps,self.ndrives), **kwargs)

    def build(self, input_shape):
        self.baseham.build((*input_shape[:-2],input_shape[-1]))
        if self.lindblads is not None:
            init = 'zeros'
            self.lindbladweight = self.add_weight(name='lindbladweight', 
                                                  shape=(self.lenlindblads,),
                                                  initializer=init,
                                                  trainable=True,
                                                  regularizer=self.regularizer)
        super().build(input_shape)
        
    def call(self, x):
        x = tf.transpose(x, [1,0,2]) # [time, batch, drives]
        def propH(prev_out, curr_in):
            H = self.baseham.tensorflowH(curr_in) # [batch, i, j]
            hr = tf.einsum('bij,bjk->bik', H, prev_out)
            rh = tf.einsum('bij,bjk->bik', prev_out, H)
            unitary = -1j*(hr-rh)
            return unitary
        def propL(prev_out, curr_in):
            lrl = tf.einsum('tij,bjk,tkl->btil', self.lindblads, prev_out, self.lindbladsdag)
            llr = tf.einsum('tij,bjk->btik', self.lindbladssq, prev_out)
            rll = tf.einsum('bij,tjk->btik', prev_out, self.lindbladssq)
            nonunitary = tf.einsum('btij,t->bij', lrl-(llr+rll)/2, tf.cast(self.lindbladweight, 'complex128'))
            return nonunitary
        def prop(prev_out, curr_in):
            p = propH(prev_out, curr_in)
            if self.lindblads is not None:
                p += propL(prev_out, curr_in)
            res = prev_out + p/self.timesteps
            if self.hermify:
                res = (res+tf.transpose(tf.conj(res),perm=[0,2,1]))/2
            if self.normalize:
                res = res/tf.trace(res)
            return res
        init = tf.zeros((tf.shape(x)[1],4**self.nqubits), dtype=tf.complex128)
        init += tf.one_hot(0, 4**self.nqubits, dtype=tf.complex128)
        init = tf.reshape(init, (tf.shape(x)[1],2**self.nqubits,2**self.nqubits))
        rho = tf.scan(prop, x, initializer=init)[-1]
        return tf.abs(tf.matrix_diag_part(rho))
    
    @property
    def trainable_weights(self):
        if not self.trainable:
            return []
        if isinstance(self.baseham, Layer):
            if self.lindblads is not None:
                return self.baseham.trainable_weights+[self.lindbladweight]
            return self.baseham.trainable_weights
        return []

    @property
    def non_trainable_weights(self):
        if isinstance(self.baseham, Layer):
            if not self.trainable:
                if self.lindblads is not None:
                    return self.baseham.weights+[self.lindbladweight]
                return self.baseham.weights
            return self.baseham.non_trainable_weights
        return []
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], 2**self.nqubits)
    
class StateProbabilitiesTimeDepLindbladRK4(Layer):
    '''Calculate state probabilities given input time-dependent drives under Lindblad equation using RK4.
    
    The evaluation is done with the RK4 method.
    `timesteps` specifies the timesteps of duration
    `1/timesteps` time units. `normalize` specifies whether
    the state matrix is normalized after each time step.
    `hermify` specifies whether the state matrix is projected
    onto the space of Hermitian matrices after each step.
    `hermify` is executed before `normalize`.
    `baseham` is either `StateProbabilitiesPaulied` or
    `StateProbabilities`. `lindblads` is the list of Lindblad operators.
    `l1_lambda` is the regulizer for the Lindblad weights.
    '''
    def __init__(self, timesteps, baseham, lindblads=None, l1_lambda=None, normalize=False, hermify=False, **kwargs):
        self.timesteps = timesteps
        self.baseham = baseham
        if lindblads is not None: # TODO sparsity
            self.lenlindblads = lindblads.shape[0]
            lindbladsdag = np.transpose(np.conj(lindblads),[0,2,1])
            lindbladssq = np.einsum('bij,bjk->bik', lindbladsdag, lindblads)
            self.lindblads = tf.constant(value=lindblads, dtype='complex128')
            self.lindbladsdag = tf.constant(value=lindbladsdag, dtype='complex128')
            self.lindbladssq = tf.constant(value=lindbladssq, dtype='complex128')
        else:
            self.lindblads = None
        self.regularizer = L1variable(l1=l1_lambda) if l1_lambda else None
        self.ndrives = baseham.ndrives
        self.nqubits = baseham.nqubits
        self.normalize = normalize
        self.hermify = hermify
        super().__init__(input_shape=(self.timesteps,self.ndrives), **kwargs)

    def build(self, input_shape):
        self.baseham.build((*input_shape[:-2],input_shape[-1]))
        if self.lindblads is not None:
            init = 'zeros'
            self.lindbladweight = self.add_weight(name='lindbladweight', 
                                                  shape=(self.lenlindblads,),
                                                  initializer=init,
                                                  trainable=True,
                                                  regularizer=self.regularizer)
        super().build(input_shape)
        
    def call(self, x):
        x = tf.transpose(x, [1,0,2]) # [time, batch, drives]
        def propH(prev_out, curr_in):
            H = self.baseham.tensorflowH(curr_in) # [batch, i, j]
            hr = tf.einsum('bij,bjk->bik', H, prev_out)
            rh = tf.einsum('bij,bjk->bik', prev_out, H)
            unitary = -1j*(hr-rh)
            return unitary
        def propL(prev_out, curr_in):
            lrl = tf.einsum('tij,bjk,tkl->btil', self.lindblads, prev_out, self.lindbladsdag)
            llr = tf.einsum('tij,bjk->btik', self.lindbladssq, prev_out)
            rll = tf.einsum('bij,tjk->btik', prev_out, self.lindbladssq)
            nonunitary = tf.einsum('btij,t->bij', lrl-(llr+rll)/2, tf.cast(self.lindbladweight, 'complex128'))
            return nonunitary
        def prop(prev_out, curr_in):
            p = propH(prev_out, curr_in)
            if self.lindblads is not None:
                p += propL(prev_out, curr_in)
            return p/self.timesteps
        def propRK4(prev_out, curr_in):
            k1 = prop(prev_out, curr_in)
            k2 = prop(prev_out+k1/2, curr_in)
            k3 = prop(prev_out+k2/2, curr_in)
            k4 = prop(prev_out+k3, curr_in)
            res = prev_out + (k1+2*k2+2*k3+k4)/6
            if self.hermify:
                res = (res+tf.transpose(tf.conj(res),perm=[0,2,1]))/2
            if self.normalize:
                res = res/tf.trace(res)
            return res
        init = tf.zeros((tf.shape(x)[1],4**self.nqubits), dtype=tf.complex128)
        init += tf.one_hot(0, 4**self.nqubits, dtype=tf.complex128)
        init = tf.reshape(init, (tf.shape(x)[1],2**self.nqubits,2**self.nqubits))
        rho = tf.scan(propRK4, x, initializer=init)[-1]
        return tf.abs(tf.matrix_diag_part(rho))
    
    @property
    def trainable_weights(self):
        if not self.trainable:
            return []
        if isinstance(self.baseham, Layer):
            if self.lindblads is not None:
                return self.baseham.trainable_weights+[self.lindbladweight]
            return self.baseham.trainable_weights
        return []

    @property
    def non_trainable_weights(self):
        if isinstance(self.baseham, Layer):
            if not self.trainable:
                if self.lindblads is not None:
                    return self.baseham.weights+[self.lindbladweight]
                return self.baseham.weights
            return self.baseham.non_trainable_weights
        return []
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], 2**self.nqubits)