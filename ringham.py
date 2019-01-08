import itertools
import functools
import operator
import qutip
import numpy as np

T = qutip.tensor
S = lambda *args: functools.reduce(operator.add, args)
plusHC = lambda arg: arg+qutip.dag(arg)
one2n = qutip.gate_expand_1toN
two2n = qutip.gate_expand_2toN
sz = qutip.sigmaz()
sx = qutip.sigmax()
sy = qutip.sigmay()
sm = qutip.sigmam()
sp = qutip.sigmap()

def Hsingles(n):
    '''A list of single-qubit Pauli ops.'''
    szs = [one2n(sz, n, i) for i in range(n)]
    sxs = [one2n(sx, n, i) for i in range(n)]
    sys = [one2n(sy, n, i) for i in range(n)]
    return szs+sxs+sys

def qubitring_Hneighbors(n):
    '''A list of neighbor-pair interaction ops.'''
    return [two2n(plusHC(T(sm,sp)), n, targets=(i,(i+1)%n)) for i in range(n if n>2 else 1)]

def qubitring_all_ops(n):
    '''A list of single-qubit Pauli ops and neighbor-pair interaction ops.'''
    return Hsingles(n)+qubitring_Hneighbors(n)

def qubitring_H0(n, epsilons, etas):
    '''The sum `epsilon[i] sz[i] + eta[j] (s+[j] s-[j+1] + hc)`.'''
    assert (n!=2 and n==len(epsilons)==len(etas)) or (n==2==len(epsilons)==len(etas)+1)
    Hsingle = S(*(one2n(sz*e, n, i) for i,e in enumerate(epsilons)))
    Hneighbors = S(*(two2n(plusHC(T(sm,sp))*e, n, targets=(i,(i+1)%n)) for i,e in enumerate(etas)))
    return Hsingle+Hneighbors

def qubitring_Hdrives(n, deltas):
    '''A list of `delta[i] sx/y/z[i]` and `delta[j] (s+[j] s-[j+1] + hc)`.
    
    Importantly, its order is the same as the order of qubitring_all_ops or qubitring_H0.'''
    assert len(deltas)==qubitring_ndrives(n)
    Hxs = [one2n(sx, n, i) for i in range(n)]
    Hys = [one2n(sy, n, i) for i in range(n)]
    Hzs = [one2n(sz, n, i) for i in range(n)]
    Hneighbors = [two2n(plusHC(T(sm,sp)), n, targets=(i,(i+1)%n)) for i in range(n)]
    if n==2: Hneighbors = Hneighbors[:1]
    return [d*h for d,h in zip(deltas,Hzs+Hxs+Hys+Hneighbors)]

def qubitring_perfect_pauli_weights(n, epsilons, etas, deltas, noise=0):
    '''Turn epsilons, etas, deltas into the weights that can be put in one of the hamest estimators.'''
    assert (n!=2 and n==len(epsilons)==len(etas)) or (n==2==len(epsilons)==len(etas)+1)
    ndrives = qubitring_ndrives(n)
    nallops = len(qubitring_all_ops(n)) # TODO this is a bit silly (creating the list just to count its elements)
    assert len(deltas)==ndrives
    w = np.zeros((nallops, ndrives))
    b = np.zeros((nallops, ))
    w[:,:][np.diag_indices(4*n-1 if n==2 else 4*n)] = deltas
    b[0:n] = epsilons
    b[-1 if n==2 else -n:] = etas
    if noise:
        w *= np.random.normal(loc=1, scale=noise, size=w.shape)
        b *= np.random.normal(loc=1, scale=noise, size=b.shape)
    return w.T, b

def qubitring_ndrives(n):
    return n*4-1 if n==2 else n*4

def qubitring_datagen(H0, Hdrives,
                      batch_size=512, average_over=float('inf'),
                      n=2,
                      Delta=1., step_reset=1,
                      seed=42,
                      spam=None,
                      specify_drives=None
                     ):
    '''Generate random test data (unitary dynamics).'''
    if spam:
        b2d = lambda l:int(''.join(map(str,l)),2)
        bad_preps = np.array([2**_ for _ in range(0,n)],dtype=int)
        spam_stoch_mat = np.eye(2**n)*(1-n*spam)
        for bits in itertools.product([0,1], repeat=n):
            bits = list(bits)
            for i in range(len(bits)):
                target_bits = bits.copy()
                target_bits[i] = (not target_bits[i])+0
                spam_stoch_mat[b2d(bits),b2d(target_bits)] = spam
    state = np.random.RandomState(seed=seed)
    mem = []
    for step in range(step_reset):
        if specify_drives is not None:
            drives = specify_drives
        else:
            drives = state.normal(loc=0,scale=Delta,size=batch_size*Hdrives.shape[0])
        drives.shape = (batch_size,Hdrives.shape[0],1,1)
        H = H0+np.sum(drives*Hdrives,axis=1)
        w,v = np.linalg.eigh(H)
        exps = np.exp(-1j*w)
        vconj = v.conj()
        amp = np.einsum('is,ijs->ij', vconj[:,0,:]*exps, v)
        probs = np.abs(amp)**2
        if spam: # TODO this striding is horribly ugly and probably inefficient
            exps.shape = (exps.shape[0], 1, exps.shape[1])
            amps = np.einsum('iks,ijs->ikj', vconj[:,bad_preps,:]*exps, v)
            probs = (1-n*spam)*probs + spam*(np.sum(np.abs(amps)**2, axis=1))
        # XXX very slow correctness check (for no spam)
        #for i in range(batch_size):
        #    assert np.sum(np.abs(amp[i,:]-scipy.linalg.expm(-1j*H[i,...])[:,0])) < 1e-10
        drives.shape = (batch_size,Hdrives.shape[0])
        if spam:
            probs = np.dot(probs,spam_stoch_mat)
        if average_over == float('inf'):
            mem.append((drives, probs))
        else:
            multinomial_samples = np.stack((state.multinomial(average_over, p) for p in probs))
            mem.append((drives, multinomial_samples/average_over))
        yield mem[-1]
    yield from itertools.cycle(mem)
    
def qubitring_datagen_lindblad(H0, Hdrives, Ls,
                      batch_size=512, average_over=float('inf'),
                      n=2,
                      Delta=1., step_reset=1,
                      seed=42,
                      spam=None,
                      specify_drives=None,
                      options=None
                     ):
    '''Generate random test data (non-unitary dynamics).'''
    Ls = [qutip.Qobj(_) for _ in Ls]
    ground_state = np.zeros(H0.shape[0])
    ground_state[0] = 1
    if spam:
        b2d = lambda l:int(''.join(map(str,l)),2)
        bad_preps = np.array([2**_ for _ in range(0,n)],dtype=int)
        spam_stoch_mat = np.eye(2**n)*(1-n*spam)
        for bits in itertools.product([0,1], repeat=n):
            bits = list(bits)
            for i in range(len(bits)):
                target_bits = bits.copy()
                target_bits[i] = (not target_bits[i])+0
                spam_stoch_mat[b2d(bits),b2d(target_bits)] = spam
    state = np.random.RandomState(seed=seed)
    mem = []
    for step in range(step_reset):
        if specify_drives is not None:
            drives = specify_drives
        else:
            drives = state.normal(loc=0,scale=Delta,size=batch_size*Hdrives.shape[0])
        drives.shape = (batch_size,Hdrives.shape[0],1,1)
        H = H0+np.sum(drives*Hdrives,axis=1)
        probs = []
        for h in H:
            h = qutip.Qobj(h)
            r = qutip.mesolve(h,qutip.Qobj(ground_state),np.linspace(0,1,10),c_ops=Ls,options=options)
            probs.append(np.abs(r.states[-1].full().diagonal()))
        probs = np.array(probs)
        if spam:
            raise NotImplementedError
            probs = (1-n*spam)*probs + ...
        drives.shape = (batch_size,Hdrives.shape[0])
        if spam:
            probs = np.dot(probs,spam_stoch_mat)
        if average_over == float('inf'):
            mem.append((drives, probs))
        else:
            multinomial_samples = np.stack((state.multinomial(average_over, p) for p in probs))
            mem.append((drives, multinomial_samples/average_over))
        yield mem[-1]
    yield from itertools.cycle(mem)