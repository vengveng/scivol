"""Independent MP recursion for shape-only density controls; explicit seed."""
import mpmath as mp
import numpy as np

def variance_path(y,theta,p,q,family,h0):
    with mp.workdps(120):
        v=[mp.mpf(float(a)) for a in theta]
        errors=[mp.mpf(float(e)) for e in y]
        hs=[mp.mpf(float(h0))]
        delta=v[1+2*p+q] if family=="aparch" else mp.mpf(2)
        powers=[hs[0]**(delta/2)]
        beta=1+(1 if family=="garch" else 2)*p
        for t in range(1,len(y)):
            value=v[0]
            for j in range(min(t,p)):
                e=errors[t-1-j]
                if family=="aparch": value+=v[1+j]*(abs(e)-v[1+p+j]*e)**delta
                else: value+=(v[1+j]+(v[1+p+j] if family!="garch" and e<0 else 0))*e*e
            value+=mp.fsum(v[beta+j]*powers[t-1-j] for j in range(min(t,q)))
            powers.append(value)
            hs.append(value**(2/delta))
        return np.array(list(map(float,hs)))
