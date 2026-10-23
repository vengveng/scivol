"""Historical G/FG rounded-score control; not the old Python public API."""
import numpy as np
from .transport import JointSample

def test_historical_G_FG_rounding_identity():
    rng=np.random.default_rng(214)
    raw=rng.normal(size=96)
    external=rng.normal(size=(96,2))
    data=raw[5:]*2
    lag=np.r_[0.,data[:-1]]
    features=np.column_stack((np.ones(len(data)),lag,external[5:]))
    model=JointSample(data,features,3,2,0)
    theta=np.array([.5039229709333392,.005126124767175115,.00019356986989024254,
        -.0021151791473364213,.27542684708262793,.024,.028181826334211017,
        .024000000000000004,-.00012850871681422942,-.0011746371095916908,
        .001176778705595566,.4300874033706335,.4308056202490993,1.9974353706675203])
    np.testing.assert_array_equal(model.evaluate(theta,2)[1],model.evaluate(theta,3)[1])
