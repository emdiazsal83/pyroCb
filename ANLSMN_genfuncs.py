# generate LS-s style data
import numpy as onp
from statsmodels.tsa.arima_process import ArmaProcess
from latentNoise_funcs_gen import *


def sample_LSs(n):
    ran = onp.random.normal(size=n)
    noise_exp = 1
    noise_var = onp.random.uniform(size=n, low=1, high=2)
    noisetmp = (onp.sqrt(noise_var) * abs(ran)) ** (noise_exp) * onp.sign(ran)
    x_pa = noisetmp

    a_sig = onp.random.uniform(size=1, low=-2, high=2)
    bern = onp.random.binomial(size=1, n=1, p=0.5)
    b_sig = bern * onp.random.uniform(size=1, low=0.5, high=2) + (1 - bern) * onp.random.uniform(size=1, low=-2,
                                                                                                 high=-0.5)
    c_sig = onp.random.exponential(size=1, scale=1 / 4) + 1
    x_child = c_sig * (b_sig * (x_pa + a_sig)) / (1 + abs(b_sig * (x_pa + a_sig)))
    ran = onp.random.normal(size=n)
    noise_var_ch = onp.random.uniform(size=n, low=1, high=2)
    z = (0.2 * onp.sqrt(noise_var_ch) * abs(ran)) ** (noise_exp) * onp.sign(ran)
    x_child = x_child + (x_child - min(x_child)) * z
    return x_pa, x_child, z


def sample_Add2NonAdd(n, alpha):
    ran = onp.random.normal(size=n)
    noise_exp = 1
    noise_var = onp.random.uniform(size=n, low=1, high=2)
    noisetmp = (onp.sqrt(noise_var) * abs(ran)) ** (noise_exp) * onp.sign(ran)
    x_pa = noisetmp

    a_sig = onp.random.uniform(size=1, low=-2, high=2)
    bern = onp.random.binomial(size=1, n=1, p=0.5)
    b_sig = bern * onp.random.uniform(size=1, low=0.5, high=2) + (1 - bern) * onp.random.uniform(size=1, low=-2,
                                                                                                 high=-0.5)
    c_sig = onp.random.exponential(size=1, scale=1 / 4) + 1
    x_child = c_sig * (b_sig * (x_pa + a_sig)) / (1 + abs(b_sig * (x_pa + a_sig)))
    ran = onp.random.normal(size=n)
    noise_var_ch = onp.random.uniform(size=n, low=1, high=2)
    z = (0.2 * onp.sqrt(noise_var_ch) * abs(ran)) ** (noise_exp) * onp.sign(ran)

    # alpha = 0 -> beta
    beta = -onp.log(1 - onp.exp(-alpha))
    print("beta",beta)
    x_child = x_child +  z * (onp.exp(-alpha)*1 + onp.exp(-beta)*(x_child - min(x_child)))
    return x_pa, x_child, z


def sample_LS(n):
    ran = onp.random.normal(size=n)
    noise_exp = 1
    noise_var = onp.random.uniform(size=n, low=1, high=2)
    noisetmp = (onp.sqrt(noise_var) * abs(ran)) ** (noise_exp) * onp.sign(ran)
    x_pa = noisetmp

    kern_pa = rbf_kernel_matrix({'gamma': 0.5}, x_pa, x_pa)
    mu = onp.zeros(n)
    x_child = onp.random.multivariate_normal(mean=mu, cov=kern_pa, size=1)
    x_child = x_child.flatten()

    ran = onp.random.normal(size=n)
    noise_var_ch = onp.random.uniform(size=n, low=1, high=2)
    z = (0.2 * onp.sqrt(noise_var_ch) * abs(ran)) ** (noise_exp) * onp.sign(ran)
    x_child = x_child + (x_child - min(x_child)) * z
    return x_pa, x_child, z


def sample_AN(n):
    ran = onp.random.normal(size=n)
    noise_exp = 1
    noise_var = onp.random.uniform(size=n, low=1, high=2)
    noisetmp = (onp.sqrt(noise_var) * abs(ran)) ** (noise_exp) * onp.sign(ran)
    x_pa = noisetmp

    kern_pa = rbf_kernel_matrix({'gamma': 0.5}, x_pa, x_pa)
    mu = onp.zeros(n)
    x_child = onp.random.multivariate_normal(mean=mu, cov=kern_pa, size=1)
    x_child = x_child.flatten()

    ran = onp.random.normal(size=n)
    noise_var_ch = onp.random.uniform(size=n, low=1, high=2)
    z = (0.2 * onp.sqrt(noise_var_ch) * abs(ran)) ** (noise_exp) * onp.sign(ran)
    x_child = x_child + z
    return x_pa, x_child, z


def sample_ANs(n):
    ran = onp.random.normal(size=n)
    noise_exp = 1
    noise_var = onp.random.uniform(size=n, low=1, high=2)
    noisetmp = (onp.sqrt(noise_var) * abs(ran)) ** (noise_exp) * onp.sign(ran)
    x_pa = noisetmp

    a_sig = onp.random.uniform(size=1, low=-2, high=2)
    bern = onp.random.binomial(size=1, n=1, p=0.5)
    b_sig = bern * onp.random.uniform(size=1, low=0.5, high=2) + (1 - bern) * onp.random.uniform(size=1, low=-2,
                                                                                                 high=-0.5)
    c_sig = onp.random.exponential(size=1, scale=1 / 4) + 1
    x_child = c_sig * (b_sig * (x_pa + a_sig)) / (1 + abs(b_sig * (x_pa + a_sig)))
    ran = onp.random.normal(size=n)
    noise_var_ch = onp.random.uniform(size=n, low=1, high=2)
    z = (0.2 * onp.sqrt(noise_var_ch) * abs(ran)) ** (noise_exp) * onp.sign(ran)
    x_child = x_child + z
    return x_pa, x_child, z


def sample_MNU(n):
    ran = onp.random.normal(size=n)
    noise_exp = 1.0
    noise_var = onp.random.uniform(size=n, low=1.0, high=2.0)
    noisetmp = (onp.sqrt(noise_var) * abs(ran)) ** (noise_exp) * onp.sign(ran)
    x_pa = noisetmp

    a_sig = onp.random.uniform(size=1, low=-2.0, high=2.0)
    bern = onp.random.binomial(size=1, n=1, p=0.5)
    b_sig = bern * onp.random.uniform(size=1, low=0.5, high=2.0) + (1 - bern) * onp.random.uniform(size=1, low=-2.0,high=-0.5)
    c_sig = onp.random.exponential(size=1, scale=1 / 4) + 1
    x_child = c_sig * (b_sig * (x_pa + a_sig)) / (1.0 + abs(b_sig * (x_pa + a_sig)))
    ran = onp.random.normal(size=n)
    noise_var_ch = onp.random.uniform(size=n, low=1.0, high=2.0)
    z = onp.random.uniform(size=n, low=0.0, high=1.0)
    x_child = x_child * z
    return x_pa, x_child, z


def sample_LSs_ts(x_pa, z):
    a_sig = onp.random.uniform(size=1, low=-2, high=2)
    bern = onp.random.binomial(size=1, n=1, p=0.5)
    b_sig = bern * onp.random.uniform(size=1, low=0.5, high=2) + (1 - bern) * onp.random.uniform(size=1, low=-2,
                                                                                                 high=-0.5)
    c_sig = onp.random.exponential(size=1, scale=1 / 4) + 1

    x_child = c_sig * (b_sig * (x_pa + a_sig)) / (1 + abs(b_sig * (x_pa + a_sig)))
    x_child = x_child + (x_child - min(x_child)) * z
    return x_child


def sample_MNU_ts(x_pa, z):
    a_sig = onp.random.uniform(size=1, low=-2, high=2)
    bern = onp.random.binomial(size=1, n=1, p=0.5)
    b_sig = bern * onp.random.uniform(size=1, low=0.5, high=2) + (1 - bern) * onp.random.uniform(size=1, low=-2,
                                                                                                 high=-0.5)
    c_sig = onp.random.exponential(size=1, scale=1 / 4) + 1

    x_child = c_sig * (b_sig * (x_pa + a_sig)) / (1 + abs(b_sig * (x_pa + a_sig)))
    x_child = x_child * z
    return x_child


def sample_ANs_ts(x_pa, z):
    a_sig = onp.random.uniform(size=1, low=-2, high=2)
    bern = onp.random.binomial(size=1, n=1, p=0.5)
    b_sig = bern * onp.random.uniform(size=1, low=0.5, high=2) + (1 - bern) * onp.random.uniform(size=1, low=-2,
                                                                                                 high=-0.5)
    c_sig = onp.random.exponential(size=1, scale=1 / 4) + 1

    x_child = c_sig * (b_sig * (x_pa + a_sig)) / (1 + abs(b_sig * (x_pa + a_sig)))
    x_child = x_child + z
    return x_child

def sample_AN_ts(x_pa, z):


    n = x_pa.shape[0]


    kern_pa = rbf_kernel_matrix({'gamma': 0.5}, x_pa, x_pa)
    mu = onp.zeros(n)
    x_child = onp.random.multivariate_normal(mean=mu, cov=kern_pa, size=1)
    x_child = x_child.flatten()


    x_child = x_child + z
    return x_child[:,None]

def genCoefs(p):
    if p > 0:
        num_comp = onp.linspace(0,p,p+1, dtype=int)[::2]
        #print("num_comp: ", num_comp)
        num_comp = num_comp[onp.random.randint(0, num_comp.shape[0],1)]
        num_real = p - num_comp
        #print("num_real: ", num_real)
        #print("num_comp: ", num_comp)
        r_comp = onp.random.uniform(1,4, int(num_comp/2))
        roots_real = onp.random.uniform(1,4,num_real)
        theta = onp.random.uniform(0,2*onp.pi, int(num_comp/2))
        x = onp.cos(theta)*r_comp
        y = onp.sin(theta)*r_comp
        roots_comp = x+y*1j
        roots_comp = onp.hstack([roots_comp, onp.conjugate(roots_comp)])
        #print("num roots comp: ", roots_comp.shape[0])
        roots = onp.hstack([roots_real, roots_comp])
        #print("num roots: ", roots.shape[0])
        coefs = onp.poly(roots)
        coefs = coefs[::-1]/coefs[coefs.shape[0]-1]
    else:
        roots = None
        coefs = onp.array([1])
    return roots, coefs

def gen_ts(max_order, N, func):
    order_ar_x = onp.random.randint(max_order, size=1)[0]
    order_ma_x = onp.random.randint(max_order, size=1)[0]
    order_ar_z = onp.random.randint(max_order, size=1)[0]
    order_ma_z = onp.random.randint(max_order, size=1)[0]
    orders = (order_ar_x, order_ma_x, order_ar_z, order_ma_z)

    _, coefs_ar_x = genCoefs(order_ar_x)
    _, coefs_ma_x = genCoefs(order_ma_x)
    AR_x = ArmaProcess(coefs_ar_x, coefs_ma_x)
    x = AR_x.generate_sample(nsample=N)
    x = ((x - onp.mean(x)) / onp.std(x)) * 1.24

    _, coefs_ar_z = genCoefs(order_ar_x)
    _, coefs_ma_z = genCoefs(order_ma_x)
    AR_z = ArmaProcess(coefs_ar_z, coefs_ma_z)
    z = AR_z.generate_sample(nsample=N)
    z = ((z - onp.mean(z)) / onp.std(z)) * 0.24

    y = func(x, z)

    # x = norml(x)
    # z = norml(z)
    # y = norml(y)

    return x, y, z, orders


def gen_ts_slow(max_order, N, func):
    order_ar_x = onp.random.randint(max_order, size=1)[0]
    order_ma_x = onp.random.randint(max_order, size=1)[0]
    orders = (order_ar_x, order_ma_x)

    _, coefs_ar_x = genCoefs(order_ar_x)
    _, coefs_ma_x = genCoefs(order_ma_x)
    AR_x = ArmaProcess(coefs_ar_x, coefs_ma_x)
    x = AR_x.generate_sample(nsample=N)[:,None]
    x = ((x - onp.mean(x)) / onp.std(x)) * 1.24

    t = onp.linspace(1, N, N)[:, None]

    ran = onp.random.normal(size=N)
    noise_var_ch = onp.random.uniform(size=N, low=1, high=2)
    z_nois = (0.2 * onp.sqrt(noise_var_ch) * abs(ran)) ** (1) * onp.sign(ran)
    a = onp.random.uniform(1, 50, size=1)
    print("a :", a)
    z = sample_AN_ts(t / N * a, z_nois)

    z = ((z - onp.mean(z)) / onp.std(z)) * 0.24


    y = func(x, z)

    return x, y, z, orders

def gen_ts_slow_slow(N, func):

    t = onp.linspace(1, N, N)[:, None]

    ran = onp.random.normal(size=N)
    noise_var_ch = onp.random.uniform(size=N, low=1, high=2)
    x_nois = (0.2 * onp.sqrt(noise_var_ch) * abs(ran)) ** (1) * onp.sign(ran)
    a = onp.random.uniform(1, 50, size=1)
    print("a :", a)
    x = sample_AN_ts(t / N * a, x_nois)

    x = ((x - onp.mean(x)) / onp.std(x)) * 1.24

    ran = onp.random.normal(size=N)
    noise_var_ch = onp.random.uniform(size=N, low=1, high=2)
    z_nois = (0.2 * onp.sqrt(noise_var_ch) * abs(ran)) ** (1) * onp.sign(ran)
    b = onp.random.uniform(1, 50, size=1)
    print("b :", b)
    z = sample_AN_ts(t / N * b, z_nois)

    z = ((z - onp.mean(z)) / onp.std(z)) * 0.24


    y = func(x, z)

    return x, y, z


def gen_ts2(max_order, N, func):
    x, y, z, orders = gen_ts(max_order, N, func)
    X = onp.vstack([x, y, z]).T

    return [X.tolist(), orders]


def gen_ts_slow2(max_order, N, func):
    x, y, z, orders = gen_ts_slow(max_order, N, func)
    X = onp.hstack([x, y, z])

    return [X.tolist(), orders]

def gen_ts_slow_slow2(N, func):
    x, y, z  = gen_ts_slow_slow(N, func)
    X = onp.hstack([x, y, z])

    return [X.tolist(), None]