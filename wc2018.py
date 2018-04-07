"""
The World Cup 2018 has the following format.

 A1 ----+
     RS1|----+
 B2 ----+    |
          QF1|----+
 C1 ----+    |    |
     RS2|----+    |
 D2 ----+         |
               SF1|----+
 E1 ----+         |    |
     RS5|----+    |    |
 F2 ----+    |    |    |
          QF3|----+    |
 G1 ----+    |         |
     RS6|----+         |
 H2 ----+              |
                       |
                      F|------
                       |
 B1 ----+              |
     RS3|----+         |
 A2 ----+    |         |
          QF2|----+    |
 D1 ----+    |    |    |
     RS4|----+    |    |
 C2 ----+         |    |
               SF2|----+
 F1 ----+         |
     RS7|----+    |
 E2 ----+    |    |
          QF4|----+
 H1 ----+    |
     RS8|----+
 G2 --- +

C1: winner of group C
F2: runner-up of group F
RS3: 3rd match in the Round of Sixteen
QF2: 2nd match in the QuarterFinals
SF1: 1st match in the SemiFinals
F: Final

Some convenient properties occur from this format.

In the analysis that follows the suffixes A and B are appended to the match
codes. For example, QF2A/QF2B means team A/B (top/bottom team as it appears in
the graph).

P(SF_1A, SF_2A)
   = \sum_{GA} P(SF_1A, SF_2A, GA_ABCD)
   = \sum_{GA} P(SF_1A, SF_2A | GA_ABCD) P(GA_ABCD)
   = \sum_{GA} P(SF_1A | GA_ABCD) P(SF_2A | GA_ABCD)

P(SF_1B, SF_2B) = \sum_{GA P(SF_1B | GA_EFGH) P(SF_2B | GA_EFGH) P(GA_EFGH)

P(SF_1A, SF_1B, SF_2A, SF_2B) = P(SF_1A, SF_2A) P(SF_1B, SF_2B)

P(F) = \sum_{SF} P(F | SF1A, SF1B, SF2A, SF2B) P(SF1A, SF2A) P(SF1B, SF2B)

"""

from math import exp, log
from pprint import pprint as pp
import collections
import itertools
import re


DEFAULT_RATING_SIGNED = 0.


class Dist(list):
    """
    A distribution.

    >>> Dist([('a', 80), ('b', 20)])
    [('a', 80), ('b', 20)]
    """

def cons_dist(d):
    result = collections.defaultdict(float)
    for ev, p in d:
        result[ev] += p
    return Dist(list(result.iteritems()))


def prod(ns):
    """
    Product of the passed list of numbers.

    >>> prod([])
    1
    >>> prod([15])
    15
    >>> prod([2, 5])
    10
    >>> prod([2, 25, 30])
    1500
    """
    if len(ns) == 0:
        return 1
    elif len(ns) == 1:
        return ns[0]
    else:
        return ns[0] * prod(ns[1:])


def fnd(fn, *args):
    """
    Apply function to arguments, if distribution is encountered return a
    distribution.

    >>> from operator import add
    >>> fnd(add, 5, 8)
    13
    >>> fnd(add, Dist([(8, .6), (7, .3), (4, .1)]), Dist([(9, .7), (2, .3)]))
    [(17, 0.42), (10, 0.18), (16, 0.21), (9, 0.09), (13, 0.06999999999999999), (6, 0.03)]
    >>> fnd(lambda x,y: x/(x+y), 5., Dist([(3., .8), (5., .2)]))
    [(0.625, 0.8), (0.5, 0.2)]
    >>> fnd(lambda x,y: x/(x+y), Dist([(5., 7), (4., 3)]), Dist([(3., 8), (5., 2)]))
    [(0.625, 56), (0.5, 14), (0.5714285714285714, 24), (0.4444444444444444, 6)]
    """
    args = list(args)
    for argi, arg in enumerate(args):
        if not isinstance(arg, Dist):
            args[argi] = Dist(((arg, 1.),))
    result = []
    for case in itertools.product(*args):
        result.append((
            fn(*[ci[0] for ci in case]),
            prod([ci[1] for ci in case])
        ))
    if len(result) == 1:
        assert result[0][1] == 1
        return result[0][0]
    else:
        return Dist(result)


def prob_match(t1, t2, m):
    return m[t1] / (m[t1] + m[t2])


def group(ts, m):
    """
    >>> from pprint import pprint as pp
    >>> m = {'a': 1., 'b': 1., 'c': 1., 'd': 1.}
    >>> d = group(['a', 'b', 'c', 'd'], m)
    >>> assert len(set(o[1] for o in d)) == 1
    >>> m = {'a': 5., 'b': 8., 'c': 10., 'd': 15.}
    >>> pp(group(['a', 'b', 'c', 'd'], m))
    [(('a', 'b'), 0.03189792663476874),
     (('a', 'c'), 0.03987240829346093),
     (('a', 'd'), 0.05980861244019138),
     (('b', 'a'), 0.03508771929824561),
     (('b', 'c'), 0.07017543859649122),
     (('b', 'd'), 0.10526315789473684),
     (('c', 'a'), 0.046992481203007516),
     (('c', 'b'), 0.07518796992481203),
     (('c', 'd'), 0.14097744360902253),
     (('d', 'a'), 0.08581235697940504),
     (('d', 'b'), 0.13729977116704806),
     (('d', 'c'), 0.17162471395881007)]
    """
    ms = [m[t] for t in ts]
    sum_ms = sum(ms)
    result = []
    for pair in itertools.permutations(ts, 2):
        t1, t2 = pair
        p1 = m[t1] / sum_ms
        p2 = m[t2] / (sum_ms - m[t1])
        result.append((pair, p1 * p2))
    return Dist(result)
        


def bracket(ts, m):
    """
    >>> from pprint import pprint as pp
    >>> m = {'a': 1., 'b': 1., 'c': 1., 'd': 1.}
    >>> d = bracket(['a', 'b', 'c', 'd'], m)
    >>> assert len(set(o[1] for o in d)) == 1
    >>> m = {'a': 5., 'b': 8., 'c': 10., 'd': 15.}
    >>> pp(bracket(['a', 'b'], m))
    [('a', 0.38461538461538464), ('b', 0.6153846153846154)]
    >>> pp(cons_dist(bracket([Dist([('a', .8), ('b', .2)]), 'c'], m)))
    [('a', 0.26666666666666666),
     ('c', 0.6444444444444445),
     ('b', 0.08888888888888889)]
    >>> pp(cons_dist(bracket(['a', 'b', 'c', 'd'], m)))
    [('a', 0.10897435897435898),
     ('c', 0.23931623931623933),
     ('b', 0.23782980304719437),
     ('d', 0.41387959866220736)]
    >>> pp(cons_dist(bracket(['a', 'b', 'c', 'd'] * 4, m)))
    [('a', 0.04526190228123093),
     ('c', 0.2227355710595658),
     ('b', 0.1748484237340898),
     ('d', 0.5571541029251142)]
    """
    ts = list(ts)
    for ti, t in enumerate(ts):
        if not isinstance(t, Dist):
            ts[ti] = Dist([(t, 1.)])
    if len(ts) == 1:
        result = ts[0]
    elif len(ts) == 2:
        result = []
        for (t1, p1), (t2, p2) in itertools.product(*ts):
            result += [
                (t1, p1 * p2 * prob_match(t1, t2, m)),
                (t2, p1 * p2 * prob_match(t2, t1, m)),
            ]
        result = Dist(result)
    elif len(ts) > 2:
        left, right = ts[:len(ts)/2], ts[len(ts)/2:]
        result = bracket([bracket(left, m), bracket(right, m)], m)
    return result


def combd(*ds):
    """
    Join distributions of independent events.

    >>> from pprint import pprint as pp
    >>> combd([('a1', 8), ('a2', 2)], [('b1', 6), ('b2', 4)])
    [(('a1', 'b1'), 48), (('a1', 'b2'), 32), (('a2', 'b1'), 12), (('a2', 'b2'), 8)]
    >>> pp(combd([('a1', 8), ('a2', 2)], [('b1', 6), ('b2', 4)], [('c1', 7), ('c2', 3)]))
    [(('a1', 'b1', 'c1'), 336),
     (('a1', 'b1', 'c2'), 144),
     (('a1', 'b2', 'c1'), 224),
     (('a1', 'b2', 'c2'), 96),
     (('a2', 'b1', 'c1'), 84),
     (('a2', 'b1', 'c2'), 36),
     (('a2', 'b2', 'c1'), 56),
     (('a2', 'b2', 'c2'), 24)]
    """
    result = []
    for comb in itertools.product(*ds):
        event = tuple(o[0] for o in comb)
        p = prod([o[1] for o in comb])
        result.append((event, p))
    return result


def p_ga(gs, m):
    """
    Joint probability of group assignments.

    >>> from pprint import pprint as pp
    >>> m = {'a': 1., 'b': 1., 'c': 1., 'd': 1.}
    >>> d = p_ga([['a', 'b'], ['c', 'd']], m)
    >>> assert len(set(o[1] for o in d)) == 1
    >>> m = {'a': 5., 'b': 8., 'c': 10., 'd': 15., 'e': 18., 'f': 21.}
    >>> pp(p_ga([['a', 'b', 'c'], ['d', 'e', 'f']], m))
    [((('a', 'b'), ('d', 'e')), 0.012386968908708041),
     ((('a', 'b'), ('d', 'f')), 0.014451463726826045),
     ((('a', 'b'), ('e', 'd')), 0.013419216317767043),
     ((('a', 'b'), ('e', 'f')), 0.01878690284487386),
     ((('a', 'b'), ('f', 'd')), 0.017079002586248962),
     ((('a', 'b'), ('f', 'e')), 0.020494803103498754),
     ((('a', 'c'), ('d', 'e')), 0.01548371113588505),
     ((('a', 'c'), ('d', 'f')), 0.018064329658532555),
     ((('a', 'c'), ('e', 'd')), 0.016774020397208805),
     ((('a', 'c'), ('e', 'f')), 0.023483628556092324),
     ((('a', 'c'), ('f', 'd')), 0.021348753232811202),
     ((('a', 'c'), ('f', 'e')), 0.02561850387937344),
     ((('b', 'a'), ('d', 'e')), 0.014864362690449648),
     ((('b', 'a'), ('d', 'f')), 0.01734175647219125),
     ((('b', 'a'), ('e', 'd')), 0.01610305958132045),
     ((('b', 'a'), ('e', 'f')), 0.02254428341384863),
     ((('b', 'a'), ('f', 'd')), 0.020494803103498754),
     ((('b', 'a'), ('f', 'e')), 0.0245937637241985),
     ((('b', 'c'), ('d', 'e')), 0.029728725380899296),
     ((('b', 'c'), ('d', 'f')), 0.0346835129443825),
     ((('b', 'c'), ('e', 'd')), 0.0322061191626409),
     ((('b', 'c'), ('e', 'f')), 0.04508856682769726),
     ((('b', 'c'), ('f', 'd')), 0.04098960620699751),
     ((('b', 'c'), ('f', 'e')), 0.049187527448397),
     ((('c', 'a'), ('d', 'e')), 0.021438984649686993),
     ((('c', 'a'), ('d', 'f')), 0.025012148757968155),
     ((('c', 'a'), ('e', 'd')), 0.023225566703827576),
     ((('c', 'a'), ('e', 'f')), 0.0325157933853586),
     ((('c', 'a'), ('f', 'd')), 0.02955981216850782),
     ((('c', 'a'), ('f', 'e')), 0.035471774602209384),
     ((('c', 'b'), ('d', 'e')), 0.03430237543949919),
     ((('c', 'b'), ('d', 'f')), 0.04001943801274905),
     ((('c', 'b'), ('e', 'd')), 0.03716090672612412),
     ((('c', 'b'), ('e', 'f')), 0.05202526941657376),
     ((('c', 'b'), ('f', 'd')), 0.04729569946961251),
     ((('c', 'b'), ('f', 'e')), 0.05675483936353501)]
    """
    dists = [group(g, m) for g in gs]
    return combd(*dists)


def p_sfij_given_ga(i, ga, m):
    """
    j does not make a difference here so it is not required in the arguments.

    >>> from pprint import pprint as pp
    >>> ga = 'abcd'
    >>> gb = 'efgh'
    >>> gc = 'ijkl'
    >>> gd = 'mnop'
    >>> m = dict(zip(ga + gb + gc + gd, [1. for _ in range(1, 17)]))
    >>> d = cons_dist(p_sfij_given_ga(1, [('a','b'), ('e','f'), ('i','j'), ('m','n')], m))
    >>> assert len(set(o[1] for o in d)) == 1
    >>> m = dict(zip(ga + gb + gc + gd, [float(n) for n in range(1, 17)]))
    >>> pp(cons_dist(p_sfij_given_ga(1, [('a','b'), ('e','f'), ('i','j'), ('m','n')], m)))
    [('a', 0.011387163561076604),
     ('i', 0.25155279503105593),
     ('f', 0.29068322981366457),
     ('n', 0.44637681159420284)]
    """
    if i == 1:
        return bracket([ga[0][0], ga[1][1], ga[2][0], ga[3][1]], m)
    elif i == 2:
        return bracket([ga[1][0], ga[0][1], ga[3][0], ga[2][1]], m)
	

def p_sf1j_sf2j(gs, m):
    """
    >>> from pprint import pprint as pp
    >>> ga = 'ab'
    >>> gb = 'cd'
    >>> gc = 'ef'
    >>> gd = 'gh'
    >>> m = dict(zip(ga + gb + gc + gd, [float(n) for n in range(1, 9)]))
    >>> dist = p_sf1j_sf2j([ga, gb, gc, gd], m)
    >>> assert abs(sum(o[1] for o in dist) - 1.0) < 1e-15
    >>> sorted(dist)[-1]
    (('h', 'g'), 0.08104988102520536)
    """
    result = []
    for ga, pga in p_ga(gs, m):
        pd = combd(p_sfij_given_ga(1, ga, m),
                   p_sfij_given_ga(2, ga, m))
        for sfpair, psfpair in pd:
            result.append((sfpair, psfpair * pga))
    return cons_dist(result)


def p_sf1a_sf2a_sf1b_sf2b(gs, m):
    """
    Joint probability of SF1A, SF2A, SF1B, SF2B.

    >>> from pprint import pprint as pp
    >>> ga = 'ab'
    >>> gb = 'cd'
    >>> gc = 'ef'
    >>> gd = 'gh'
    >>> ge = 'ij'
    >>> gf = 'kl'
    >>> gg = 'mn'
    >>> gh = 'op'
    >>> gs = [ga, gb, gc, gd, ge, gf, gg, gh]
    >>> m = dict(zip(itertools.chain(*gs), [float(n) for n in range(1, 17)]))
    >>> dist = sorted(p_sf1a_sf2a_sf1b_sf2b(gs, m))
    >>> assert abs(sum(o[1] for o in dist) - 1.0) < 1e-15
    >>> dist[-1]
    ((('h', 'g'), ('p', 'o')), 0.003958437863589254)
    """
    return cons_dist(combd(
                p_sf1j_sf2j(gs[:4], m),
                p_sf1j_sf2j(gs[4:], m)))


def p_f(gs, m):
    """
    Probability for the winner.


    >>> from pprint import pprint as pp
    >>> ga = 'ab'
    >>> gb = 'cd'
    >>> gc = 'ef'
    >>> gd = 'gh'
    >>> ge = 'ij'
    >>> gf = 'kl'
    >>> gg = 'mn'
    >>> gh = 'op'
    >>> gs = [ga, gb, gc, gd, ge, gf, gg, gh]
    >>> m = dict(zip(itertools.chain(*gs), [1.] * 16))
    >>> d = p_f(gs, m)
    >>> assert len(set(o[1] for o in d)) == 1
    >>> m = dict(zip(itertools.chain(*gs), [float(n) for n in range(1, 17)]))
    >>> dist = sorted(p_f(gs, m))
    >>> assert abs(sum(o[1] for o in dist) - 1.0) < 1e-15
    >>> dist[-1]
    ('p', 0.14805532756139597)
    """
    sf_dist = p_sf1a_sf2a_sf1b_sf2b(gs, m)
    result = []
    for sfpairs, psfpairs in sf_dist:
        d = bracket([
            sfpairs[0][0], sfpairs[1][0],
            sfpairs[0][1], sfpairs[1][1],
        ], m)
        for winner, pwinner in d:
            result.append((
                winner, pwinner * psfpairs
            ))
    return cons_dist(result)


def calc_ps_diffs(p_ref, p_est):
    assert set(p_ref.keys()) == set(p_est.keys())
    result = {}
    for k in p_ref:
        result[k] = p_ref[k] - p_est[k]
    return result


def run_from_file(inc):
    fo = open('wc_odds.csv', 'r')
    gi = list('abcdefgh')
    gs = [[] for i in xrange(8)]
    p_ref = {}
    current_group = None
    for line in fo:
        line = line.strip()
        if not line:
            continue
        mo = re.match(r'(.+?)([0-9.]+)', line)
        if mo:
            p_ref[mo.group(1).strip()] = 1. / float(mo.group(2).strip())
        elif line in gi:
            current_group = line
        elif current_group is not None:
            gs[gi.index(current_group)].append(line)


    m = dict(zip(itertools.chain(*gs), [exp(DEFAULT_RATING_SIGNED)] * 32))
    for repi in itertools.count():
        d = p_f(gs, m)
        p_est = dict(d)
        pp('Iteration %s' % repi)
        ps_diffs = calc_ps_diffs(p_ref, p_est)
        for k in sorted(p_est, key = lambda k: -p_est[k]):
            print k, 1./p_est[k], ps_diffs[k], m[k]
        print
        print
        for t, diff in ps_diffs.iteritems():
            m[t] = exp( log(m[t]) + diff * inc )


if __name__ == '__main__':
    run_from_file(2.)
