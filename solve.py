import math
import re
import sys
from dataclasses import dataclass
from functools import partial
from itertools import count

from joblib import Memory

memory = Memory('.joblib')

from z3 import And, Int, Or, Solver, Sum, sat

Pos = tuple[int, int]


def add(a: Pos, b: Pos) -> Pos:
    return (a[0] + b[0]) % ROWS, (a[1] + b[1]) % COLS


@dataclass
class Sequence:
    letter: str
    coords: tuple[Pos, ...]


def make_rule(line):
    letter = line[0]
    rule = line[7:]
    last = line.split()[-1]

    pats = {
        'cube': is_cube,
        'square': is_square,
        'power': lambda x: is_power(x, b=int(last)),
        'multiple': lambda x: is_multiple(x, of=int(last)),
        'palindrome': is_palindrome,
    }

    for pat, func in pats.items():
        if pat in rule:
            return letter, func, rule

    raise NotImplementedError(f'Unknown rule: {line}')


def is_cube(x: int):
    return int(x**(1 / 3))**3 == x


def is_square(x: int):
    return int(x**(1 / 2))**2 == x


@memory.cache
def is_power(x: int, b: int):
    for e in range(x + 1):
        if b**e == x:
            return True
    return False


def is_multiple(x: int, of: int):
    return x % of == 0


def is_palindrome(x: int):
    return all(a == b for a, b in zip(str(x), str(x)[::-1]))


def extract_sequence(
    start: Pos,
    vert: bool,
):
    offset = (1, 0) if vert else (0, 1)
    coords = (start, )

    p = add(start, offset)
    while p not in STARTS.values():
        coords += (p, )
        p = add(p, offset)

    letter = G[start].upper() if vert else G[start].lower()
    return Sequence(letter, coords)


def extract_number(s: Sequence):
    return int(''.join(str(GELF[c]) for c in s.coords))


def part2():

    incorrect_letters = {s.letter for s in incorrect}
    correct_positions = {p for s in correct for p in s.coords}
    incorrect_positions = {
        p
        for s in incorrect
        for p in s.coords
    } - correct_positions

    solver = Solver()
    ngrid = {
        p: (GELF[p] if p in correct_positions else Int(f'p_{p[0]}_{p[1]}'))
        for p in G
    }

    for p in incorrect_positions:
        assert not isinstance(ngrid[p], int)
        solver.add(ngrid[p] >= 0)
        solver.add(ngrid[p] <= 9)

    ztemps = []
    seqnum = {}

    for letter, rule, line in RULES:
        if letter not in incorrect_letters:
            continue
        s = sequences[letter]
        z = Sum(ngrid[p] * 10**i for i, p in enumerate(s.coords[::-1]))

        if 'cube' in line:
            ztemps.append(t := Int(f'tmp_{len(ztemps)}'))
            solver.add(z == t * t * t)
        elif 'square' in line:
            ztemps.append(t := Int(f'tmp_{len(ztemps)}'))
            solver.add(z == t * t)
        elif 'multiple' in line:
            ztemps.append(t := Int(f'tmp_{len(ztemps)}'))
            last = int(line.split()[-1])
            solver.add(z == last * t)
        elif 'power' in line:
            base = int(line.split()[-1])
            conds = []
            for e in count():
                v = base**e
                if math.log10(v) > len(s.coords):
                    break
                conds.append(z == v)
            solver.add(Or(conds))

        elif 'palindrome' in line:
            solver.add(
                And(
                    ngrid[p] == ngrid[q]
                    for p, q in zip(s.coords, s.coords[::-1])
                )
            )

    print('solving')
    assert solver.check() == sat, f'Model not satisfied: {solver.check()}'
    m = solver.model()

    s = ''
    for r in range(ROWS):
        for c in range(COLS):
            v = ngrid[r, c]
            s += str(v if isinstance(v, int) else m.evaluate(v))
        s += '\n'

    print(s)
    s = re.sub(r'[0,2,4,6,8]', '.', s)
    print(s)
    return sum(map(int, re.findall(r'\d+', s)))


if len(sys.argv) > 1:
    print('reading', sys.argv[1])
    file = open(sys.argv[1])
else:
    print('reading from stdin...')
    file = sys.stdin

a, b, cvals = file.read().strip().split('\n\n')

G = {
    (r, c): v
    for r, line in enumerate(a.split('\n'))
    for c, v in enumerate(line)
}

GELF = {
    (r, c): int(v)
    for r, line in enumerate(cvals.split('\n'))
    for c, v in enumerate(line)
}

RULES = list(map(make_rule, b.split('\n')))

ROWS = 1 + max(r for r, _ in G)
COLS = 1 + max(c for _, c in G)
STARTS = {v: k for k, v in G.items() if v != '.'}

sequences = {}

for letter, start in STARTS.items():
    v = extract_sequence(start, True)
    h = extract_sequence(start, False)
    sequences[v.letter] = v
    sequences[h.letter] = h

correct = []
incorrect = []

a1 = 0
for letter, rule, line in RULES:
    s = sequences[letter]
    v = extract_number(s)
    if not rule(v):
        a1 += v
        incorrect.append(s)
    else:
        correct.append(s)

print('part1:', a1)

a2 = part2()

print('part2:', a2)

assert a1 == 1401106
assert a2 == 517533251
