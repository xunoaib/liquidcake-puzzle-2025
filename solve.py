import math
import sys
from dataclasses import dataclass
from functools import partial

'''
a.b.
.c.d
e...
.f..

a is a cube
b is a palindrome
c is a power of 2
d is a square
e is a multiple of 127
f is a multiple of 29
A is a multiple of 12
B is a multiple of 3099
C is a palindrome
D is a multiple of 2253
E is a multiple of 3
F is a power of 3
'''

from z3 import And, Int, Or, Solver, sat

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
            return letter, func

    raise NotImplementedError(f'Unknown rule: {line}')


def is_cube(x: int):
    return int(x**(1 / 3))**3 == x


def is_square(x: int):
    return int(x**(1 / 2))**2 == x


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
    return int(''.join(GELF[c] for c in s.coords))


a, b, c = sys.stdin.read().strip().split('\n\n')

G = {
    (r, c): v
    for r, line in enumerate(a.split('\n'))
    for c, v in enumerate(line)
}

GELF = {
    (r, c): v
    for r, line in enumerate(c.split('\n'))
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

t = 0
for letter, rule in RULES:
    s = sequences[letter]
    v = extract_number(s)

    if not rule(v):
        print('wrong', v)
        t += v
    else:
        print('right', v)

print('part1:', t)
