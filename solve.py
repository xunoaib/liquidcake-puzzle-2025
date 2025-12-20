import sys
from dataclasses import dataclass

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

    return Sequence(G[start], coords)


a, b = sys.stdin.read().strip().split('\n\n')

G = {
    (r, c): v
    for r, line in enumerate(a.split('\n'))
    for c, v in enumerate(line)
}

ROWS = 1 + max(r for r, _ in G)
COLS = 1 + max(c for _, c in G)
STARTS = {v: k for k, v in G.items() if v != '.'}

for letter, start in STARTS.items():
    sv = extract_sequence(start, True)
    sh = extract_sequence(start, False)
    print(sh)
