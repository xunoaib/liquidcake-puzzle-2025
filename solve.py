import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import override

Pos = tuple[int, int]

RULE_PREDICATES = {
    'square': lambda x, _: int(x**(1 / 2))**2 == x,
    'cube': lambda x, _: round(x**(1 / 3))**3 == x,
    'power': lambda x, b: is_power(x, b),
    'multiple': lambda x, b: x % b == 0,
    'palindrome': lambda x, _: str(x) == str(x)[::-1],
}

CANDIDATE_FUNCS = {
    'square': lambda n, _: exponent_candidates(n, 2),
    'cube': lambda n, _: exponent_candidates(n, 3),
    'palindrome': lambda n, _: palindrome_candidates(n),
    'power': lambda n, b: power_candidates(n, b),
    'multiple': lambda n, b: multiple_candidates(n, b),
}


class Rule:

    def __init__(self, text):
        m = re.match(
            r'(.*?) is a (square|cube|power|multiple|palindrome)(?: of (\d+))?$',
            text
        )
        assert m, text
        self.letter, self.type, of = m.groups()
        self.of = int(of) if of else None

    def valid(self, x: int):
        return RULE_PREDICATES[self.type](x, self.of)

    @override
    def __repr__(self) -> str:
        of = f' of {self.of}' if self.of else ''
        return f'<{self.letter} is {self.type}{of}>'


@dataclass
class Sequence:
    letter: str
    coords: list[Pos]
    rule: Rule
    index: dict[Pos, int] = field(init=False, repr=False)
    _all: list[str] = field(init=False, default_factory=list, repr=False)

    def __post_init__(self):
        self.index = {p: i for i, p in enumerate(self.coords)}

    def all_candidates(self) -> list[str]:
        self._all = list(
            CANDIDATE_FUNCS[self.rule.type](len(self.coords), self.rule.of)
        )
        return self._all

    def filter_candidates(self, known: dict[Pos, str]):
        assert self._all
        vals = list(map(known.get, self.coords))
        return [
            c for c in self._all
            if all(v is None or v == c for v, c in zip(vals, c))
        ]

    def candidates(self, known: dict[Pos, str]):
        if not self._all:
            self._all = self.all_candidates()
        self._all = self.filter_candidates(known)
        return self._all


def exponent_candidates(ndigits, exponent):
    lo = round((10**(ndigits - 1))**(1 / exponent))
    hi = round((10**ndigits - 1)**(1 / exponent))
    for k in range(lo, hi + 1):
        v = k**exponent
        if len(str(v)) == ndigits:
            yield str(v)


def palindrome_candidates(ndigits):
    half = (ndigits + 1) // 2
    for h in range(10**(half - 1), 10**half):
        s = str(h)
        yield s + s[-2::-1] if ndigits % 2 else s + s[::-1]


def power_candidates(ndigits, b):
    v = b
    while len(str(v)) < ndigits:
        v *= b
    while len(str(v)) == ndigits:
        yield str(v)
        v *= b


def multiple_candidates(ndigits, b):
    lo = (10**(ndigits - 1) + b - 1) // b * b
    hi = 10**ndigits
    for v in range(lo, hi, b):
        yield str(v)


def is_power(x: int, b: int):
    if x < 1 or b < 2:
        return False

    while x % b == 0:
        x //= b
    return x == 1


def add(a: Pos, b: Pos):
    return (a[0] + b[0]) % ROWS, (a[1] + b[1]) % COLS


def extract_sequence(start: Pos, vert: bool):
    letter = STARTS[start].upper() if vert else STARTS[start].lower()
    offset = (1, 0) if vert else (0, 1)
    coords = [start]
    while (p := add(coords[-1], offset)) not in STARTS:
        coords.append(p)
    return Sequence(letter, coords, RULES[letter])


def resolve_intersections(known: dict[Pos, str]):
    updated = False
    for p in sorted(GUESSES.keys() - known.keys()):
        cands = cell_candidates(p, known)
        if len(cands) == 1:
            known[p] = next(iter(cands))
            updated = True
            print(f'Fixing {p} => {known[p]} ({len(GUESSES)-len(known)} left)')
    return updated


def cell_candidates(p: Pos, known: dict[Pos, str]):
    s0, s1 = SEQS_AT[p]
    i0 = s0.index[p]
    i1 = s1.index[p]
    c0 = {str(c)[i0] for c in s0.candidates(known)}
    c1 = {str(c)[i1] for c in s1.candidates(known)}
    return c0 & c1


def part2(known: dict[Pos, str]):

    while resolve_intersections(known):
        pass

    if unknown := set(GUESSES) - set(known):
        print(f'\n\033[93mSome cells are not fully constrained\033[0m')
        print('Cells:', unknown)
        for p in unknown:
            cands = cell_candidates(p, known)
            print(f'cell candidates @ {p} = {cands}')
            known[p] = next(iter(cands))
            print(f'Arbitrarily fixing {p} => {known[p]}')

    out = '\n'.join(
        ''.join(known[r, c] for c in range(COLS)) for r in range(ROWS)
    )
    masked = re.sub(r'[0,2,4,6,8]', '.', out)

    print()
    print(out)
    print()
    print(masked)
    print()

    return sum(map(int, re.findall(r'\d+', masked)))


if len(sys.argv) > 1:
    print('reading from', sys.argv[1])
    file = open(sys.argv[1])
else:
    print('reading from stdin...')
    file = sys.stdin

grid_txt, rules_txt, vals_txt = file.read().strip().split('\n\n')

STARTS = {
    (r, c): v
    for r, line in enumerate(grid_txt.split('\n'))
    for c, v in enumerate(line) if v != '.'
}

RULES = {r.letter: r for r in map(Rule, rules_txt.split('\n'))}

GUESSES = {
    (r, c): v
    for r, line in enumerate(vals_txt.split('\n'))
    for c, v in enumerate(line)
}

ROWS = 1 + max(r for r, _ in GUESSES)
COLS = 1 + max(c for _, c in GUESSES)

a1 = 0
known = {}

# Collect vert/horiz letter sequences
SEQS_AT: dict[Pos, list[Sequence]] = defaultdict(list)
for pos, ch in STARTS.items():
    for vert in [True, False]:
        seq = extract_sequence(pos, vert)
        for p in seq.coords:
            SEQS_AT[p].append(seq)

        # validate rule for part 1
        value = int(''.join(GUESSES[p] for p in seq.coords))
        if seq.rule.valid(value):
            known.update({p: GUESSES[p] for p in seq.coords})
        else:
            a1 += value

# known = {}  # also solvable w/o any values

print('part1:', a1)
print('part2:', a2 := part2(known))  # fix known values

assert a1 == 1401106
assert a2 == 517533251
