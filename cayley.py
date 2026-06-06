"""
Left Cayley graph generator for finitely presented semigroups and monoids.

Usage (library):
    from cayley import cayley_graph
    cayley_graph(['a', 'b'], [('aa', 'a'), ('bb', 'b')], ball_size=5)

Usage (CLI):
    python cayley.py a,b aa:a bb:b -n 5 --view
    python cayley.py a,b aa:a bb:b aba:a bab:b -n 4 --engine neato --view
    python cayley.py a,b --semigroup -n 3 --view   # free semigroup (no relations)
"""

import argparse
import sys
from typing import Callable, Optional

import graphviz


COLORS = [
    '#1f77b4',  # blue
    '#d62728',  # red
    '#2ca02c',  # green
    '#ff7f0e',  # orange
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#17becf',  # cyan
]


SUPERSCRIPTS = str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹')


def exp_label(word: str) -> str:
    """Format consecutive generator runs with superscript exponents, e.g. 'qqpp' → 'q²p²'."""
    if not word:
        return 'ε'
    parts = []
    i = 0
    while i < len(word):
        ch = word[i]
        run = 1
        while i + run < len(word) and word[i + run] == ch:
            run += 1
        parts.append(ch if run == 1 else ch + str(run).translate(SUPERSCRIPTS))
        i += run
    return ''.join(parts)


def grid_pos(word: str, generators: list[str], spacing: float = 1.5) -> str:
    """Compute a pinned neato position 'x,y!' for a grid layout.

    generators[0] goes downward (y-axis), generators[1] goes rightward (x-axis).
    Assumes canonical words are of the form g0^a g1^b (e.g. bicyclic monoid).
    """
    x = word.count(generators[1]) * spacing if len(generators) > 1 else 0.0
    y = -word.count(generators[0]) * spacing
    return f'{x},{y}!'


def normalize(word: str, rules: list[tuple[str, str]], max_steps: int = 100_000) -> str:
    """Apply rewriting rules leftmost-first until fixpoint.

    Raises RuntimeError if max_steps is reached (likely a non-terminating system).
    """
    for step in range(max_steps):
        for lhs, rhs in rules:
            if lhs in word:
                word = word.replace(lhs, rhs, 1)
                break
        else:
            return word  # fixpoint reached
    raise RuntimeError(
        f"Normalization did not terminate after {max_steps} steps — "
        "check that your rules form a terminating rewriting system."
    )


def build_cayley_graph(
    generators: list[str],
    rules: list[tuple[str, str]],
    ball_size: int,
    monoid: bool = True,
) -> tuple[dict[str, int], list[tuple[str, str, str]]]:
    """Build the left Cayley graph ball of radius ball_size.

    An edge g: w -> g·w (left-multiply by generator g).

    Returns:
        nodes: canonical word -> BFS depth from identity (or generators if semigroup)
        edges: list of (source, target, generator) for the induced subgraph of the ball
    """
    if monoid:
        initial: dict[str, int] = {normalize('', rules): 0}
    else:
        initial = {}
        for g in generators:
            canon = normalize(g, rules)
            initial.setdefault(canon, 1)

    visited: dict[str, int] = dict(initial)
    frontier: set[str] = set(initial)

    for depth in range(ball_size):
        next_frontier: set[str] = set()
        for word in frontier:
            for gen in generators:
                target = normalize(gen + word, rules)
                if target not in visited:
                    visited[target] = depth + 1
                    next_frontier.add(target)
        frontier = next_frontier
        if not frontier:
            break  # finite semigroup/monoid fully enumerated

    # Collect all edges induced by the ball (including back-edges and self-loops).
    seen: set[tuple[str, str, str]] = set()
    edges: list[tuple[str, str, str]] = []
    for word in visited:
        for gen in generators:
            target = normalize(gen + word, rules)
            if target in visited:
                e = (word, target, gen)
                if e not in seen:
                    seen.add(e)
                    edges.append(e)

    return visited, edges


def draw_cayley_graph(
    nodes: dict[str, int],
    edges: list[tuple[str, str, str]],
    generators: list[str],
    output_file: str = 'cayley_graph',
    engine: str = 'dot',
    fmt: str = 'png',
    view: bool = False,
    label_fn: Optional[Callable[[str], str]] = None,
    pos_fn: Optional[Callable[[str], str]] = None,
) -> str:
    """Render nodes/edges to an image via Graphviz. Returns the output file path."""
    gen_color = {g: COLORS[i % len(COLORS)] for i, g in enumerate(generators)}

    dot = graphviz.Digraph(engine=engine)
    dot.attr(bgcolor='white', pad='0.5')
    dot.attr('node', shape='circle', style='filled', fillcolor='#ddeeff',
             fontname='Helvetica', margin='0.15')
    dot.attr('edge', fontname='Helvetica', fontsize='11')

    for word in sorted(nodes, key=lambda w: (nodes[w], w)):
        node_id = word or '__eps__'
        label = label_fn(word) if label_fn else (word or 'ε')
        attrs: dict[str, str] = {'pos': pos_fn(word)} if pos_fn else {}
        dot.node(node_id, label=label, **attrs)

    for src, tgt, gen in edges:
        dot.edge(
            src or '__eps__',
            tgt or '__eps__',
            label=gen,
            color=gen_color[gen],
            fontcolor=gen_color[gen],
        )

    return dot.render(output_file, format=fmt, cleanup=True, view=view)


def cayley_graph(
    generators: list[str],
    rules: list[tuple[str, str]],
    ball_size: int,
    monoid: bool = True,
    output_file: str = 'cayley_graph',
    engine: str = 'dot',
    fmt: str = 'png',
    view: bool = False,
    label_fn: Optional[Callable[[str], str]] = None,
    pos_fn: Optional[Callable[[str], str]] = None,
) -> str:
    """Build and render the Cayley graph. Returns the output file path."""
    nodes, edges = build_cayley_graph(generators, rules, ball_size, monoid)
    return draw_cayley_graph(nodes, edges, generators, output_file, engine, fmt, view,
                             label_fn=label_fn, pos_fn=pos_fn)


def build_schutzenberger_graph(
    generators: list[str],
    rules: list[tuple[str, str]],
    element: str,
    ball_size: int,
) -> tuple[dict[str, int], list[tuple[str, str, str]]]:
    """Find the Schützenberger graph of element in the left Cayley graph.

    Explores up to ball_size steps from element by left-multiplying generators,
    then extracts the strongly connected component of element from that ball.
    The SCC equals the L-class of element (elements generating the same principal
    left ideal) within the explored region.
    """
    root = normalize(element, rules)

    # Forward BFS from root
    visited: dict[str, int] = {root: 0}
    frontier: set[str] = {root}
    for depth in range(ball_size):
        next_frontier: set[str] = set()
        for word in frontier:
            for gen in generators:
                target = normalize(gen + word, rules)
                if target not in visited:
                    visited[target] = depth + 1
                    next_frontier.add(target)
        frontier = next_frontier
        if not frontier:
            break

    # Collect all edges within the ball and build reverse adjacency list
    radj: dict[str, list[str]] = {w: [] for w in visited}
    seen: set[tuple[str, str, str]] = set()
    all_edges: list[tuple[str, str, str]] = []
    for word in visited:
        for gen in generators:
            target = normalize(gen + word, rules)
            if target in visited:
                e = (word, target, gen)
                if e not in seen:
                    seen.add(e)
                    all_edges.append(e)
                    radj[target].append(word)

    # Backward BFS from root on reversed edges — finds all nodes from which root is reachable
    back_visited: set[str] = {root}
    back_frontier: set[str] = {root}
    while back_frontier:
        nxt: set[str] = set()
        for node in back_frontier:
            for pred in radj[node]:
                if pred not in back_visited:
                    back_visited.add(pred)
                    nxt.add(pred)
        back_frontier = nxt

    # SCC = forward-reachable from root ∩ nodes from which root is reachable
    scc: set[str] = set(visited) & back_visited
    scc_nodes = {w: visited[w] for w in scc}
    scc_edges = [(s, t, g) for s, t, g in all_edges if s in scc and t in scc]
    return scc_nodes, scc_edges


def schutzenberger_graph(
    generators: list[str],
    rules: list[tuple[str, str]],
    element: str,
    ball_size: int,
    output_file: str = 'schutzenberger_graph',
    engine: str = 'dot',
    fmt: str = 'png',
    view: bool = False,
    label_fn: Optional[Callable[[str], str]] = None,
    pos_fn: Optional[Callable[[str], str]] = None,
) -> str:
    """Build and render the Schützenberger graph of element. Returns the output file path."""
    nodes, edges = build_schutzenberger_graph(generators, rules, element, ball_size)
    return draw_cayley_graph(nodes, edges, generators, output_file, engine, fmt, view,
                             label_fn=label_fn, pos_fn=pos_fn)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_rules(strs: list[str]) -> list[tuple[str, str]]:
    rules = []
    for s in strs:
        if ':' not in s:
            raise argparse.ArgumentTypeError(f"Rule '{s}' must be 'lhs:rhs'")
        lhs, rhs = s.split(':', 1)
        rules.append((lhs, rhs))
    return rules


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate the left Cayley graph of a finitely presented semigroup/monoid.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Monoid with relations a²=a, b²=b (two idempotents), ball of depth 5
  python cayley.py a,b aa:a bb:b -n 5 --view

  # Band (a²=a, b²=b, aba=a, bab=b), neato layout
  python cayley.py a,b aa:a bb:b aba:a bab:b -n 4 --engine neato --view

  # Free semigroup on {a,b} up to depth 3 (no relations)
  python cayley.py a,b --semigroup -n 3 --view
""",
    )
    parser.add_argument('generators',
                        help="Comma-separated generator symbols, e.g. 'a,b'")
    parser.add_argument('rules', nargs='*',
                        help="Oriented rewriting rules as 'lhs:rhs', e.g. 'aa:a' 'bb:b'")
    parser.add_argument('-n', '--ball-size', type=int, default=5,
                        help='BFS depth from identity (default: 5)')
    parser.add_argument('--semigroup', action='store_true',
                        help='No identity element; BFS starts from the generators')
    parser.add_argument('-o', '--output', default='cayley_graph',
                        help='Output filename without extension (default: cayley_graph)')
    parser.add_argument('--engine', default='dot',
                        choices=['dot', 'neato', 'fdp', 'sfdp', 'circo', 'twopi'],
                        help='Graphviz layout engine (default: dot)')
    parser.add_argument('--format', default='png',
                        choices=['png', 'svg', 'pdf'],
                        help='Output image format (default: png)')
    parser.add_argument('--view', action='store_true',
                        help='Open the rendered image automatically')
    parser.add_argument('--exp-labels', action='store_true',
                        help='Use superscript exponent notation for repeated generators '
                             '(e.g. qqpp → q²p²)')
    parser.add_argument('--grid', action='store_true',
                        help='Grid layout: generators[0] goes downward, generators[1] '
                             'goes rightward; switches engine to neato')
    parser.add_argument('--element', default=None, metavar='WORD',
                        help='If given, draw the Schützenberger graph (L-class SCC) of '
                             'this element instead of the full Cayley graph')

    args = parser.parse_args()
    gens = [g.strip() for g in args.generators.split(',')]
    rules = _parse_rules(args.rules)

    label_fn = exp_label if args.exp_labels else None

    engine = args.engine
    pos_fn = None
    if args.grid:
        if len(gens) < 2:
            parser.error('--grid requires at least 2 generators')
        pos_fn = lambda w: grid_pos(w, gens)
        if engine == 'dot':
            engine = 'neato'

    if args.element is not None:
        nodes, edges = build_schutzenberger_graph(gens, rules, args.element, args.ball_size)
        print(f"Schützenberger graph of '{args.element}': {len(nodes)} elements, "
              f"{len(edges)} edges", file=sys.stderr)
        output = args.output if args.output != 'cayley_graph' else 'schutzenberger_graph'
    else:
        nodes, edges = build_cayley_graph(gens, rules, args.ball_size, monoid=not args.semigroup)
        print(f"Elements in ball of depth {args.ball_size}: {len(nodes)}", file=sys.stderr)
        print(f"Edges: {len(edges)}", file=sys.stderr)
        output = args.output

    path = draw_cayley_graph(
        nodes, edges, gens,
        output_file=output,
        engine=engine,
        fmt=args.format,
        view=args.view,
        label_fn=label_fn,
        pos_fn=pos_fn,
    )
    print(path)


if __name__ == '__main__':
    main()
