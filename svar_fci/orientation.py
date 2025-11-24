from collections import deque
from typing import Set, List, Generator

from .graph import DynamicPAG, NULL, CIRCLE, ARROW, TAIL

# --------------------------------------------------------------------
# Dynamic pds_s (time-restricted possible-d-sep)
# --------------------------------------------------------------------


def pds_s(graph: DynamicPAG, i: int, j: int) -> Set[int]:
    """
    Time-restricted possible-d-sep set pds_s(X_i, X_j, P).
    Paths must respect the collider/triangle condition and lag <= maxlag.
    """
    info_i = graph.decode_node(i)
    info_j = graph.decode_node(j)
    maxlag = max(info_i.lag, info_j.lag)

    result = set()
    queue = deque([[i]])
    visited_paths = set()

    def path_ok(path):
        if len(path) < 3:
            return True
        for a, b, c in zip(path[:-2], path[1:-1], path[2:]):
            if not (graph.is_collider(a, b, c) or graph.forms_triangle(a, b, c)):
                return False
        return True

    while queue:
        path = queue.popleft()
        last = path[-1]
        for nb in graph.neighbors(last):
            if nb in path:
                continue
            new_path = path + [nb]
            key = tuple(new_path)
            if key in visited_paths:
                continue
            visited_paths.add(key)
            if not path_ok(new_path):
                continue
            info_nb = graph.decode_node(nb)
            if info_nb.lag <= maxlag:
                result.add(nb)
            queue.append(new_path)

    result.discard(i)
    result.discard(j)
    return result


# --------------------------------------------------------------------
# Path predicates
# --------------------------------------------------------------------


def is_uncovered(graph: DynamicPAG, path: List[int]) -> bool:
    if len(path) < 3:
        return True
    for idx in range(1, len(path) - 1):
        if graph.is_adjacent(path[idx - 1], path[idx + 1]):
            return False
    return True


def is_circle_edge(graph: DynamicPAG, u: int, v: int) -> bool:
    return graph.is_adjacent(u, v) and graph.M[u, v] == CIRCLE and graph.M[v, u] == CIRCLE


def is_circle_path(graph: DynamicPAG, path: List[int]) -> bool:
    if len(path) < 2:
        return False
    for a, b in zip(path[:-1], path[1:]):
        if not is_circle_edge(graph, a, b):
            return False
    return True


def uncovered_circle_paths(graph: DynamicPAG, a: int, b: int) -> Generator[List[int], None, None]:
    """
    Generate uncovered circle-circle paths from a to b (length >= 3).
    """
    queue = deque([[a]])
    while queue:
        path = queue.popleft()
        last = path[-1]
        for nb in graph.neighbors(last):
            if nb in path:
                continue
            if not is_circle_edge(graph, last, nb):
                continue
            new_path = path + [nb]
            if not is_uncovered(graph, new_path):
                continue
            if nb == b and len(new_path) >= 3:
                yield new_path
            else:
                queue.append(new_path)


def is_pd_edge(graph: DynamicPAG, u: int, v: int) -> bool:
    """
    Edge u ?-? v is potentially directed from u to v if:
    - it is not into u (mark at u != ARROW)
    - it is not out of v (mark at v != TAIL)
    """
    if not graph.is_adjacent(u, v):
        return False
    return graph.M[v, u] != ARROW and graph.M[u, v] != TAIL


def uncovered_pd_paths(graph: DynamicPAG, start: int, end: int) -> Generator[List[int], None, None]:
    stack = [(start, [start])]
    while stack:
        node, path = stack.pop()
        for nb in graph.neighbors(node):
            if nb in path:
                continue
            if not is_pd_edge(graph, node, nb):
                continue
            new_path = path + [nb]
            if not is_uncovered(graph, new_path):
                continue
            if nb == end:
                yield new_path
            else:
                stack.append((nb, new_path))


def is_discriminating_path(graph: DynamicPAG, path: List[int], beta: int, theta: int, gamma: int) -> bool:
    """
    Check if path <theta, q1, ..., qk, beta, gamma> discriminates for beta.
    """
    if len(path) < 4:
        return False
    if path[0] != theta or path[-1] != gamma:
        return False
    if beta not in path[1:-1]:
        return False
    if graph.is_adjacent(theta, gamma):
        return False

    beta_idx = path.index(beta)
    if beta_idx == 0 or beta_idx >= len(path) - 1:
        return False
    # beta must be immediately before gamma on the path
    if path[beta_idx + 1] != gamma:
        return False

    # every vertex between theta and beta is a collider and a parent of gamma
    for idx in range(1, beta_idx):
        v = path[idx]
        prev_v = path[idx - 1]
        next_v = path[idx + 1]
        if not graph.is_collider(prev_v, v, next_v):
            return False
        if not (graph.is_adjacent(v, gamma) and graph.M[v, gamma] == ARROW and graph.M[gamma, v] == TAIL):
            return False

    return True


def discriminating_paths(graph: DynamicPAG, theta: int, gamma: int, beta: int) -> Generator[List[int], None, None]:
    """
    Generate simple paths from theta to gamma that are discriminating for beta.
    """
    stack = [(theta, [theta])]
    n_limit = graph.n_nodes + 1
    while stack:
        node, path = stack.pop()
        for nb in graph.neighbors(node):
            if nb in path:
                continue
            new_path = path + [nb]
            if len(new_path) > n_limit:
                continue
            if nb == gamma:
                if is_discriminating_path(graph, new_path, beta, theta, gamma):
                    yield new_path
            else:
                stack.append((nb, new_path))


# --------------------------------------------------------------------
# R1-R10 (Zhang 2008) orientation rules
# --------------------------------------------------------------------


def apply_rules(G: DynamicPAG):
    """
    Apply Zhang's orientation rules R1-R10 to exhaustion.
    """
    n = G.n_nodes

    def non_adjacent(a, b):
        return not G.is_adjacent(a, b)

    def arrow_into(child, parent):
        return G.is_adjacent(parent, child) and G.M[parent, child] == ARROW

    def circle_at(i, j):
        return G.is_adjacent(i, j) and G.M[i, j] == CIRCLE

    def tail_at(i, j):
        return G.is_adjacent(i, j) and G.M[i, j] == TAIL

    changed = True
    while changed:
        changed = False

        # ----------------- R1 -----------------
        for beta in range(n):
            for alpha in G.neighbors(beta):
                if not arrow_into(beta, alpha):
                    continue
                for gamma in G.neighbors(beta):
                    if gamma == alpha:
                        continue
                    if not circle_at(gamma, beta):
                        continue
                    if not non_adjacent(alpha, gamma):
                        continue
                    if not (G.M[beta, gamma] == ARROW and G.M[gamma, beta] == TAIL):
                        G.orient_with_homology(beta, gamma, ARROW, TAIL)
                        changed = True

        # ----------------- R2 -----------------
        for beta in range(n):
            for alpha in G.neighbors(beta):
                for gamma in G.neighbors(beta):
                    if gamma == alpha:
                        continue
                    if not (arrow_into(beta, alpha) and arrow_into(gamma, beta)):
                        continue
                    if not G.is_adjacent(alpha, gamma):
                        continue
                    if G.M[alpha, gamma] != CIRCLE:
                        continue
                    preserved = G.M[gamma, alpha]
                    if G.M[alpha, gamma] != ARROW:
                        G.orient_with_homology(alpha, gamma, ARROW, preserved)
                        changed = True

        # ----------------- R3 -----------------
        for beta in range(n):
            for alpha in G.neighbors(beta):
                if not arrow_into(beta, alpha):
                    continue
                for gamma in G.neighbors(beta):
                    if gamma == alpha:
                        continue
                    if not arrow_into(beta, gamma):
                        continue
                    if not non_adjacent(alpha, gamma):
                        continue
                    for theta in range(n):
                        if theta in (alpha, beta, gamma):
                            continue
                        if not (G.is_adjacent(alpha, theta) and G.is_adjacent(theta, gamma) and G.is_adjacent(theta, beta)):
                            continue
                        if not (circle_at(alpha, theta) and circle_at(gamma, theta)):
                            continue
                        if not circle_at(theta, beta):
                            continue
                        current_mark_beta = G.M[theta, beta]
                        if not (current_mark_beta == ARROW and G.M[beta, theta] == TAIL):
                            G.orient_with_homology(theta, beta, ARROW, TAIL)
                            changed = True

        # ----------------- R4 -----------------
        for beta in range(n):
            for theta in range(n):
                if theta == beta:
                    continue
                for gamma in range(n):
                    if gamma in (theta, beta):
                        continue
                    if not G.is_adjacent(beta, gamma):
                        continue
                    if not circle_at(gamma, beta):
                        continue
                    found = False
                    for path in discriminating_paths(G, theta, gamma, beta):
                        found = True
                        sepset = G.get_sepset(theta, gamma)
                        if beta in sepset:
                            if not (G.M[beta, gamma] == ARROW and G.M[gamma, beta] == TAIL):
                                G.orient_with_homology(beta, gamma, ARROW, TAIL)
                                changed = True
                        else:
                            b_idx = path.index(beta)
                            if b_idx == 0:
                                continue
                            alpha = path[b_idx - 1]
                            if not (G.M[alpha, beta] == ARROW and G.M[beta, alpha] == ARROW):
                                G.orient_with_homology(alpha, beta, ARROW, ARROW)
                                changed = True
                            if not (G.M[beta, gamma] == ARROW and G.M[gamma, beta] == ARROW):
                                G.orient_with_homology(beta, gamma, ARROW, ARROW)
                                changed = True
                        break
                    if found:
                        continue

        # ----------------- R5 -----------------
        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                if not (G.is_adjacent(a, b) and G.M[a, b] == CIRCLE and G.M[b, a] == CIRCLE):
                    continue
                for path in uncovered_circle_paths(G, a, b):
                    if len(path) < 3:
                        continue
                    gamma = path[1]
                    theta = path[-2]
                    if not non_adjacent(a, theta):
                        continue
                    if not non_adjacent(b, gamma):
                        continue
                    # orient all edges on path as tail-tail plus a-b
                    for u, v in zip(path[:-1], path[1:]):
                        if not (tail_at(u, v) and tail_at(v, u)):
                            G.orient_with_homology(u, v, TAIL, TAIL)
                            changed = True
                    if not (tail_at(a, b) and tail_at(b, a)):
                        G.orient_with_homology(a, b, TAIL, TAIL)
                        changed = True
                    break

        # ----------------- R6 -----------------
        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                if not (G.is_adjacent(a, b) and tail_at(a, b) and tail_at(b, a)):
                    continue
                for c in G.neighbors(b):
                    if c in (a, b):
                        continue
                    if G.M[c, b] != CIRCLE:
                        continue
                    G.orient_with_homology(b, c, G.M[b, c], TAIL)
                    changed = True

        # ----------------- R7 -----------------
        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                if not G.is_adjacent(a, b):
                    continue
                if not (tail_at(b, a) and circle_at(a, b)):
                    continue
                for c in range(n):
                    if c in (a, b):
                        continue
                    if not G.is_adjacent(c, b):
                        continue
                    if G.M[c, b] != CIRCLE:
                        continue
                    if not non_adjacent(a, c):
                        continue
                    if G.M[c, b] != TAIL:
                        G.orient_with_homology(b, c, G.M[b, c], TAIL)
                        changed = True
                        break

        # ----------------- R8 -----------------
        for a in range(n):
            for b in G.neighbors(a):
                for c in G.neighbors(b):
                    if c == a:
                        continue
                    # beta -> gamma
                    if not arrow_into(c, b):
                        continue
                    # case1: alpha -> beta, case2: alpha -o beta
                    case1 = arrow_into(b, a)
                    case2 = (G.is_adjacent(a, b) and G.M[a, b] == CIRCLE and G.M[b, a] == TAIL)
                    if not (case1 or case2):
                        continue
                    # alpha o-> gamma
                    if not (G.is_adjacent(a, c) and G.M[a, c] == ARROW and G.M[c, a] == CIRCLE):
                        continue
                    if not (G.M[a, c] == ARROW and G.M[c, a] == TAIL):
                        G.orient_with_homology(a, c, ARROW, TAIL)
                        changed = True

        # ----------------- R9 -----------------
        for a in range(n):
            for c in range(n):
                if c == a:
                    continue
                if not (G.is_adjacent(a, c) and G.M[a, c] == ARROW and G.M[c, a] == CIRCLE):
                    continue
                for path in uncovered_pd_paths(G, a, c):
                    if len(path) < 3:
                        continue
                    beta = path[1]
                    if not non_adjacent(beta, c):
                        continue
                    if not (G.M[a, c] == ARROW and G.M[c, a] == TAIL):
                        G.orient_with_homology(a, c, ARROW, TAIL)
                        changed = True
                    break

        # ----------------- R10 -----------------
        for a in range(n):
            for c in range(n):
                if c == a:
                    continue
                if not (G.is_adjacent(a, c) and G.M[a, c] == ARROW and G.M[c, a] == CIRCLE):
                    continue
                parents = [v for v in range(n) if arrow_into(c, v)]
                for beta in parents:
                    if beta == a:
                        continue
                    for theta in parents:
                        if theta in (beta, a):
                            continue
                        p1 = next(uncovered_pd_paths(G, a, beta), None)
                        if p1 is None:
                            continue
                        p2 = next(uncovered_pd_paths(G, a, theta), None)
                        if p2 is None:
                            continue
                        if len(p1) < 2 or len(p2) < 2:
                            continue
                        mu = p1[1]
                        omega = p2[1]
                        if mu == omega:
                            continue
                        if not non_adjacent(mu, omega):
                            continue
                        if not (G.M[a, c] == ARROW and G.M[c, a] == TAIL):
                            G.orient_with_homology(a, c, ARROW, TAIL)
                            changed = True
                        break
                    if changed:
                        break
                if changed:
                    break
