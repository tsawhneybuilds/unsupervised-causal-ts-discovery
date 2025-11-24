import unittest
import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from svar_fci.graph import DynamicPAG, CIRCLE, ARROW, TAIL, NULL
from svar_fci.orientation import apply_rules

class TestOrientationRules(unittest.TestCase):
    def setUp(self):
        # Simple 4-node graph at a single lag for basic rule testing
        self.var_names = ["A", "B", "C", "D", "E"]
        self.max_lag = 0
        self.G = DynamicPAG(self.var_names, self.max_lag)
        # Clear all edges to start with empty graph for precise testing
        self.G.M[:] = NULL
        
        # Node indices
        self.A = self.G.node_index(0, 0)
        self.B = self.G.node_index(1, 0)
        self.C = self.G.node_index(2, 0)
        self.D = self.G.node_index(3, 0)
        self.E = self.G.node_index(4, 0)

    def _connect(self, u, v, mark_u, mark_v):
        """
        Connect u and v with specified marks.
        mark_u is the mark at u's end.
        mark_v is the mark at v's end.
        """
        self.G.M[u, v] = mark_v  # mark at v
        self.G.M[v, u] = mark_u  # mark at u

    def _connect_circle(self, u, v):
        self._connect(u, v, CIRCLE, CIRCLE)

    def test_R1_double_triangle(self):
        """
        R1: If A *-> B o-* C, and A, C not adjacent:
            Orient B o-* C as B -> C.
        """
        # A *-> B (e.g., A -> B)
        self._connect(self.A, self.B, TAIL, ARROW) # Tail at A, Arrow at B
        
        # B o-* C (e.g., B o-o C)
        self._connect(self.B, self.C, CIRCLE, CIRCLE)
        
        # A, C not adjacent (default)

        apply_rules(self.G)

        # Check B -> C: Tail at B (-1), Arrow at C (1)
        self.assertEqual(self.G.M[self.B, self.C], ARROW)
        self.assertEqual(self.G.M[self.C, self.B], TAIL)

    def test_R2_chain_cycle(self):
        """
        R2: If A -> B *-> C (or A *-> B -> C) and A *-o C:
            Orient A *-o C as A *-> C.
        """
        # A -> B
        self._connect(self.A, self.B, TAIL, ARROW)
        # B -> C
        self._connect(self.B, self.C, TAIL, ARROW)
        # A *-o C (e.g., tail at A, circle at C)
        self._connect(self.A, self.C, TAIL, CIRCLE)

        apply_rules(self.G)

        # Check A *-> C (arrow at C)
        self.assertEqual(self.G.M[self.A, self.C], ARROW)

    def test_R3_double_triangle_diamond(self):
        """
        R3: If A *-> B <-* C, A *-o D o-* C, A, C not adjacent, and D *-o B:
            Orient D *-o B as D *-> B.
        """
        # Collider A *-> B <-* C
        self._connect(self.A, self.B, TAIL, ARROW)
        self._connect(self.C, self.B, TAIL, ARROW)
        
        # A *-o D o-* C
        self._connect(self.A, self.D, TAIL, CIRCLE)
        self._connect(self.C, self.D, TAIL, CIRCLE)
        
        # D *-o B
        self._connect(self.D, self.B, TAIL, CIRCLE)

        apply_rules(self.G)

        # Expect D *-> B (arrow at B)
        self.assertEqual(self.G.M[self.D, self.B], ARROW)

    def test_R5_uncovered_circle_path(self):
        """
        R5: For A o-o B with uncovered circle path A - X - Y - B
        (all circles), with A not adjacent Y and B not adjacent X,
        orient all edges on the path and A-B as undirected (tail-tail).
        """
        X = self.C
        Y = self.D
        # Circle path A-X-Y-B
        self._connect_circle(self.A, X)
        self._connect_circle(X, Y)
        self._connect_circle(Y, self.B)
        # Edge A o-o B
        self._connect_circle(self.A, self.B)

        apply_rules(self.G)

        for (u, v) in [(self.A, X), (X, Y), (Y, self.B), (self.A, self.B)]:
            self.assertEqual(self.G.M[u, v], TAIL)
            self.assertEqual(self.G.M[v, u], TAIL)

    def test_R4_discriminating_path_collider(self):
        """
        R4: Discriminating path U ... A B C for B.
        If B is NOT in Sepset(U, C), then A *-> B <-* C.
        """
        # Path U -> A -- B -- C, but need specific structure for discr path
        # U, ..., A, B, C
        # Here U=A, A=B, B=C, C=D for simplicity? No, lengths.
        # Let's do U=A, A=B, B=C, C=D.
        # Path: A <-> B <-> C o-o D ?
        # Definition: A, B, C, D. 
        # D is end. A is start. B, C are intermediate.
        # Need A adjacent to D? No, A and D NOT adjacent.
        # Vertices between A and C (i.e., B) must be colliders on path and parents of D.
        
        # Let's try path: A <-> B <-> C.
        # B is collider? No, let's construct exactly:
        # A <-> B ... B must be collider. A -> B <- X?
        pass

    # Additional tests can be added for R8-R10

if __name__ == '__main__':
    unittest.main()
