"""
This module implements QSearch's Leap Compiler
as a native tool plugin to QFAST.
"""

import qsearch
import qsearch
from qsearch import unitaries, advanced_unitaries, leap_compiler, multistart_solvers, parallelizers, reoptimizing_compiler

from qfast import utils
from qfast.instantiation import nativetool


class QSearchTool ( nativetool.NativeTool ):
    """Synthesize tool built on QSearch's Leap Compiler."""

    def get_maximum_size ( self ):
        """
        The maximum size of a unitary matrix (in qubits) that can be
        decomposed with this tool.

        Returns:
            (int): The qubit count this tool can handle.
        """

        # Larger unitaries can be decomposed with this tool,
        # however, solution quality is best at 3 qubits.
        return 3

    def synthesize ( self, utry ):
        """
        Synthesis function with this tool.

        Args:
            utry (np.ndarray): The unitary to synthesize.

        Returns
            qasm (str): The synthesized QASM output.

        Raises:
            TypeError: If utry is not a valid unitary.

            ValueError: If the utry has invalid dimensions.
        """

        if not utils.is_unitary( utry, tol = 1e-14 ):
            raise TypeError( "utry must be a valid unitary." )

        if utry.shape[0] > 2 ** self.get_maximum_size():
            raise ValueError( "utry has incorrect dimensions." )

        solver = qsearch.solvers.LeastSquares_Jac_SolverNative()
        assembler_style = qsearch.assembler.ASSEMBLY_IBMOPENQASM
        options = qsearch.options.Options()
        options.target = utry
        options.verbosity = 0
        compiler = qsearch.leap_compiler.LeapCompiler( solver = solver )
        output = compiler.compile( options )
        output = qsearch.assembler.assemble( output["structure"],
                                             output["vector"],
                                             assembler_style )
        return output

