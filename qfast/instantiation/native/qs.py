"""
This module implements QSearch's Leap Compiler
as a native tool plugin to QFAST.
"""

import qsearch
from qsearch import options, assemblers, leap_compiler, post_processing, multistart_solvers

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
        
        # Pass options into qsearch, being maximally quiet, and set the target to utry
        opts = options.Options()
        opts.target = utry
        opts.verbosity = 0
        opts.write_to_stdout = False
        opts.reoptimize_size = 7
        opts.solver = multistart_solvers.MultiStart_Solver( 24 )
        # use the LEAP compiler, which scales better than normal qsearch
        compiler = leap_compiler.LeapCompiler()
        output = compiler.compile( opts )
        # LEAP requires some post-processing
        post_processor = post_processing.LEAPReoptimizing_PostProcessor()
        output = post_processor.post_process_circuit( output, opts )
        output = assemblers.ASSEMBLER_IBMOPENQASM.assemble( output )
        return output

