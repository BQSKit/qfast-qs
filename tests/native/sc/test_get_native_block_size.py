import tensorflow as tf
import numpy      as np

from qfast.native.sc import get_native_block_size


class TestSCGetNativeBlockSize ( tf.test.TestCase ):

    def test_sc_get_native_block_size ( self ):
        block_size = get_native_block_size()
        self.assertEqual( block_size, 3 )


if __name__ == '__main__':
    tf.test.main()
