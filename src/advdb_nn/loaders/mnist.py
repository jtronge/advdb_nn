"""Util functions for loading MNIST-like data.

See http://yann.lecun.com/exdb/mnist/ for format details.
"""

def read_labels(fname):
    """Read a label file."""
    with open(fname, 'rb') as fp:
        magic_num = int.from_bytes(fp.read(4), 'big')
        item_count = int.from_bytes(fp.read(4), 'big')
        for i in range(item_count):
            label = int.from_bytes(fp.read(1), 'big')
            yield label


def read_images(fname):
    """Read an image file."""
    with open(fname, 'rb') as fp:
        magic_num = int.from_bytes(fp.read(4), 'big')
        img_count = int.from_bytes(fp.read(4), 'big')
        row_count = int.from_bytes(fp.read(4), 'big')
        col_count = int.from_bytes(fp.read(4), 'big')
        for i in range(img_count):
            img = []
            for row in range(row_count):
                img.extend(b for b in fp.read(col_count))
            yield img
