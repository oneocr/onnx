package oneocr.onnx;

public final class Tensor {
    final float[] data;
    final long[] shape;

    Tensor(float[] data, long[] shape) {
        this.data = data;
        this.shape = shape;
    }

    public int dim(int axis) {
        return axis < shape.length ? (int) shape[axis] : 1;
    }

    public int rank() {
        return shape.length;
    }

    public float at(int index) {
        return data[index];
    }

    public float at(int row, int col, int width) {
        return data[row * width + col];
    }

    public int length() {
        return data.length;
    }

    public Tensor plane(int offset, int count) {
        var out = new float[count];
        System.arraycopy(data, offset, out, 0, count);
        return new Tensor(out, new long[]{count});
    }
}
