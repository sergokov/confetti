package io.summalabs.confetti.ml;

import org.apache.hadoop.io.Writable;
import org.apache.spark.mllib.linalg.Vector;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * @author Sergey Kovalev.
 */
public class VectorWritable implements Writable {
    private Vector vector;

    public VectorWritable() {
    }

    public VectorWritable(Vector vector) {
        this.vector = vector;
    }

    public void write(DataOutput out) throws IOException {
        double[] vector = this.vector.toArray();
        out.writeInt(vector.length);
        for(double value : vector) {
            out.writeDouble(value);
        }
    }

    public void readFields(DataInput in) throws IOException {
        double[] vector = new double[in.readInt()];
        for (int i = 0; i < vector.length; i++) {
            vector[i]  = in.readDouble();
        }

    }
}
