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

    public VectorWritable(Vector vector) {
        this.vector = vector;
    }

    @Override
    public void write(DataOutput out) throws IOException {

    }

    @Override
    public void readFields(DataInput in) throws IOException {

    }
}
