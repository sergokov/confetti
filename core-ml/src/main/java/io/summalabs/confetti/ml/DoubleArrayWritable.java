package io.summalabs.confetti.ml;

import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Writable;

/**
 * @author Sergey Kovalev.
 */
public class DoubleArrayWritable extends ArrayWritable {

    public DoubleArrayWritable(){
        super(DoubleWritable.class);
    }

    public DoubleArrayWritable(double[] values) {
        super(DoubleWritable.class);
        DoubleWritable[] valuesWritable = new DoubleWritable[values.length];
        for (int i = 0; i < values.length; i++) {
            DoubleWritable value = new DoubleWritable(values[i]);
            valuesWritable[i] = value;
        }
        set(valuesWritable);
    }

    public double[] toArray() {
        Writable[] writableValues = get();
        double[] values = new double[writableValues.length];
        for (int i = 0; i < writableValues.length; i++) {
            values[i] = ((DoubleWritable) writableValues[i]).get();
        }
        return values;
    }
}
