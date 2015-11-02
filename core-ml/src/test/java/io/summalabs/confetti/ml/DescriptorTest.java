package io.summalabs.confetti.ml;

import jdk.nashorn.internal.ir.annotations.Ignore;
import org.apache.commons.io.IOUtils;
import org.junit.Assert;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import org.opencv.core.Core;

import java.io.IOException;

/**
 * @author Sergey Kovalev.
 */
public class DescriptorTest {
    private static byte[] image;

    @BeforeClass
    public static void setUp() throws IOException {
        image = IOUtils.toByteArray(DescriptorTest.class.getResourceAsStream("/T001_01.png"));
    }

//    @Test
    public void testBuildDescriptor() {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        Descriptor descriptor = new Descriptor(image, Descriptor.DEF_BIN_NUMBER, true);
        double[] value = descriptor.getValue();
        Assert.assertArrayEquals(
                new double[] {0, 0, 0, 1, 2},
                value,
                1);
    }

    @Test
    public void testDistL1() {
        Descriptor descriptorOne = new Descriptor(new double[] {0, 0, 0, 1, 2});
        Descriptor descriptorTwo = new Descriptor(new double[] {1, 2, 0, 0, 0});
        double distL1 = descriptorOne.distL1(descriptorTwo);
        Assert.assertTrue(distL1 == 6);
    }

    @Test(expected = NullPointerException.class)
    public void testExceptionOnNullArgument() {
        double[] arg = null;
        new Descriptor(arg);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testExceptionOnEmptyArgument() {
        new Descriptor(new double[]{});
    }
}
