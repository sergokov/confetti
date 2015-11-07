package io.summalabs.confetti.ml;

import jdk.nashorn.internal.ir.annotations.Ignore;
import org.apache.commons.io.IOUtils;
import org.junit.Assert;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import org.opencv.core.Core;

import java.io.*;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

/**
 * @author Sergey Kovalev.
 */
public class DescriptorTest {
    private static byte[] image;

    public static <T extends Comparable<T>> int findMinIndexInArray(T[] data) {
        int ret  = 0;
        T tmpVal = data[0];
        for(int ii=0; ii<data.length; ii++) {
            if(tmpVal.compareTo(data[ii])==1) {
                tmpVal=data[ii];
                ret = ii;
            }
        }
        return ret;
    }

    @BeforeClass
    public static void setUp() throws IOException {
        image = IOUtils.toByteArray(DescriptorTest.class.getResourceAsStream("/T001_01.png"));
    }

    @Test
    public void testBuildDescriptor() {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        Descriptor descriptor = new Descriptor(image, Descriptor.DEF_BIN_NUMBER, true);
        double[] value = descriptor.getValue();
        Assert.assertArrayEquals(
                new double[] {0.06855, 0.05535, 0.0846, 0.053925, 0.142925, 0.145025, 0.061375, 0.048175, 0.038025,
                        0.05125, 0.018525, 0.035675, 0.039075, 0.071525, 0.057525, 0.028475},
                value,
                1);
    }
    @Test
    public void testDescriptorClassificationWithiutSpark() {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        class IdxData {
            public IdxData(String path, int id) {
                this.setData(path,id);
            }
            public IdxData() {
                path    = null;
                clsId   = -1;
            }
            void setData(String path, int id) {
                this.path   = path;
                this.clsId  = id;
            }
            public String  path;
            public int     clsId;
        }
        // (1) Prepare working paths:
        Path pathIdx = Paths.get(DescriptorTest.class
                .getResource("/Yandex.Shad_ImVideo_Competition1_anon/comp1_train/cls.txt").getPath());
        Path pathDir = pathIdx.getParent();
        Scanner in = null;
        int PAR_DESCRIPTOR_BINS = Descriptor.DEF_BIN_NUMBER*2;
        int score = 0;
        try {
            // (2) Load info about images and build descriptor Lists:
            String strPathIdx = pathIdx.toAbsolutePath().toString();
            in = new Scanner(new FileReader(strPathIdx));
            ArrayList<IdxData>      arrData = new ArrayList<IdxData>();
            ArrayList<Descriptor>   arrDsc  = new ArrayList<Descriptor>();
            System.out.print("* Load data & build descriptors...");
            while (in.hasNext()) {
                String tmp = in.nextLine();
                String[] tmpSpl = tmp.split(",");
                String tmpPath  = pathDir.resolve(tmpSpl[0]).toString();
                int tmpIdx      = Integer.parseInt(tmpSpl[1]);
                arrData.add(new IdxData(tmpPath, tmpIdx));
                byte[] tmpImageRaw = IOUtils.toByteArray(new FileInputStream(tmpPath));
                Descriptor tmpDsc = new Descriptor(tmpImageRaw, PAR_DESCRIPTOR_BINS, true);
                arrDsc.add(tmpDsc);
            }
            System.out.println(" ... [done]");
            // (3) Calculate distance map: not effective way, because calculations are duplicated
            int numImages = arrData.size();
            Double[] arrDist = new Double[numImages];
            int[] arrNgbhIdx = new int[numImages];
            Double[] tmpDist = new Double[numImages];
            System.out.print("* Calculate NGBH distances...");
            for (int ii=0; ii<numImages; ii++) {
                Descriptor currDsc = arrDsc.get(ii);
                for(int jj = 0; jj<numImages; jj++) {
                    if(ii!=jj) {
                        tmpDist[jj]=currDsc.distL1(arrDsc.get(jj));
                    } else {
                        tmpDist[jj]=Double.MAX_VALUE;
                    }
                }
                //
                int tmpIdxNgbh  = DescriptorTest.findMinIndexInArray(tmpDist);
                arrNgbhIdx[ii]  = arrData.get(tmpIdxNgbh).clsId;
                arrDist[ii]     = tmpDist[tmpIdxNgbh];
            }
            System.out.println(" ... [done]");
            // (4) Check score (number of correct matches):

            for(int ii=0; ii<numImages; ii++) {
                if(arrData.get(ii).clsId==arrNgbhIdx[ii]) {
                    score++;
                }
            }
            double accuracy = 100*(double)score/(double)numImages;
            System.out.println("Accuracy: " + score + "/" + numImages + " : ~" + accuracy + "%");

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        Assert.assertEquals(522,score);
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
