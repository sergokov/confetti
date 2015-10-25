package io.summalabs.confetti.image;

import org.apache.commons.cli.*;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocalFileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.hdfs.DistributedFileSystem;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Arrays;

public class ImageToSequenceImport {

    private static final Options options = new Options();
    private static final Parser parser = (Parser) new BasicParser();

    static {
        options.addOption("f", "force", false, "force overwrite if output file already exists");
    }

    private static void usage() {
        // usage
        HelpFormatter formatter = new HelpFormatter();
        formatter.printHelp("ImageToSequenceImport.jar [options] <image directory> <output file>", options);
        System.exit(0);
    }

    public static void main(String[] args) throws IOException {

        // Attempt to parse the command line arguments
        CommandLine line = null;
        try {
            line = parser.parse(options, args);
        } catch (ParseException exp) {
            usage();
        }
        if (line == null) {
            usage();
        }

        String[] leftArgs = line.getArgs();
        if (leftArgs.length != 2) {
            usage();
        }

        String imageDir = leftArgs[0];
        String outputFile = leftArgs[1];

        boolean overwrite = false;
        if (line.hasOption("f")) {
            overwrite = true;
        }

        System.out.println("Input image directory: " + imageDir);
        System.out.println("Output file: " + outputFile);
        System.out.println("Overwrite file if it exists: " + (overwrite ? "true" : "false"));
        System.out.println("Use specific config params");

        File folder = new File(imageDir);
        File[] files = folder.listFiles();
        Arrays.sort(files);

        if (files == null) {
            System.err.println(String.format("Did not find any files in the local FS directory [%s]", imageDir));
            System.exit(0);
        }

        Configuration conf = new Configuration();
        conf.set("fs.hdfs.impl", DistributedFileSystem.class.getName());
        conf.set("fs.file.impl", LocalFileSystem.class.getName());
        conf.addResource(new Path("/usr/local/hadoop/etc/hadoop/core-site.xml"));

        FileSystem fs = FileSystem.get(conf);
        SequenceFile.Writer writer =
                SequenceFile.createWriter(fs, conf, new Path(outputFile), Text.class, BytesWritable.class);
        try {
            for (File file : files) {
                FileInputStream in = new FileInputStream(file);
                byte buffer[] = new byte[in.available()];
                in.read(buffer);
                writer.append(new Text(file.getName()), new BytesWritable(buffer));
                in.close();
                System.out.println(" ** added: " + file.getName());
            }
            System.out.println("Created output sequence file: " + outputFile);
        } catch (Exception e) {
            System.out.println("Exception MESSAGE = " + e.getMessage());
        } finally {
            IOUtils.closeStream(writer);
        }
    }

}
