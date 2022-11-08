// package com.microsoft.diskann;

import java.io.Closeable;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;

public class FileInputStreamMediator implements Closeable {

    private File file;
    private FileInputStream stream;

    public FileInputStreamMediator(final String canonicalPath) throws FileNotFoundException {
        // TODO: add your preconditions here, make sure the file exists, is readable, etc
        // we're not doing that because we're early days yet.
        this.stream = new FileInputStream(new File(canonicalPath));
    }

    public boolean read(final int[] readFroms, final int[] readUntils, final long[] readIntos) throws Exception {
        // these 3 arrays must be the same length
        // the readIntos are currently just the memory we're supposed to read into, however one does that across JNI
        // I'm going to punt on that and instead just print out all the readIntos and untils and we'll figure out
        // how to actually read, using Java IO, <into> some pre-allocated C++ buffers
        // in an absolute worst case scenario we can create our own return list of buffers and do a memcopy
        // from one into another, but I'd prefer not to do that
        if (readFroms.length != readUntils.length || readFroms.length != readIntos.length) {
            throw new Exception("The 3 parallel arrays must be the same size"); // this is a terrible error messaging pattern, but I don't know what control I have over *exceptions*
        }
        for (int i = 0; i < readFroms.length; i++) {
            System.out.println(String.format("I would have read from %d until %d and placed it into %d", readFroms[i], readUntils[i], readIntos[i]));
        }
        return true;
    }

    public void close() throws IOException {
        this.stream.close();
    }

}